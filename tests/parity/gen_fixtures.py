"""Generate golden fixtures from the Python DeZero reference (vendor/dezero-python).

Run with an interpreter that has numpy (and matplotlib, which dezero imports):
    python tests/parity/gen_fixtures.py

## Fixture format

Every array is stored rank-generically as `{"shape": [...], "data": [flat]}`,
in C order. This handles 0-d scalars (`shape: []`) through the 4-d tensors the
CNN steps will need, without the Rust side guessing at nesting depth.

Fixtures are deterministic: inputs are either literal or drawn under a fixed
np.random.seed. Never rely on seeding matching between numpy and Rust's `rand`
-- always ship the explicit input array (see docs/ARCHITECTURE.md).
"""

import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "vendor" / "dezero-python"))

import numpy as np  # noqa: E402
from dezero import Variable  # noqa: E402
import dezero.functions as F  # noqa: E402

FIXTURES_DIR = pathlib.Path(__file__).resolve().parent / "fixtures"
FIXTURES_DIR.mkdir(exist_ok=True)


def arr(a):
    """Serialise an ndarray rank-generically."""
    a = np.asarray(a, dtype=np.float64)
    return {"shape": list(a.shape), "data": a.ravel(order="C").tolist()}


def write_fixture(name, payload):
    (FIXTURES_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {name}.json")


def unary(name, fn, x):
    """Forward + backward fixture for a single-input function."""
    v = Variable(np.array(x, dtype=np.float64))
    y = fn(v)
    y.backward()
    write_fixture(
        name,
        {"input": arr(x), "output": arr(y.data), "grad": arr(v.grad.data)},
    )


def binary(name, fn, x0, x1):
    """Forward + backward fixture for a two-input function."""
    a = Variable(np.array(x0, dtype=np.float64))
    b = Variable(np.array(x1, dtype=np.float64))
    y = fn(a, b)
    y.backward()
    write_fixture(
        name,
        {
            "input0": arr(x0),
            "input1": arr(x1),
            "output": arr(y.data),
            "grad0": arr(a.grad.data),
            "grad1": arr(b.grad.data),
        },
    )


# --- step 24: the book's benchmark optimisation functions -----------------


def sphere(a, b):
    return a**2 + b**2


def matyas(a, b):
    return 0.26 * (a**2 + b**2) - 0.48 * a * b


def goldstein(a, b):
    return (
        1 + (a + b + 1) ** 2 * (19 - 14 * a + 3 * a**2 - 14 * b + 6 * a * b + 3 * b**2)
    ) * (
        30
        + (2 * a - 3 * b) ** 2
        * (18 - 32 * a + 12 * a**2 + 48 * b - 36 * a * b + 27 * b**2)
    )


def main():
    np.random.seed(0)
    x = np.random.randn(2, 3)
    # Keep exp inputs modest so float64 comparisons stay well-conditioned.
    small = np.array([[-1.0, -0.5, 0.0], [0.5, 1.0, 1.5]])
    pos = np.array([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]])

    # --- step 02: Square (forward only; the original fixture) ------------
    write_fixture("square", {"input": arr(x), "output": arr((Variable(x) ** 2).data)})

    # --- steps 03-08: exp, composition, backprop -------------------------
    unary("exp", F.exp, small)
    unary("square_backward", lambda v: v**2, x)
    unary("composed_sq_exp_sq", lambda v: (F.exp(v**2)) ** 2, small)

    # --- steps 11-14: add, and the repeated-variable accumulation case ----
    binary("add", lambda a, b: a + b, x, np.random.randn(2, 3))
    unary("add_same_var", lambda v: v + v, x)  # dy/dx == 2

    # --- steps 20-22: the full arithmetic operator set -------------------
    binary("mul", lambda a, b: a * b, x, np.random.randn(2, 3))
    binary("sub", lambda a, b: a - b, x, np.random.randn(2, 3))
    binary("div", lambda a, b: a / b, x, pos)
    unary("neg", lambda v: -v, x)
    unary("pow3", lambda v: v**3, x)
    unary("composite_arith", lambda v: (v * v + v) / 2.0 - v, x)

    # --- step 24: deeply nested compositions -----------------------------
    # 0-d (the book's own case) and 2-d, so the ops are exercised at both ranks.
    for fn_name, fn in (("sphere", sphere), ("matyas", matyas), ("goldstein", goldstein)):
        binary(f"{fn_name}_scalar", fn, np.array(1.0), np.array(1.0))
        binary(f"{fn_name}_2d", fn, x, np.random.randn(2, 3))

    # --- step 27: sin, and its Taylor-series approximation ---------------
    unary("sin", F.sin, small)
    unary("cos", F.cos, small)
    unary("tanh", F.tanh, small)

    gen_higher_order()
    gen_newton()
    gen_tensor_ops()
    gen_broadcast_matrix()
    gen_nn()
    gen_datasets()


def gen_higher_order():
    """Steps 33-35: nth derivatives via repeated create_graph backward."""
    # sin, to 4th derivative (step 34's plot, as numbers).
    xs = np.linspace(-7.0, 7.0, 21)
    v = Variable(xs.copy())
    y = F.sin(v)
    y.backward(create_graph=True)
    derivs = [arr(y.data)]
    for _ in range(3):
        derivs.append(arr(v.grad.data))
        gx = v.grad
        v.cleargrad()
        gx.backward(create_graph=True)
    write_fixture("sin_higher_order", {"input": arr(xs), "derivatives": derivs})

    # tanh, to 3rd derivative (step 35), at 0-d like the book.
    t = np.array(1.0)
    v = Variable(t.copy())
    y = F.tanh(v)
    y.backward(create_graph=True)
    derivs = [arr(y.data)]
    for _ in range(2):
        derivs.append(arr(v.grad.data))
        gx = v.grad
        v.cleargrad()
        gx.backward(create_graph=True)
    write_fixture("tanh_higher_order", {"input": arr(t), "derivatives": derivs})

    # y = x^4 - 2x^2 -- step 33's function, first and second derivative.
    z = np.array(2.0)
    v = Variable(z.copy())
    y = v**4 - 2 * v**2
    y.backward(create_graph=True)
    gx = v.grad
    first = arr(gx.data)
    v.cleargrad()
    gx.backward()
    write_fixture(
        "quartic_second_deriv",
        {"input": arr(z), "output": arr(y.data), "grad": first, "grad2": arr(v.grad.data)},
    )


def gen_newton():
    """Step 33: 10 Newton iterations on y = x^4 - 2x^2 from x=2."""
    v = Variable(np.array(2.0))
    trace = [float(v.data)]
    for _ in range(10):
        y = v**4 - 2 * v**2
        v.cleargrad()
        y.backward(create_graph=True)
        gx = v.grad
        v.cleargrad()
        gx.backward()
        gx2 = v.grad
        v.data -= gx.data / gx2.data
        trace.append(float(v.data))
    write_fixture("newton_quartic", {"start": 2.0, "iterations": 10, "trace": trace})



def gen_tensor_ops():
    """Steps 38-39: reshape, transpose, sum (axis/keepdims combinations)."""
    np.random.seed(1)
    x2 = np.random.randn(2, 3)
    x3 = np.random.randn(2, 3, 4)

    # --- reshape ---------------------------------------------------------
    for tag, src, shape in (
        ("2d_to_1d", x2, (6,)),
        ("2d_to_3d", x2, (1, 2, 3)),
        ("3d_to_2d", x3, (6, 4)),
        ("to_scalarish", np.array([[7.0]]), (1,)),
    ):
        v = Variable(src.copy())
        y = F.reshape(v, shape)
        y.backward()
        write_fixture(
            f"reshape_{tag}",
            {
                "input": arr(src),
                "shape": list(shape),
                "output": arr(y.data),
                "grad": arr(v.grad.data),
            },
        )

    # --- transpose -------------------------------------------------------
    v = Variable(x2.copy())
    y = F.transpose(v)
    y.backward()
    write_fixture(
        "transpose_2d",
        {"input": arr(x2), "output": arr(y.data), "grad": arr(v.grad.data)},
    )

    # --- sum, over every axis/keepdims combination that matters ----------
    cases = [
        ("all", x2, None, False),
        ("all_keepdims", x2, None, True),
        ("axis0", x2, 0, False),
        ("axis1", x2, 1, False),
        ("axis0_keepdims", x2, 0, True),
        ("axis1_keepdims", x2, 1, True),
        ("3d_axis0", x3, 0, False),
        ("3d_axis1", x3, 1, False),
        ("3d_axis2", x3, 2, False),
        ("3d_axis1_keepdims", x3, 1, True),
    ]
    for tag, src, axis, keepdims in cases:
        v = Variable(src.copy())
        y = F.sum(v, axis=axis, keepdims=keepdims)
        y.backward()
        write_fixture(
            f"sum_{tag}",
            {
                "input": arr(src),
                "axis": axis,
                "keepdims": keepdims,
                "output": arr(y.data),
                "grad": arr(v.grad.data),
            },
        )


def gen_broadcast_matrix():
    """Step 40: broadcast_to / sum_to, plus broadcasting through arithmetic.

    This is the widest matrix in the suite on purpose -- numpy and ndarray
    disagree at the edges, and a wrong gradient here is silent.
    """
    np.random.seed(2)

    # --- broadcast_to ----------------------------------------------------
    bcases = [
        ("scalar_to_2d", np.array(3.0), (2, 3)),
        ("row_to_2d", np.random.randn(3), (2, 3)),
        ("col_to_2d", np.random.randn(2, 1), (2, 3)),
        ("1_to_3d", np.random.randn(1, 3, 1), (2, 3, 4)),
        ("noop", np.random.randn(2, 3), (2, 3)),
    ]
    for tag, src, shape in bcases:
        v = Variable(src.copy())
        y = F.broadcast_to(v, shape)
        # Back-propagate from a scalar: when shape already matches, Python's
        # broadcast_to returns x itself with no graph node, so y.backward()
        # would crash. Seeding via sum() is mathematically identical (both
        # seed ones over y) and covers the identity case too.
        F.sum(y).backward()
        write_fixture(
            f"broadcast_to_{tag}",
            {
                "input": arr(src),
                "shape": list(shape),
                "output": arr(y.data),
                "grad": arr(v.grad.data),
            },
        )

    # --- sum_to ----------------------------------------------------------
    scases = [
        ("2d_to_row", np.random.randn(2, 3), (1, 3)),
        ("2d_to_col", np.random.randn(2, 3), (2, 1)),
        ("2d_to_scalar", np.random.randn(2, 3), (1, 1)),
        ("3d_to_2d", np.random.randn(2, 3, 4), (3, 4)),
        ("3d_to_1", np.random.randn(2, 3, 4), (1, 1, 1)),
        ("noop", np.random.randn(2, 3), (2, 3)),
    ]
    for tag, src, shape in scases:
        v = Variable(src.copy())
        y = F.sum_to(v, shape)
        F.sum(y).backward()
        write_fixture(
            f"sum_to_{tag}",
            {
                "input": arr(src),
                "shape": list(shape),
                "output": arr(y.data),
                "grad": arr(v.grad.data),
            },
        )

    # --- broadcasting *through* the arithmetic ops (retires divergence 2) -
    acases = [
        ("add_row", "add", np.random.randn(2, 3), np.random.randn(3)),
        ("add_col", "add", np.random.randn(2, 3), np.random.randn(2, 1)),
        ("add_scalar_arr", "add", np.random.randn(2, 3), np.array(2.0)),
        ("mul_row", "mul", np.random.randn(2, 3), np.random.randn(3)),
        ("mul_col", "mul", np.random.randn(2, 3), np.random.randn(2, 1)),
        ("sub_row", "sub", np.random.randn(2, 3), np.random.randn(3)),
        ("div_row", "div", np.random.randn(2, 3), np.abs(np.random.randn(3)) + 0.5),
        # reversed operand order: the smaller array on the left
        ("add_rev", "add", np.random.randn(3), np.random.randn(2, 3)),
        ("sub_rev", "sub", np.random.randn(3), np.random.randn(2, 3)),
        ("3d_add", "add", np.random.randn(2, 3, 4), np.random.randn(3, 4)),
    ]
    ops = {
        "add": lambda a, b: a + b,
        "mul": lambda a, b: a * b,
        "sub": lambda a, b: a - b,
        "div": lambda a, b: a / b,
    }
    for tag, opname, a, b in acases:
        binary(f"bcast_{tag}", ops[opname], a, b)

def gen_nn():
    """Steps 41-48: matmul, linear, activations, losses, and a full training run.

    Every weight is shipped explicitly. Rust's `rand` will never reproduce
    numpy's stream, so "seed both sides" is not an option -- see
    docs/ARCHITECTURE.md.
    """
    np.random.seed(3)

    # --- step 41: matmul -------------------------------------------------
    for tag, a_shape, b_shape in (("2x3_3x4", (2, 3), (3, 4)), ("1x3_3x1", (1, 3), (3, 1))):
        a = np.random.randn(*a_shape)
        b = np.random.randn(*b_shape)
        binary(f"matmul_{tag}", F.matmul, a, b)

    # --- step 42-43: linear, sigmoid, relu, mean_squared_error ------------
    x = np.random.randn(4, 3)
    W = np.random.randn(3, 2)
    b = np.random.randn(2)

    v_x, v_W, v_b = Variable(x.copy()), Variable(W.copy()), Variable(b.copy())
    y = F.linear(v_x, v_W, v_b)
    F.sum(y).backward()
    write_fixture(
        "linear",
        {
            "x": arr(x), "W": arr(W), "b": arr(b),
            "output": arr(y.data),
            "gx": arr(v_x.grad.data), "gW": arr(v_W.grad.data), "gb": arr(v_b.grad.data),
        },
    )

    unary("sigmoid", F.sigmoid, np.array([[-2.0, -0.5, 0.0], [0.5, 1.0, 2.0]]))
    unary("relu", F.relu, np.array([[-2.0, -0.5, 0.0], [0.5, 1.0, 2.0]]))

    p_, t_ = np.random.randn(4, 2), np.random.randn(4, 2)
    binary("mean_squared_error", F.mean_squared_error, p_, t_)

    # --- step 47: softmax and softmax_cross_entropy -----------------------
    logits = np.random.randn(4, 3)
    unary("softmax", lambda v: F.softmax(v), logits)

    labels = np.array([0, 2, 1, 0])
    v = Variable(logits.copy())
    loss = F.softmax_cross_entropy(v, labels)
    loss.backward()
    write_fixture(
        "softmax_cross_entropy",
        {
            "logits": arr(logits), "labels": labels.tolist(),
            "output": arr(loss.data), "grad": arr(v.grad.data),
        },
    )

    gen_training_run()


def gen_training_run():
    """Step 42/44-46: a full two-layer training run, weights pinned.

    This is the integration fixture -- if Layer/Parameter/Optimizer/loss all
    work together, the loss trace matches step for step.
    """
    import dezero
    from dezero import Model, optimizers
    import dezero.layers as L

    np.random.seed(4)
    x = np.random.rand(20, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(20, 1)

    W1 = np.random.randn(1, 10) * 0.01
    b1 = np.zeros(10)
    W2 = np.random.randn(10, 1) * 0.01
    b2 = np.zeros(1)

    class TwoLayer(Model):
        def __init__(self):
            super().__init__()
            self.l1 = L.Linear(10, in_size=1)
            self.l2 = L.Linear(1, in_size=10)

        def forward(self, t):
            return self.l2(F.sigmoid(self.l1(t)))

    model = TwoLayer()
    model.l1.W.data = W1.copy()
    model.l1.b.data = b1.copy()
    model.l2.W.data = W2.copy()
    model.l2.b.data = b2.copy()

    opt = optimizers.SGD(lr=0.2).setup(model)

    losses = []
    for _ in range(50):
        pred = model(Variable(x))
        loss = F.mean_squared_error(pred, Variable(y))
        model.cleargrads()
        loss.backward()
        opt.update()
        losses.append(float(loss.data))

    write_fixture(
        "training_two_layer",
        {
            "x": arr(x), "y": arr(y),
            "W1": arr(W1), "b1": arr(b1), "W2": arr(W2), "b2": arr(b2),
            "lr": 0.2, "iterations": 50,
            "losses": losses,
            "final_W1": arr(model.l1.W.data), "final_b1": arr(model.l1.b.data),
            "final_W2": arr(model.l2.W.data), "final_b2": arr(model.l2.b.data),
        },
    )


def gen_datasets():
    """Steps 48-50: spiral classification, DataLoader batching, optimizer hooks."""
    import dezero
    from dezero import optimizers
    from dezero.models import MLP
    from dezero.datasets import Spiral
    from dezero import DataLoader

    # --- step 48: the spiral dataset, shipped explicitly -----------------
    # get_spiral() uses np.random.randn + np.random.permutation, so the data
    # itself must travel with the fixture; a seed would not reproduce it.
    train = Spiral(train=True)
    xs = np.array([train[i][0] for i in range(len(train))], dtype=np.float64)
    ts = np.array([train[i][1] for i in range(len(train))], dtype=int)
    write_fixture("spiral_data", {"x": arr(xs), "t": ts.tolist()})

    # --- step 48: a full classification training run ---------------------
    np.random.seed(7)
    W1 = np.random.randn(2, 10) * 0.1
    b1 = np.zeros(10)
    W2 = np.random.randn(10, 3) * 0.1
    b2 = np.zeros(3)

    model = MLP((10, 3))
    # Force the lazily-shaped weights into existence, then pin them.
    model(Variable(xs[:1].copy()))
    model.l0.W.data = W1.copy(); model.l0.b.data = b1.copy()
    model.l1.W.data = W2.copy(); model.l1.b.data = b2.copy()

    opt = optimizers.SGD(lr=1.0).setup(model)
    losses = []
    for _ in range(300):
        pred = model(Variable(xs))
        loss = F.softmax_cross_entropy(pred, ts)
        model.cleargrads()
        loss.backward()
        opt.update()
        losses.append(float(loss.data))

    write_fixture(
        "spiral_training",
        {
            "W1": arr(W1), "b1": arr(b1), "W2": arr(W2), "b2": arr(b2),
            "lr": 1.0, "iterations": 300, "losses": losses,
            "final_W1": arr(model.l0.W.data), "final_b1": arr(model.l0.b.data),
            "final_W2": arr(model.l1.W.data), "final_b2": arr(model.l1.b.data),
        },
    )

    # --- step 50: DataLoader batching, shuffle off so it is deterministic -
    loader = DataLoader(train, batch_size=32, shuffle=False)
    batches = []
    for bx, bt in loader:
        batches.append({"x": arr(bx), "t": np.asarray(bt).tolist()})
    write_fixture(
        "dataloader_batches",
        {"batch_size": 32, "data_size": len(train), "batches": batches},
    )

    # --- step 50: optimizer hooks ----------------------------------------
    gen_optimizer_hooks()


def gen_optimizer_hooks():
    """Step 50: WeightDecay and ClipGrad applied to a known gradient."""
    from dezero import optimizers
    from dezero.core import Parameter

    def fresh():
        p1 = Parameter(np.array([[1.0, -2.0], [3.0, -4.0]]))
        p2 = Parameter(np.array([0.5, -1.5]))
        p1.grad = Variable(np.array([[0.1, 0.2], [0.3, 0.4]]))
        p2.grad = Variable(np.array([1.0, -2.0]))
        return p1, p2

    p1, p2 = fresh()
    optimizers.WeightDecay(0.1)([p1, p2])
    write_fixture(
        "hook_weight_decay",
        {
            "rate": 0.1,
            "p1": arr([[1.0, -2.0], [3.0, -4.0]]), "p2": arr([0.5, -1.5]),
            "g1_in": arr([[0.1, 0.2], [0.3, 0.4]]), "g2_in": arr([1.0, -2.0]),
            "g1_out": arr(p1.grad.data), "g2_out": arr(p2.grad.data),
        },
    )

    for tag, max_norm in (("clips", 0.5), ("noop", 100.0)):
        p1, p2 = fresh()
        optimizers.ClipGrad(max_norm)([p1, p2])
        write_fixture(
            f"hook_clip_grad_{tag}",
            {
                "max_norm": max_norm,
                "g1_in": arr([[0.1, 0.2], [0.3, 0.4]]), "g2_in": arr([1.0, -2.0]),
                "g1_out": arr(p1.grad.data), "g2_out": arr(p2.grad.data),
            },
        )


if __name__ == "__main__":
    main()
