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

if __name__ == "__main__":
    main()
