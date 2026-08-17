"""Generate golden fixtures from the Python DeZero reference (vendor/dezero-python).

Run with an interpreter that has numpy (and matplotlib, which dezero imports):
    python tests/parity/gen_fixtures.py

Each fixture lands in tests/parity/fixtures/<name>.json as plain nested lists,
so the Rust side needs nothing beyond serde_json.

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


def write_fixture(name, payload):
    (FIXTURES_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {name}.json")


def unary(name, fn, x):
    """Forward + backward fixture for a single-input function."""
    v = Variable(x.copy())
    y = fn(v)
    y.backward()
    write_fixture(
        name,
        {
            "input": x.tolist(),
            "output": y.data.tolist(),
            "grad": v.grad.data.tolist(),
        },
    )


def binary(name, fn, x0, x1):
    """Forward + backward fixture for a two-input function."""
    a, b = Variable(x0.copy()), Variable(x1.copy())
    y = fn(a, b)
    y.backward()
    write_fixture(
        name,
        {
            "input0": x0.tolist(),
            "input1": x1.tolist(),
            "output": y.data.tolist(),
            "grad0": a.grad.data.tolist(),
            "grad1": b.grad.data.tolist(),
        },
    )


def main():
    np.random.seed(0)
    x = np.random.randn(2, 3)
    # Keep exp inputs modest so float64 comparisons stay well-conditioned.
    small = np.array([[-1.0, -0.5, 0.0], [0.5, 1.0, 1.5]])
    pos = np.array([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]])

    # --- step 02: Square -------------------------------------------------
    # Historical fixture: forward only, kept at its original shape so the
    # existing parity_square test keeps passing unchanged.
    write_fixture("square", {"input": x.tolist(), "output": (Variable(x) ** 2).data.tolist()})

    # --- steps 03-08: exp, composition, backprop -------------------------
    unary("exp", F.exp, small)
    unary("square_backward", lambda v: v**2, x)

    # step03's composed function: y = square(exp(square(x)))
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

    # A composite exercising several ops and a reused variable at once.
    unary("composite_arith", lambda v: (v * v + v) / 2.0 - v, x)


if __name__ == "__main__":
    main()
