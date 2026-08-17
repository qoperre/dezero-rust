"""Generate golden fixtures from the Python DeZero reference (vendor/dezero-python).

Run with a venv that has numpy installed:
    python tests/parity/gen_fixtures.py

Writes one fixture per ported unit to tests/parity/fixtures/<unit>.json as
{"input": [...], "output": [...]} -- plain nested lists so the Rust side has
no extra parsing dependency beyond serde_json.
"""
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "vendor" / "dezero-python"))

import numpy as np
from dezero import Variable

FIXTURES_DIR = pathlib.Path(__file__).resolve().parent / "fixtures"
FIXTURES_DIR.mkdir(exist_ok=True)


def write_fixture(name: str, payload: dict) -> None:
    (FIXTURES_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {name}.json")


def gen_square() -> None:
    # Mirrors DeZero book step01 (Variable box) + step02 (Square function).
    np.random.seed(0)
    x = np.random.randn(2, 3)
    y = Variable(x) ** 2
    write_fixture("square", {"input": x.tolist(), "output": y.data.tolist()})


if __name__ == "__main__":
    gen_square()
