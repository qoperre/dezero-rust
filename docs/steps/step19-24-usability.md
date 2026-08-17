# Steps 19–24 — usability

**Status:** done

## What was already there

Steps 19–23 were satisfied by the core built in steps 1–18, because the core
was deliberately built in its final shape rather than the book's incremental
one. Verified rather than assumed:

| Step | Requirement | Where |
|---|---|---|
| 19 | `shape`/`ndim`/`size`/`len`/`name`, `repr` | `core/variable.rs` — all present, plus `Display`/`Debug` matching Python's `variable(...)` |
| 20 | operator overloading `+`, `*` | `core/ops.rs` `impl_binary_operator!` |
| 21 | mixing `Variable` with scalars | `V op f64` and `f64 op V` impls (8 per operator) |
| 22 | `-`, `/`, `**`, unary neg, reversed ops | `Sub`/`Div`/`Pow`/`Neg` |
| 23 | package layout | `core/`, `functions/`, `utils/` module tree |

Python's `dtype` property has no counterpart: the port is `f64`-only
(divergence 1). Python's `__array_priority__` is deliberately not ported
(divergence 7) — it exists to stop numpy's `ndarray.__mul__` from beating
`Variable.__rmul__`, and since we never overload operators on `ArrayD` there
is no competing dispatcher.

## Step 24 — the real work

The book's three benchmark optimisation functions, as parity tests:

- **sphere** `x² + y²`
- **matyas** `0.26(x² + y²) − 0.48xy`
- **Goldstein-Price** — the deeply nested one

These add no library code; they are pure compositions of existing ops. Their
value is as a stress test of the generation-ordered backward traversal, which
is exactly what deep nesting exercises. Each runs at **0-d** (the book's own
case, `Variable(np.array(1.0))`) and at **2-d**.

Goldstein-Price at `(1, 1)` reproduces the book's published gradients:
`dx = -5376`, `dy = 8064`.

## Fixture format change

Fixtures moved from nested `Vec<Vec<f64>>` to rank-generic
`{"shape": [...], "data": [flat]}` in C order.

This was forced by step 24's 0-d scalars (`shape: []` has no nesting to
infer), and it is what the CNN steps will need for 4-d tensors. Changing it
now cost one file; changing it at step 57 would have cost every fixture and
every test.

`crates/dezero/tests/parity_square.rs` was folded into `parity_core.rs` at
the same time — two files with duplicate helpers, no longer worth keeping
apart. The original `square.json` fixture and a test pinning `square()` (not
just the `x * x` spelling) both survive the move.

## Verification

```
cargo test    98 passed  (70 unit + 18 parity + 10 doctests)
cargo clippy --all-targets -- -D warnings   clean
cargo fmt --all --check                     clean
```

17 fixtures now compared against Python, forward at `rtol=1e-5`, gradients at
`rtol=1e-4`.

## Next

Step 25–26: DOT-graph emission. Then the first genuinely hard group, 27–35
(higher-order derivatives), which is the acid test of the architecture.
