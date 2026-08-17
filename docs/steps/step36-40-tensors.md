# Steps 36–40 — tensor operations and broadcasting

**Status:** done

## The bug the fixture matrix was built to catch

`ndarray`'s binary operators are **asymmetric**: `&a + &b` stretches only the
*right* operand onto the left one's shape. So `matrix + row` works and
`row + matrix` **panics**. numpy broadcasts symmetrically.

This is exactly why the `bcast_add_rev` / `bcast_sub_rev` fixtures exist —
the smaller operand on the left. Had the port simply forwarded to `ndarray`'s
operators, half the broadcasting cases would have been silently wrong or
crashed only in user code.

Resolved by never using `ndarray`'s operators for binary ops. `zip_map`
computes the numpy broadcast shape, calls `.broadcast()` on **both** operands,
and combines with `Zip::map_collect`. Equal shapes take a fast path.

## What was added

| Module | Contents |
|---|---|
| `utils/shape.rs` | array-level `sum_to`, `reshape_sum_backward`, `normalize_axis`, `broadcast_shape` |
| `functions/shape.rs` | `Reshape`/`reshape`, `Transpose`/`transpose`, `Variable::{reshape, transpose, t}` |
| `functions/reduce.rs` | `Axes`, `Sum`/`sum`/`sum_all`, `SumTo`/`sum_to`, `BroadcastTo`/`broadcast_to`, `Variable::sum` |

`utils::sum_to` and `reshape_sum_backward` are statement-by-statement ports of
Python's, as planned — they are subtle and the fixtures catch drift.

`broadcast_to` and `sum_to` are mutually inverse in backward, and both
reproduce Python's identity case (returning the input unchanged, with no graph
node, when the shape already matches).

### Modelling `sum`'s axis

Python accepts `None | int | tuple`. Rust gets a two-variant enum with `From`
impls so all three spellings stay call-site-identical:

```rust
pub enum Axes { All, Only(Vec<isize>) }

sum(&x, Axes::All, false)    // F.sum(x)
sum(&x, 1, true)             // F.sum(x, axis=1, keepdims=True)
sum(&x, [0, 2], false)       // F.sum(x, axis=(0, 2))
sum(&x, -1, false)           // negative axes normalised
```

`Axes::resolve` rejects duplicates, mirroring numpy's `ValueError`.

## Divergences 2 and 3 retired

Both are gone, verified by `grep` returning nothing for `TODO(step-40)`,
`require_same_shape`, or `constant_like`:

- Arithmetic ops now broadcast, and each backward ends in `fold_broadcast` —
  Python's `if x0.shape != x1.shape: gx = sum_to(gx, x.shape)` verbatim.
- Scalars stay 0-d, matching Python's `as_array(2.0)`. `Square`/`Pow`/`Tanh`
  backward use 0-d constants too, which is *closer* to the reference than
  before.

One pre-existing test was deliberately removed:
`mismatched_shapes_are_rejected_until_step_40`, whose entire premise was
divergence 2. It is replaced by six broadcasting tests including
`incompatible_shapes_are_still_rejected`, which pins the *new* panic for
genuinely non-broadcastable shapes.

## Other numpy/`ndarray` gaps, and how each was closed

| Gap | Resolution |
|---|---|
| No multi-axis reduction | Descending loop of `sum_axis` + `insert_axis` (descending so smaller indices stay valid). Summation order differs from numpy's pairwise reduction at ~1e-16 — well inside tolerance. |
| No `squeeze(axes)` | `lead` × `index_axis_move(Axis(0), 0)` |
| Non-contiguous reshape | `to_shape(IxDyn(..))` — borrows when it can, copies when it can't, which is numpy's behaviour. `into_shape_with_order` would have errored where numpy silently copies. Pinned by a test that reshapes a *transposed* variable and checks logical C order. |
| Zero-length axes, 0-d→n-d, rank-0 `Zip` | Checked and found to agree; no workaround needed. |

## Coverage beyond the shipped fixtures

All 61 fixtures are referenced (the fixture directory was diffed against the
names in `parity_core.rs`; nothing is orphaned).

The most valuable extra case has no fixture and is now a permanent test:
`(2,1,4) * (3,1)`, where **both** operands are stretched and each gradient
folds over a *different* axis set. Every shipped fixture has one operand
already at the output shape, so this case would otherwise have gone untested.
It reproduces Python bit-for-bit.

## Verification

```
cargo test    210 passed  (149 unit + 35 parity + 26 doctests), up from 122
cargo clippy --all-targets -- -D warnings   clean
cargo fmt --all --check                     clean
cargo doc --no-deps                         0 warnings
```

## Deliberately left undone, so they don't become silent gaps

- **`transpose(axes=...)`** — the permutation form is not ported. `transpose`
  reverses all axes (numpy `.T`), which is already correct at rank 3+. No step
  through 40 uses the permutation form, and its backward needs `argsort` of
  the inverse permutation. Divergence 9.
- **`reshape(-1)`** — no inferred-dimension placeholder. `flatten` is the only
  caller that needs it and can compute the length from `Variable::size`.
  Divergence 10.

## Next

Steps 41–48: `matmul`, linear regression, then the `Layer`/`Parameter`/
`Optimizer` abstractions — the second-most consequential design decision after
the core graph.
