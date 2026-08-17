# Intentional divergences from Python DeZero

This port aims for **numerical** equivalence with `vendor/dezero-python`, not
line-by-line structural equivalence. Where Rust makes a different choice
correct or necessary, it is recorded here rather than silently absorbed.

Parity tests assert against the Python reference within
`rtol=1e-5, atol=1e-8` (numpy `allclose` defaults); anything that cannot hold
to that is a divergence and belongs in this file.

| # | Area | Python | Rust port | Why |
|---|------|--------|-----------|-----|
| 1 | Element type | `np.float64` by default, but any dtype | `f64` only | DeZero's book code is float64 throughout. A generic `T: Float` adds pervasive type noise for no pedagogical gain. Revisit only if a step demands it. |

<!-- Rows are appended as each step surfaces a real divergence. Keep the
     table ordered by the step that introduced the entry. -->

## Pending / expected divergences

These are anticipated from the roadmap but not yet reached. Each becomes a
table row (with the real detail) when its step lands.

- **Step 52–53 — GPU/CuPy.** The book swaps `numpy` for `cupy` behind a
  `get_array_module` shim. There is no CUDA backend here; the steps are
  ported as a no-op device abstraction and documented, not faked.
- **Step 53/54 — model save/load.** Python uses `np.savez`/pickle. Rust
  needs an explicit, versioned serialization format instead of Python's
  object pickling.
- **Step 26 — graph visualization.** Python shells out to Graphviz `dot`.
  The Rust port emits the same DOT source; rendering to PNG stays optional
  and is not asserted in tests.
- **Step 58 — pretrained VGG16.** The architecture is ported; downloading
  and loading pretrained ImageNet weights is out of scope.
