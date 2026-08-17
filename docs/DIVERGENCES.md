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
| 2 | Mismatched operand shapes (step 11+) | numpy broadcasts; grads folded back via `sum_to` | **panics**, with a message naming step 40 | `sum_to` arrives at step 40. Panicking is honest; broadcasting now would return wrong-shaped gradients silently. Removed when step 40 lands. |
| 3 | Scalar operands (step 21) | stays 0-d, relies on numpy broadcasting | materialised to the peer operand's shape | Numerically identical, costs memory. Removed when step 40 lands. |
| 4 | `backward()` on a leaf | raises `AttributeError` | no-op (after seeding `grad`) | Strictly friendlier; no test depends on the crash. |
| 5 | Gradient of a dropped output | crashes (`None.grad`) | zeros of the recorded shape | `FunctionInner` keeps `output_shapes` so the zero is well-defined. Additive and defensive; currently unreachable. |
| 6 | Error signalling in ops | Python raises | **panics** with `# Panics` docs | `Op::forward` returns `Vec<ArrayD>` and operator overloads must return `Variable`, so no `Result` can thread through arithmetic. Confined to missing-data and arity/shape violations. |
| 7 | `__array_priority__ = 200` | needed so numpy defers to `Variable.__rmul__` | **not ported** | We never overload operators on `ArrayD`, so no competing dispatcher exists. The problem class does not occur in Rust. |
| 8 | `Config` storage | class attribute (process-global) | `thread_local!` | `cargo test` runs tests on multiple threads; a global would leak one test's `no_grad()` into another. Safer than the reference at no cost. |

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
