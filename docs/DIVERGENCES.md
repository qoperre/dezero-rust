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
| ~~2~~ | ~~Mismatched operand shapes~~ | — | **retired at step 40** | Ops broadcast numpy-style and fold gradients back with `sum_to`. |
| ~~3~~ | ~~Scalar operands~~ | — | **retired at step 40** | Scalars stay 0-d, matching Python's `as_array(2.0)`. |
| 4 | `backward()` on a leaf | raises `AttributeError` | no-op (after seeding `grad`) | Strictly friendlier; no test depends on the crash. |
| 5 | Gradient of a dropped output | crashes (`None.grad`) | zeros of the recorded shape | `FunctionInner` keeps `output_shapes` so the zero is well-defined. Additive and defensive; currently unreachable. |
| 6 | Error signalling in ops | Python raises | **panics** with `# Panics` docs | `Op::forward` returns `Vec<ArrayD>` and operator overloads must return `Variable`, so no `Result` can thread through arithmetic. Confined to missing-data and arity/shape violations. |
| 7 | `__array_priority__ = 200` | needed so numpy defers to `Variable.__rmul__` | **not ported** | We never overload operators on `ArrayD`, so no competing dispatcher exists. The problem class does not occur in Rust. |
| 8 | `Config` storage | class attribute (process-global) | `thread_local!` | `cargo test` runs tests on multiple threads; a global would leak one test's `no_grad()` into another. Safer than the reference at no cost. |
| 9 | `transpose(axes=...)` (step 38) | optional permutation argument | **not ported**; `transpose` reverses all axes (numpy `.T`) | No step through 40 uses the permutation form. Its backward needs `argsort` of the inverse permutation; add it when a step actually needs it rather than shipping untested code. |
| 10 | `reshape(-1)` placeholder (step 38) | `-1` infers one dimension | concrete lengths only | `flatten` is the only caller that needs it and can compute the length from `Variable::size` when that step lands. |
| 11 | `utils::sum_to` target validation | numpy silently returns a wrong-shaped array | **checks** that a non-1 target axis equals the source axis | Never fires on a legitimate call path; turns a silent wrong-gradient bug into a loud one. |
| 12 | `L.Linear` weight dtype (step 44) | defaults to `float32` | `f64` | Amends row 1 at the step where it bites. The `training_two_layer` fixture pins all four weight arrays as float64, which is why that parity is 1e-16 rather than 1e-7. |
| 13 | Weight-init RNG (step 44) | `np.random.randn` | hand-rolled SplitMix64 + Box–Muller in `utils::random`, `thread_local!`, `dezero::seed()` | `ndarray` is the only permitted dependency. Same distribution, different stream — which is exactly why every fixture ships explicit weights. |
| 14 | `Optimizer.setup` (step 46) | stores the target, re-reads `target.params()` each `update()` | snapshots the parameter list | Storing `&dyn Layer` would push a lifetime into everything owning an optimizer. Equivalent for every layer in the book, since a lazily-shaped `Linear` creates its `Parameter` at construction and only fills the data later. A layer that *replaced* a parameter would need a second `setup`. |
| 15 | `softmax_cross_entropy` labels (step 47) | `t` is a second input `Variable` whose gradient is silently dropped (`backward` returns 1 value for 2 inputs; `zip` truncates) | `&[usize]` owned by the op | One input, one gradient, nothing relying on truncation. |
| 16 | `linear` optional bias (step 43) | `b=None` becomes a `Variable(None)` input | encoded in the op's arity: `linear(x, W, None)` builds a 2-input node | `apply` rejects data-less inputs and `backward_with` asserts one gradient per input. |
| 17 | `mean_squared_error` operand shapes (step 42) | broadcasts, then returns gradients shaped like the broadcast difference rather than the operands | requires equal shapes; panics otherwise | Same spirit as row 11. |
| 18 | `matmul`/`linear` rank (step 41) | `np.dot` generalises beyond 2-D | 2-D only; panics otherwise | `MatMul::backward`'s transpose does not generalise. A loud panic beats a right-shaped wrong gradient. |
| 19 | `params()` ordering (step 44) | Python's `_params` is a `set` — hash-dependent order | deterministic: own params first, then each sublayer | No consequence for SGD/Momentum. Matters for `ClipGrad`, whose total-norm sum is order-sensitive at the last bit. |
| 20 | Acronym naming | `SGD`, `MomentumSGD`, `MLP` | `Sgd`, `MomentumSgd`, `Mlp` | clippy's `upper_case_acronyms` is warn-by-default and this project builds with `-D warnings`. |
| 21 | gzip (step 51) | Python's `gzip` module | the `flate2` crate | DeZero's MNIST archives are gzip. A hand-written DEFLATE decoder is a large, error-prone unit of work with no pedagogical value here; `flate2` (pure-Rust `miniz_oxide` backend) is the de-facto standard and is what cargo itself uses. The only dependency added beyond `ndarray`. |
| 22 | `Dataset` shape (step 49) | `__getitem__` returns a tuple; `transform`/`target_transform` are per-instance fields | `input()`/`label()` split; labels are `Option<usize>` class indices; transforms are `map_input`/`map_label` combinators | A batch needs the halves in different containers (`ArrayD` vs `Vec<usize>`); a tuple would be built only to be taken apart. **Step 59's `SinCurve` is a regression dataset and will force `label` to widen.** |
| 23 | `DataLoader` iteration (step 50) | `__next__` calls `reset()` then raises `StopIteration`, so the loader silently restarts | proper `FusedIterator`: `next()` returns `None` for good; `reset()` starts the next epoch | An iterator that resurrects itself after `None` breaks the contract `.chain()`/`.zip()`/`.by_ref()` rely on. Shuffling draws from a loader-local seeded `Rng`, not a process-global stream. |
| 24 | `FreezeParam` (step 50) | `p.grad = None`; Python's own `update()` then crashes with `AttributeError`, since it snapshots the param list before hooks run | clears the gradient too, and no-ops instead of crashing | Clearing, not zeroing, is load-bearing: with a *zero* gradient `MomentumSgd` still moves the weight by its decaying velocity, so only clearing is actually frozen. Same family as row 4. |
| 25 | MNIST download (step 51) | `get_file` fetches over HTTP | cache lookup only; `MnistError::Missing { path, url }` names the file and where to put it | An HTTP client is a dependency decision beyond the one authorized for gzip. The reader is fully tested against synthetic IDX files (plain and gzipped); the fetch is the piece a user supplies. |
| 26 | IDX header (step 51) | `np.frombuffer(..., offset=16)` + `reshape(-1,1,28,28)`, discarding the header | parses and validates magic, element type, rank, and big-endian dims | A wrong file errors instead of producing plausible garbage. Only element type `0x08` (u8) is decoded. |
| 27 | **Step 52 — GPU** | `cuda.get_array_module` swaps numpy for cupy; `DataLoader` has `gpu`/`to_cpu()`/`to_gpu()` | **nothing.** No device module, no shim, no `gpu` field | There is no CUDA backend, and inventing a device abstraction that does nothing would be worse than its absence. Batches are always `ArrayD<f64>`. See `docs/GPU.md`. |
| 28 | `Optimizer` hook storage (step 50) | `self.hooks` inherited from the base class | each impl declares `hooks()`/`hooks_mut()` | Rust traits have no inherited state. Same reasoning as row 19 and the `Layer` split. |
| 29 | `SeqDataLoader` / `SinCurve` (step 60) | sequential batching over a regression dataset | **not ported** | `SinCurve` is a regression set, and `Dataset::label` is an `Option<usize>` class index (row 22). Widening it is row 22's flagged follow-up; bolting a second label type on here would make the trait worse to avoid one dataset. |
| 30 | `plot_dot_graph` (step 26) | writes a temp file, shells out to the Graphviz `dot` binary, returns a Jupyter `Image` | `get_dot_graph` returns the DOT **text**; the caller renders it | The binary may not be installed, the image is not something a test can assert on, and spawning a process belongs in a caller rather than a library. |
| 31 | Weights file format (step 53) | `np.savez_compressed` — a zip of `.npy` members | JSON: key → `{shape, data}`, with the numbers as **strings** | An inspectable file is worth a lot mid-port. The numbers are strings because `serde_json`'s parser is not correctly rounded — measured, not assumed: it returns a value 1 ULP off for ordinary weights. Rust's own `Display`/`parse` pair is exact. The two formats do not interoperate. `serde`/`serde_json` are now runtime deps. |

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
