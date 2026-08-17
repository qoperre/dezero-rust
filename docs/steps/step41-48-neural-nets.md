# Steps 41–48 — neural networks

**Status:** done (48's spiral dataset deferred with 49–50, see bottom)

## The two trait designs

### `Layer` — registration split in two

Python intercepts `__setattr__` to auto-register `Parameter`/`Layer` fields.
Rust cannot, so registration is explicit. The non-obvious choice is *splitting*
it:

```rust
pub trait Layer {
    fn own_params(&self) -> Vec<Parameter>;            // required
    fn sublayers(&self) -> Vec<&dyn Layer> { vec![] }  // only if it has children
    fn forward(&self, x: &Variable) -> Variable;       // required
    fn params(&self) -> Vec<Parameter> { /* provided, recursive */ }
    fn cleargrads(&self) { /* provided */ }
}
```

A single overridable `params()` would let an author list local parameters and
silently forget to recurse — producing a network that trains its first layer
and freezes the rest, with no error. With the split the recursion is written
once, and a new layer can only forget a *field*, which a parameter-count test
catches.

`forward` takes `&self`, not `&mut self`: `Variable` is `Rc` over
interior-mutable cells, so a layer can fill in a lazily-shaped weight during
its first forward without pushing a mutable borrow out into the training loop.
The trait is object-safe, so `Sequential` can hold `Box<dyn Layer>`.

`Model` is a marker supertrait with a blanket impl. Python's `Model` adds only
`plot()`, which needs the step-26 DOT writer this port hasn't reached —
pretending it carried more would be a lie.

### `Optimizer` — `setup` snapshots the params

`setup` stores the **parameter list**, not the target. Storing `&dyn Layer`
would put a lifetime on the optimizer and propagate it into everything that
owns one; `Vec<Parameter>` is a vector of `Rc` handles — no lifetime, and it
keeps the parameters alive.

This is equivalent for every layer in the book *precisely because* a
lazily-shaped `Linear` creates its `W` `Parameter` at construction and only
fills the data in later, so identity is stable across the transition. That is
pinned by `the_lazy_weight_path_reproduces_the_same_training_run`, which calls
`setup` before any weight exists and still reproduces the trace.

`MomentumSgd` keys velocities on `param.id()` (`Rc::as_ptr`), safe against
address reuse because the optimizer holds a strong handle to every parameter.

## The integration result

The 50-step two-layer training run matches Python essentially bit-for-bit:

| quantity | worst relative deviation |
|---|---|
| loss trace, 50 steps | **3.7e-16** |
| final `W1`/`b1`/`W2`/`b2` | 1.35e-15 / 1.37e-15 / 1.45e-15 / 6.6e-16 |
| matmul, relu, cross-entropy forward, all matmul/linear grads | **0** (exact) |
| everything else | ≤ 2.3e-15 |

That is the proof `Layer` + `Parameter` + `Optimizer` + loss all compose
correctly — no single unit test covers it.

## Gather / scatter for cross-entropy

`ndarray` has no fancy indexing, so `log_p[np.arange(N), t]` is hand-written.
`utils/array.rs` has `gather_rows` and `scatter_add_rows` written as an exact
**adjoint pair**, unit-tested as one: `<gather(x), g> == <x, scatter(g)>` over
several index patterns.

The backward uses Python's closed form `(softmax(x) − onehot)/N`, where the
one-hot matrix *is* `scatter_add_rows(labels, ones, C)` — so the scatter is
real, used, and independently testable rather than inlined into a formula.

## Beyond the brief: a hand-rolled RNG

Lazy weight init needs normal deviates, and `ndarray` is the only permitted
dependency. `utils/random.rs` is a self-contained SplitMix64 + Box–Muller with
a `thread_local!` stream and `dezero::seed()` (mirroring the `Config`
reasoning). It ships with distribution tests — mean, variance, tail shape, and
the half-open interval bound.

Same distribution as `np.random.randn`, different stream. That is exactly why
every fixture ships explicit weights rather than a seed.

## Divergences recorded

Rows 12–20 in `docs/DIVERGENCES.md`. The one worth calling out: **row 12
amends row 1**. Python's `L.Linear` defaults to `float32`, so an unpinned
Python run holds float32 weights. The fixture overwrites all four arrays with
float64 — which is *why* this parity is 1e-16 rather than 1e-7.

## Verification

```
cargo test    395 passed  (297 unit + 35 parity_core + 10 parity_nn + 53 doctests), up from 210
cargo clippy --all-targets --all-features -- -D warnings   clean
cargo fmt --all --check                                    clean
RUSTDOCFLAGS="-D warnings" cargo doc                       clean
```

All 210 pre-existing tests still pass, **none modified** — the diff on
previously-existing files is 78 insertions / 6 deletions, all additive. No new
dependencies. Suite run 3× in parallel and once with `--test-threads=1` to
rule out RNG/thread-local flakiness.

## Not yet ported (tracked, not forgotten)

Optimizer hooks (`WeightDecay`/`ClipGrad`/`FreezeParam` — step 50),
`log_softmax`, `leaky_relu`, `sigmoid_cross_entropy`, `clip`. Step 48's spiral
dataset is deferred to land with the `Dataset` abstraction in steps 49–50
rather than as a one-off.

## Next

Steps 49–52: `Dataset`, `DataLoader`, MNIST (needs an IDX/gzip reader — a real
side quest), and the GPU step, which is a documented no-op.
