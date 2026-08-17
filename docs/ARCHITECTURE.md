# Core architecture

The design the whole port rests on. Decided before step 3; changing it later
is expensive, so it is written down here rather than living in commit
messages.

## Ownership model: `Rc` + `Weak`, mirroring Python

DeZero's graph has a cycle: `Variable.creator` → `Function` → `Function.inputs`
→ `Variable`. Python breaks it by holding **outputs as `weakref`**. We mirror
that exactly:

```
Variable  = Rc<VariableInner>
Function  = Rc<FunctionInner>

VariableInner.creator : Option<Function>        // strong
FunctionInner.inputs  : Vec<Variable>           // strong
FunctionInner.outputs : Vec<Weak<VariableInner>> // WEAK — breaks the cycle
```

**Why not an arena / index graph.** An arena never drops anything until it is
cleared wholesale, so it cannot reproduce "a forward graph dies as soon as the
user drops its output" — which is load-bearing for RNN/BPTT memory (steps
59–60) and is precisely what `unchain()`/`unchain_backward()` manipulate.
Parameters must persist across training steps while intermediate nodes must
not; one arena conflates those lifetimes and would need hand-rolled
refcounting anyway. `Rc`/`Weak` gets it for free.

Accepted trade-offs: `RefCell` borrow errors are runtime, not compile-time
(mitigated by returning owned snapshots from every public accessor, never a
live `Ref`); `Rc` is single-threaded (DeZero is too).

## THE critical implementation rule

> **`Op::backward` operates on `Variable`, never on raw `ArrayD`.**

Backward must build graph nodes through the same `apply()` pipeline as
forward. If backward does raw ndarray math, higher-order derivatives (steps
33–35, Newton's method) are impossible without rewriting every op written
before then.

This is why the port does **not** follow the book's incremental narrative of
"simple version now, `weakref`/`create_graph` retrofit later" — the retrofit
would touch every op. The core is built in its final shape from the start;
individual step docs record which step's semantics each piece satisfies.

## Types

```rust
pub struct VariableInner {
    data: RefCell<Option<ArrayD<f64>>>,  // Option: Python's Variable(None) is a real state
    grad: RefCell<Option<Variable>>,      // grad is a Variable — required for create_graph
    creator: RefCell<Option<Function>>,
    generation: Cell<u32>,
    name: RefCell<Option<String>>,
}

pub trait Op {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>>;
    fn backward(&self, inputs: &[Variable], outputs: &[Variable], gys: &[Variable]) -> Vec<Variable>;
}
```

`data` is `Option` from the first commit: lazily-shaped parameters
(`Linear.W` before first forward, step 44+) need it, and changing it later
would be a breaking type change across every op.

Identity (`id()` in Python — used for `seen` sets, DOT node ids, and
optimizer per-param state) is `Rc::as_ptr` pointer equality on both newtypes.

## Backward traversal

`BinaryHeap` keyed by `generation` replaces Python's re-sort-on-every-push.
Tie order among equal generations is not semantically required.
`create_graph` is applied by wrapping the `op.backward(..)` call in a config
guard — nothing else is special-cased.

## Config / `no_grad()`

`thread_local!` + RAII guard, replacing Python's `contextmanager`:

```rust
let _g = dezero::no_grad();   // restored on scope exit
```

`thread_local!` rather than a global is deliberate: `cargo test` runs tests on
multiple threads, and a global flag would let one test's `no_grad()` leak into
another. Python has no such hazard, so this is a place the port is safer than
the reference at no cost.

## Operator overloading

A macro generates `&V op &V`, `V op V`, `V op f64`, `f64 op V` for each of
`+ - * /`. `impl Mul<Variable> for f64` is legal under the orphan rule because
a local type appears in the trait's parameters.

Python's `__array_priority__ = 200` has **no counterpart here** and is not
ported: it exists only to stop numpy's `ndarray.__mul__` from winning against
`Variable.__rmul__`. We never overload operators on `ArrayD`, so there is no
competing dispatcher.

`__getitem__` cannot be `std::ops::Index` (that must return a reference;
`get_item` constructs a new differentiable `Variable`). It is a plain method.

## Module layout

```
core/     variable, function, backward, ops, config
functions/ basic_math, shape, reduce, matmul, activation, loss, misc, batch_norm, conv
layers/   linear, rnn, conv, norm
models/   sequential, mlp
optim/    sgd, momentum, adagrad, adadelta, adam
data/     dataset, spiral, mnist, dataloader
utils/    shape arithmetic, numerical_diff, dot, npy
```

No GPU/backend module — see `DIVERGENCES.md`.

## Known hazards, tracked

- **Broadcasting** (`sum_to`/`broadcast_to`/`reshape_sum_backward`, steps
  38–40): numpy and `ndarray` differ at the edges. Port near-line-by-line and
  back with the widest fixture matrix in the project.
- **Randomness**: `rand` will never bit-match `np.random`. Any fixture
  touching random values (weight init, dropout masks, shuffling) must supply
  explicit arrays, never "seed both sides and hope".
- **Fancy indexing** (`W[x]`, `log_p[np.arange(N), t]`): `ndarray` has none.
  Gather/scatter-add are hand-written (step 47 and the loss steps).
- **Hidden side quests**: `.npz` writer (step 54) and the MNIST IDX/gzip
  reader (step 51) are real units of work, not free riders on their step.
- **`Layer` field registration**: Python intercepts `__setattr__`. Rust gets
  an explicit `Layer` trait with hand-written `params()`; revisit a derive
  macro only after the manual pattern is proven.
