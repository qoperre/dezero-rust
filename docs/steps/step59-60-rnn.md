# Steps 59–60 — recurrent layers

**Status:** done

## What makes these different

Every layer before them is a pure function of its input. `Rnn` and `Lstm`
carry a hidden state **across calls**, which changes three things:

1. **State lives in a `RefCell`.** `Layer::forward` takes `&self`, so the
   state cannot be a plain field. This is the same interior-mutability choice
   `Linear` already makes for its lazily-shaped weight, and for the same
   reason: `&mut self` would push a mutable borrow through the entire training
   loop.
2. **The graph grows without bound.** Each call links the new state to the
   previous one, so `n` unbroken steps build a graph `n` layers deep — that is
   BPTT. Nothing frees it while the state still references it.
3. **Two ways to stop that**, both from the book: `reset_state()` at a sequence
   boundary, or `unchain_backward()` to keep the state's *value* but cut its
   history (truncated BPTT).

Each `forward` reads the previous state and **drops the borrow before running
any ops** — the ops allocate and could otherwise re-enter the cell.

## Shapes

```
Rnn:   h' = tanh(x·Wx + b + h·Wh)          (the h·Wh term is absent on step 1)
Lstm:  f,i,o = sigmoid(x·Wx + b + h·Wh)
       u     = tanh   (x·Wx + b + h·Wh)
       c' = f*c + i*u                       (just i*u on step 1)
       h' = o * tanh(c')
```

Only the `x`-side transforms carry a bias — one per pair is enough, which
Python spells `nobias=True`. The recurrent weight is **hidden × hidden**, not
`in_size × hidden`; a test pins that, since it is an easy thing to get wrong.

## Two wrong expectations I wrote, both corrected against the reference

Writing the tests first surfaced two beliefs of mine that were simply false.
Both were checked by running the equivalent Python rather than by reasoning
about it:

**1. Lazy weights do not all settle on the first batch.** I asserted that one
forward pass shapes both transforms. It shapes only `x2h` — the first step has
no previous state, so `h2h` is never called and stays unshaped until step 2.
Python does exactly the same. The test now asserts one transform per step, and
says why.

**2. `unchain_backward` does not unchain the node it is called on.** I asserted
`h.creator().is_none()` afterwards. It cuts the *ancestors*: Python's loop
unchains `f.inputs`, never `self`. The test now checks that the producing step
survives and everything upstream of it is severed — which is what truncation
actually means.

Neither was an implementation bug. In both cases the code was right and my
expectation was wrong, and the reference settled it in under a minute.

## Verification

Both fixtures unroll **three timesteps with the weights pinned**, so they
check the state carried between calls rather than a single forward:

| test | what it catches |
|---|---|
| `rnn_unrolled_matches_python` | step 1 uses only `x2h`; steps 2–3 add the recurrent term, so a state that failed to carry matches the first row and diverges after |
| `rnn_bptt_reaches_every_weight` | back-propagating from the last state reaches every parameter through all three steps |
| `lstm_unrolled_matches_python` | `c` is invisible in the output except through its effect on later steps — this is the test that catches a cell state that fails to persist |

Plus 11 unit tests: parameter registration and counts, the hidden×hidden
recurrent shape, bias placement across all eight LSTM gates, state carry and
reset, BPTT reaching every gate, and that `h = o * tanh(c)` stays inside
`[-1, 1]` even when the cell state saturates.

```
cargo test    629 passed  (471 unit + 3 parity_rnn + 9 parity_conv
                           + 35 parity_core + 11 parity_data + 10 parity_nn
                           + 90 doctests)
cargo clippy --all-targets --all-features -- -D warnings   0 errors
cargo fmt --all --check                                    clean
```

Fixture field names are Python's (`x2h_W`), renamed through `serde(rename)`
rather than suppressing the snake-case lint across the file.

## Not ported

`SeqDataLoader` (step 60's sequential batching) and the `SinCurve` dataset.
`SinCurve` is a **regression** dataset, and the `Dataset` trait's
`label: Option<usize>` is a class index — widening it is divergence 22's
flagged follow-up, not something to bolt on here.

## Still open

Steps 53 (model save/load), 58 (VGG16), and 25–26 (DOT graph).
