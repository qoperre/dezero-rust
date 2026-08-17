# Steps 27–35 — higher-order derivatives

**Status:** done (25–26 deferred, see bottom)

## The headline result

**The architecture held. Zero changes to the core were needed.**

`variable.rs`, `function.rs`, `backward.rs` (implementation), and `config.rs`
were untouched; every higher-order test passed on the first run. This is the
payoff for the decision recorded in `docs/ARCHITECTURE.md` — building the core
in its final shape and writing every `Op::backward` in `Variable` arithmetic
rather than following the book's incremental narrative.

Three things were already right, and are worth naming because breaking any of
them later would reintroduce the problem:

1. `Op::backward` receives both `inputs` **and** `outputs`, so `Tanh` (whose
   derivative `1 − y²` is expressed in terms of its *output*) needed no
   interface change.
2. `backward_with`'s config guard wraps the gradient **accumulation**
   (`ops::add`), not just the op's `backward`. Accumulating in raw ndarray
   would silently detach the graph at every fan-in node — a bug that would
   only surface as a wrong second derivative.
3. `x.grad` is an `Option<Variable>`, not an array.

## What was added

`functions/basic_math.rs` gains `Sin`/`sin`, `Cos`/`cos`, `Tanh`/`tanh`,
alongside the existing `Square`/`Exp` — the same grouping Python uses. Each
backward is pure `Variable` arithmetic, so each is differentiable to arbitrary
order:

| op | backward | reads |
|---|---|---|
| `sin` | `gy * cos(x)` | input |
| `cos` | `gy * (−sin(x))` | input |
| `tanh` | `gy * (1 − y²)` | **output** |

`core::ops::one` was promoted to `pub(crate)` so the five unary ops share one
spelling of the arity check instead of five copies.

## Step coverage

| Step | Result |
|---|---|
| 27 | `sin` + the Taylor-series `my_sin` approximation and its gradient |
| 28 | Rosenbrock gradient descent — gradient at the start is exactly `(−2, 400)`; 1000 steps of `lr=0.001` land on `(0.6837118569138317, 0.4659526837427042)`, verified against the Python reference directly |
| 29 | Newton's method with a hand-written `f''(x) = 12x² − 4` |
| 30–32 | ▨ theory — the mechanism they motivate already exists |
| **33** | **Newton's method with an automatic second derivative** |
| 34 | `sin`/`cos` higher-order: the period-4 `sin → cos → −sin → −cos` cycle |
| 35 | `tanh` to third order at `x = 1` |

## Newton's method (step 33) — the acid test

The full 10-iteration trace is bit-for-bit identical to the Python fixture:

```
 0  2.0                    5  1.0000012353089454
 1  1.4545454545454546     6  1.000000000002289
 2  1.1510467893775467     7-10  1.0
 3  1.0253259289766978
 4  1.0009084519430513
```

Separately, the autodiff-driven trace is asserted **equal to the last bit**
(`assert_eq!` on `Vec<f64>`) against the same run using the closed-form
`f''(x) = 12x² − 4`. The automatic second derivative is not merely close to
the analytic one; it is the same float.

## A silent-failure mode that is now pinned

`Tanh::backward` reads its *output*, which the graph holds only through a
`Weak`. If a caller drops `y` before a second backward pass, could the
traversal fall into its zero-substitution path and quietly return a wrong
`y''`?

It cannot — the gradient graph pins `y` as an input of its own `y * y` node.
But that is a non-obvious invariant that could plausibly break later and would
fail *silently* rather than loudly, so it is now a regression test
(`output_based_backward_survives_dropping_the_forward_output`).

Second derivatives are also cross-checked against a central difference of the
*analytic* first derivative computed in plain `f64` — nothing in that
comparison touches the graph, so a backward bug cannot cancel itself out.

## Verification

```
cargo test    122 passed  (84 unit + 25 parity + 13 doctests), up from 98
cargo clippy --all-targets -- -D warnings   clean
cargo fmt --all --check                     clean
```

New fixtures: `sin`, `cos`, `tanh`, `sin_higher_order` (to y'''),
`tanh_higher_order` (to y''), `quartic_second_deriv`, `newton_quartic`.

## Steps 25–26 deferred

DOT-graph emission is isolated from everything else and blocks nothing. It is
scheduled after the tensor-operation group rather than ahead of it, so the
harder work proceeds while the architecture question is freshly settled.
Tracked as still-open in `ROADMAP.md`.

## Next

Steps 36–40: tensor operations and broadcasting (`reshape`, `transpose`,
`sum`, `broadcast_to`, `sum_to`). This is where divergences 2 and 3 — the
panic on mismatched shapes and materialised scalars — get retired.
