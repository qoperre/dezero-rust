# Steps 1–18 — the core autodiff engine

**Status:** done · **Commit:** see `git log` for "Implement core autodiff engine"

## Why these 18 steps landed together

The book builds the core incrementally: a simple `Variable`, then backprop,
then variadic functions, then `weakref`s, then `Config`. That narrative is
excellent for *reading* and wrong for *porting* — retrofitting `Weak` onto an
already-written `Rc`-only graph, or retrofitting `create_graph` onto ops whose
`backward` does raw ndarray math, means rewriting every op written before the
retrofit.

So the core was built once in its final shape. Each step's semantics are
listed below with the test that pins it, so nothing is skipped — only the
intermediate half-built versions are.

See `docs/ARCHITECTURE.md` for the design and its justification.

## What each step contributes, and what proves it

| Step | Semantics | Where it lives | Test |
|---|---|---|---|
| 01 | `Variable` as a data box | `core/variable.rs` | `Variable::new` doctest |
| 02 | `Function`/`Square` | `core/function.rs`, `functions/basic_math.rs` | `parity_square.rs` |
| 03 | function composition | `apply` chaining | `composed_chain_matches_python` |
| 04 | numerical differentiation | `utils::numerical_diff` | doctest + used by all gradient checks |
| 05 | ▨ backprop theory | — | — |
| 06 | manual backprop | `core/backward.rs` | `square_backward_matches_python` |
| 07 | automatic backprop via `creator` | `VariableInner.creator` | `composed_chain_matches_python` |
| 08 | recursion → loop | `BinaryHeap` traversal | same |
| 09 | `as_array`/auto-`grad` init | `Variable::from_scalar`, ones-like seed | `Variable::backward` doctest |
| 10 | tests / gradient check | `utils::gradient_check` | 70 unit tests |
| 11 | variadic in/out | `Op::forward(&[&ArrayD]) -> Vec<ArrayD>` | `add_matches_python` |
| 12 | improved call API | `apply` / `apply1` | — |
| 13 | backprop through variadic fns | `Op::backward(...) -> Vec<Variable>` | `add_matches_python` (both operands) |
| 14 | **grad accumulation + `cleargrad`** | `x.grad = x.grad + gx` | `repeated_variable_accumulates_grad` (`x+x` → 2) |
| 15 | ▨ topology theory | — | — |
| 16 | `generation`-ordered backprop | `Cell<u32>` + `BinaryHeap` | diamond test (`x.grad == 64`) |
| 17 | **weakrefs breaking the cycle** | `Vec<Weak<VariableInner>>` | `dropping_an_output_breaks_the_cycle` |
| 18 | `retain_grad`, `no_grad()` | `core/config.rs` RAII guard | `no_grad` test (no creator, generation 0) |

## The rule everything downstream depends on

Every `Op::backward` is written in **`Variable` arithmetic**, never raw
ndarray math. That is what makes steps 33–35 (higher-order derivatives,
Newton's method) fall out for free instead of forcing a rewrite. Verified now
by tests that take second derivatives of `square`, `exp`, and `pow` through
the same ops.

## Verification

```
cargo test    93 passed  (70 unit + 11 parity_core + 2 parity_square + 10 doctests)
cargo clippy --all-targets -- -D warnings   clean
cargo fmt --all --check                     clean
```

11 golden fixtures generated from Python DeZero (`exp`, `square_backward`,
`composed_sq_exp_sq`, `add`, `add_same_var`, `mul`, `sub`, `div`, `neg`,
`pow3`, `composite_arith`) match forward at `rtol=1e-5` and gradients at
`rtol=1e-4`.

## Divergences introduced

Recorded in `docs/DIVERGENCES.md`:

1. **Mismatched shapes panic** in `Add/Sub/Mul/Div::forward` instead of
   broadcasting. Python leans on numpy broadcasting and folds gradients back
   with `sum_to`, which arrives in step 40. Panicking with a message naming
   step 40 is honest; silently returning wrong-shaped gradients would not be.
2. **Scalars are materialised** to the peer operand's shape rather than kept
   0-d. Numerically identical, uses more memory, removed at step 40.
3. **`backward()` on a leaf is a no-op**; Python raises `AttributeError`.
4. **`FunctionInner` records `output_shapes`** so a dropped output's zero
   gradient has a defined shape. Python crashes in that situation.
5. **Two deliberate panic sites**: `Op::forward` returns `Vec<ArrayD>` and
   operator overloads must return `Variable`, so no `Result` can be threaded
   through arithmetic. Missing data and arity violations panic with
   explanatory messages and `# Panics` doc sections.

## Next

Step 19 onward proceed one at a time on this foundation. The next genuinely
hard groups are 33–35 (double backprop — the architecture's acid test) and
38–40 (broadcasting).
