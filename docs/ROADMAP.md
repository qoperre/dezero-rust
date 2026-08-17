# DeZero → Rust Port Roadmap

All 60 steps of *Deep Learning from Scratch 3*, ported to Rust on `ndarray`.

**Ground truth:** `vendor/dezero-python/` (git submodule, read-only).
**Rule:** every step that produces code ships with a parity fixture generated
from the Python reference and a passing Rust test. See
`.claude/skills/dezero-parity`.

**Legend:** ☐ todo · ☑ done · ▨ explanation-only in the book (no Rust
deliverable of its own; recorded for completeness)

---

## Phase 1 — Core autodiff (steps 1–10)

Variable/Function boxes, numerical differentiation, then backprop: manual →
automatic → recursion → loop.

- ☑ 01 — `Variable` as a data box
- ☑ 02 — `Function` / `Square`
- ☑ 03 — function composition (`Exp`, chaining)
- ☑ 04 — numerical differentiation
- ☑ 05 — ▨ backprop theory
- ☑ 06 — manual backprop
- ☑ 07 — automatic backprop (recursive, via `creator`)
- ☑ 08 — recursion → loop
- ☑ 09 — ergonomics: `as_array`, `as_variable`, auto `grad` init
- ☑ 10 — unit tests / gradient check

## Phase 2 — Variadic functions & memory (steps 11–18)

- ☑ 11 — variadic inputs/outputs (`Add`)
- ☑ 12 — improved variadic call API
- ☑ 13 — backprop through variadic functions
- ☑ 14 — repeated-use grad accumulation + `cleargrad`
- ☑ 15 — ▨ topology theory
- ☑ 16 — `generation`-ordered backprop
- ☑ 17 — weakrefs (Rust: `Weak`) to break graph cycles
- ☑ 18 — memory saving: `retain_grad`, `Config.enable_backprop`, `no_grad()`

## Phase 3 — Usability (steps 19–24)

- ☑ 19 — `Variable` properties (`shape`/`ndim`/`size`/`dtype`/`len`/`name`)
- ☑ 20 — operator overloading: `+`, `*`
- ☑ 21 — operands mixing `Variable` with scalars/arrays
- ☑ 22 — `-`, `/`, `**`, unary neg, reversed ops
- ☑ 23 — package layout (Rust: module layout)
- ☑ 24 — complex-function smoke tests (Sphere, matyas, Goldstein-Price)

## Phase 4 — Graph visualization (steps 25–26)

- ☑ 25 — ▨ Graphviz introduction
- ☑ 26 — DOT-graph emission (`plot_dot_graph` equivalent)

## Phase 5 — Higher-order derivatives (steps 27–35)

The hard part: backward must itself build a graph (`create_graph`).

- ☑ 27 — `sin`, Taylor-series approximation
- ☑ 28 — gradient descent (Rosenbrock)
- ☑ 29 — Newton's method with a hand-written 2nd derivative
- ☑ 30 — ▨ what an automatic 2nd derivative requires
- ☑ 31 — ▨ theory: `grad` as a `Variable`
- ☑ 32 — ▨ theory: double-backprop design
- ☑ 33 — automatic 2nd derivative → Newton's method
- ☑ 34 — `sin`/`cos` higher-order derivatives
- ☑ 35 — `tanh` + deep derivative graph

## Phase 6 — Tensor operations (steps 36–40)

- ☑ 36 — double-backprop application (`create_graph` in user code)
- ☑ 37 — tensor-shaped forward/backward
- ☑ 38 — `reshape`, `transpose`
- ☑ 39 — `sum` (with `axis`, `keepdims`)
- ☑ 40 — broadcasting: `broadcast_to`, `sum_to`

## Phase 7 — Neural networks (steps 41–48)

- ☑ 41 — `matmul`
- ☑ 42 — linear regression
- ☑ 43 — a hand-rolled neural net
- ☑ 44 — `Parameter` / `Layer`
- ☑ 45 — nested `Layer` → `Model`
- ☑ 46 — `Optimizer` (SGD, Momentum)
- ☑ 47 — softmax / cross-entropy
- ☑ 48 — multi-class classification (spiral dataset) — deferred to land with `Dataset` in 49–50

## Phase 8 — Datasets & training loop (steps 49–52)

- ☑ 49 — `Dataset` abstraction
- ☑ 50 — `DataLoader` (mini-batching, epochs)
- ☑ 51 — MNIST
- ☑ 52 — GPU/CuPy → **adapted**: no CUDA backend; documented as a
  deliberate divergence (see `docs/DIVERGENCES.md`)

## Phase 9 — Model persistence & regularization (steps 53–56)

- ☑ 53 — model save/load (Python pickles/npz → Rust: explicit serialization)
- ☑ 54 — dropout + `test_mode()` / `Config.train`
- ☑ 55 — ▨ CNN concepts
- ☑ 56 — ▨ convolution arithmetic

## Phase 10 — CNN (steps 57–58)

- ☑ 57 — `im2col`, `conv2d`
- ☐ 58 — pretrained VGG16 → **adapted**: architecture ported; pretrained
  weight download is out of scope (documented divergence)

## Phase 11 — RNN (steps 59–60)

- ☑ 59 — `RNN` layer, sine-wave prediction, BPTT + `unchain_backward`
- ☑ 60 — `LSTM`, `SeqDataLoader`

---

## Working agreement

1. One step at a time, in order.
2. Each step: implement → parity fixture + Rust test → `cargo fmt`/`clippy
   -D warnings`/`test` all green → write `docs/steps/stepNN.md` → commit.
3. Subagents: `rust-engineer`/`rust-pro` implement, `parity-checker`
   diagnoses numeric mismatches, `tdd-orchestrator` for test-first slices.
4. Any place the Rust port intentionally differs from Python is recorded in
   `docs/DIVERGENCES.md` rather than silently papered over.
