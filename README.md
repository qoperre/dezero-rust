# dezero-rust

A Rust port of [DeZero](https://github.com/oreilly-japan/deep-learning-from-scratch-3)
(the deep learning framework built step-by-step in *Deep Learning from Scratch 3*),
using [`ndarray`](https://docs.rs/ndarray) as the tensor backend.

## Layout

```
crates/dezero/          # the Rust port (lib crate "dezero")
vendor/dezero-python/    # git submodule: the original Python reference (read-only, pinned)
tests/parity/            # Python <-> Rust numeric-parity fixtures
  gen_fixtures.py         # regenerates fixtures/*.json from vendor/dezero-python
  fixtures/*.json         # committed golden data (input/output pairs)
.claude/
  agents/                 # rust-engineer, rust-pro, tdd-orchestrator, parity-checker
  skills/                 # tdd, rust-strict, dezero-parity
  hooks/                  # cargo fmt on save, fmt+clippy+test gate on Stop
```

## Getting started

```sh
git submodule update --init --recursive   # pulls vendor/dezero-python
cargo build
cargo test
```

## Porting workflow

Each DeZero step/function/Layer/Model is ported one slice at a time:

1. Port the unit into `crates/dezero/src/` (idiomatic Rust — see `.claude/skills/rust-strict`).
2. Add a golden fixture: extend `tests/parity/gen_fixtures.py` to run the same
   input through `vendor/dezero-python` and dump expected output.
3. Add a Cargo integration test under `crates/dezero/tests/` that loads the
   fixture and asserts the Rust output matches within tolerance.
4. If a parity test fails, invoke the `parity-checker` subagent to diagnose
   before touching code — see `.claude/skills/dezero-parity`.

CI (`.github/workflows/ci.yml`) runs `cargo fmt --check`, `clippy -D warnings`,
`build`, and `test` on every push/PR, plus a job that regenerates the parity
fixtures from the Python reference and fails if they drift from what's
committed.

## Status

First slice ported: `Variable` (step 1) + `square` (step 2), with a passing
parity test against the Python reference. The rest of the ~60 DeZero steps
are ported incrementally from here.
