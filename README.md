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

**All 60 steps ported.** See `docs/ROADMAP.md` for the step-by-step record and
`docs/steps/` for what each group involved.

```
cargo test                          655 passed, 4 ignored
cargo test --release -- --ignored     4 passed   (VGG16, slow in debug)
cargo clippy --all-targets --all-features -- -D warnings   clean
cargo fmt --all --check                                    clean
```

86 golden fixtures generated from the Python reference are compared on every
run — forward at `rtol=1e-5`, gradients at `rtol=1e-4`. The two integration
fixtures are the load-bearing ones: a 50-step MLP training run matches Python
to 3.7e-16 on the loss trace, and a 300-step spiral classification run to
4.5e-16.

Dependencies: `ndarray`, `flate2` (gzip, for MNIST), `serde`/`serde_json`
(weights files). Everything else — the autodiff engine, broadcasting, im2col,
the RNG — is written here.

### Deliberately not ported

Recorded with reasons in `docs/DIVERGENCES.md` (32 entries) rather than
silently skipped. The substantive ones:

- **GPU (step 52).** No CUDA backend, and no fake device abstraction standing
  in for one.
- **Pretrained VGG16 weights (step 58)** and the **MNIST download (step 51)** —
  both need an HTTP client this crate does not have. The MNIST *reader* is
  implemented and tested against synthetic archives.
- **`SeqDataLoader`/`SinCurve` (step 60)** — `SinCurve` is a regression
  dataset, and `Dataset::label` is a class index. Widening it is tracked, not
  forgotten.

Two divergences were **retired** when their step arrived: mismatched-shape
operands stopped panicking and started broadcasting at step 40.
