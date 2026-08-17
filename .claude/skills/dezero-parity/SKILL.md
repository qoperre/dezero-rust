---
name: dezero-parity
description: Generate and run numerical-parity fixtures comparing the Python DeZero reference (vendor/dezero-python) against the Rust port (crates/dezero). Use whenever porting a new step/function/Layer/Model from Python to Rust, or when asked to "check parity", "verify against Python", or "add a parity test".
---

# DeZero Python↔Rust Parity Testing

This skill drives the golden-fixture workflow that keeps the Rust `ndarray`-based
port numerically equivalent to the original Python/numpy DeZero implementation
vendored at `vendor/dezero-python/` (git submodule, read-only reference).

## Layout

```
tests/parity/
  gen_fixtures.py        # Python: imports vendor/dezero-python, writes fixtures/
  fixtures/<unit>.json    # {"inputs": [...], "output": [...], "grads": [...]}
  <unit>_test.rs          # Rust: loads fixtures/<unit>.json, asserts allclose
```

## Workflow (run for every newly ported unit)

1. **Pick the unit** being ported (a single `Function`, `Layer`, or a whole
   `stepNN.py`). Find its Python source in `vendor/dezero-python/dezero/` or
   `vendor/dezero-python/steps/`.
2. **Write/extend `tests/parity/gen_fixtures.py`** for this unit:
   - `sys.path.insert(0, "vendor/dezero-python")` then `import dezero`.
   - Seed numpy (`np.random.seed(<fixed>)`) and construct representative inputs:
     scalar, 1D, 2D, and at least one broadcasting case.
   - Run forward, and if the unit is differentiable, call `.backward()` and
     capture `.grad` too.
   - Dump `{"inputs": ..., "output": ..., "grads": ...}` as JSON (plain nested
     lists — keep it dependency-free on the Rust side) into
     `tests/parity/fixtures/<unit>.json`.
3. **Regenerate fixtures**: `python tests/parity/gen_fixtures.py` (requires the
   `vendor/dezero-python` submodule's own deps — see its `requirements.txt` /
   README; a local venv is fine, this never runs in the Rust CI job).
4. **Write the Rust side**: `tests/parity/<unit>_test.rs` loads the same JSON,
   runs the Rust implementation on the same inputs, and asserts closeness:
   - forward output: `rtol=1e-5, atol=1e-8` (numpy `allclose` defaults)
   - gradients: looser, `rtol=1e-4`, since DeZero's own numerical-diff checks
     (`vendor/dezero-python/dezero/utils.py::numerical_diff`) are themselves
     approximate.
5. **Run it**: `cargo test -p dezero --test <unit>_test`.
6. On failure, delegate root-causing to the `parity-checker` subagent rather
   than guessing — dtype, broadcasting-rule, and reduction/keepdims mismatches
   between numpy and `ndarray` are the most common causes.

## Rules

- Never edit anything under `vendor/dezero-python/` — it's the pinned ground
  truth submodule.
- Fixtures must be deterministic (seeded) and small enough to diff sanely in
  git — prefer shapes like `(2,3)` over large random tensors.
- A ported unit isn't "done" until it has a parity fixture + passing Rust test.
- If the Rust implementation intentionally diverges from Python (e.g. a
  different default dtype), document that in a comment next to the assertion
  instead of loosening tolerance to hide it.
