# Steps 55–57 — convolution

**Status:** done

## What landed

| Module | Contents |
|---|---|
| `utils/conv.rs` | `Pair`, `pair`, `get_conv_outsize`, `get_deconv_outsize`, `im2col_array`/`col2im_array` (step 56's content) |
| `functions/conv.rs` | `Im2col`/`im2col`, `Col2im`/`col2im`, `Conv2d`/`conv2d`, `Deconv2d`/`deconv2d`, `Pooling`/`pooling` |

Steps 55–56 are explanation-only in the book (CNN concepts, convolution
arithmetic). Their actual content — the output-size formulas — is
`utils::conv`, with tests over many kernel/stride/pad combinations.

`conv2d` operates on rank-4 `[N, C, H, W]`. `VariableInner.data` stays
`ArrayD<f64>`; the ops convert to `Ix4` internally and back to `IxDyn`.

## How this step was actually finished

The subagent implementing it hit a session limit partway through and stopped
with the work **uncommitted and unverified**. Recording what that left behind,
because "the file exists" is not the same as "the code works":

- `functions/conv.rs` was 1000+ lines that had **never been compiled** — it
  was not declared in `functions/mod.rs`, so `cargo build` never saw it.
- `utils/conv.rs` *was* wired in, and its two helpers for the unreachable
  module were dead code, which **failed the clippy gate**.
- A doctest in `utils/random.rs` failed: it used `dezero::rand`, which was
  never re-exported.

Finishing it meant wiring the module in, re-exporting the conv API, and then
running the code for the first time. Two real defects surfaced immediately:

**1. `ndarray::s![]` versus `forbid(unsafe_code)`.** The `s!` macro expands to
code carrying `allow(unsafe_code)`, which the crate's `forbid` rejects
outright — `forbid` cannot be locally overridden, by design. Fixed by using
`index_axis(Axis(1), n)`, which is the same operation without the macro.
Keeping `forbid` was worth more than the terser slice syntax.

**2. A wrong test expectation about pooling's second derivative.** The test
asserted `d²(maxpool)/dx² == 0` "because max pooling is piecewise linear".
It failed. Checking the reference settled it: Python also leaves `x.grad` as
`None` there. Max pooling routes each gradient through an **argmax index**,
and those indices are constants — so `gx` has no differentiable dependence on
`x` and there is no graph path at all. Zero and absent are different things.
The implementation was right; the expectation was wrong, and it now asserts
`x.grad().is_none()` with the reason written down.

## Verification

All 7 convolution fixtures match Python:

| fixture | what it pins |
|---|---|
| `im2col_k3s1p0`, `im2col_k3s2p1` | the column layout, with and without stride/pad |
| `conv2d_s1p0` | the plain case |
| `conv2d_s1p1` | padding preserves spatial size, `[2,4,5,5]` |
| `conv2d_s2p1` | stride *and* pad together — the combination that breaks if `get_conv_outsize` is off by one |
| `pooling_k2s2p0` | non-overlapping windows |
| `pooling_k3s1p1` | **overlapping** windows, where one input can win several windows and its gradient must *accumulate* — a scatter that overwrote instead of adding would show up only here |

Forward at `rtol=1e-5`, gradients at `rtol=1e-4`. `conv2d` checks all three
gradients (`gx`, `gW`, `gb`).

```
cargo test    609 passed  (460 unit + 7 parity_conv + 35 parity_core
                           + 11 parity_data + 10 parity_nn + 86 doctests)
cargo clippy --all-targets --all-features -- -D warnings   clean
cargo fmt --all --check                                    clean
```

## Still open in this phase

Steps 53 (model save/load), 54 (dropout), and 58 (VGG16) are **not** done.
`utils/random.rs` documents a `dropout_with_mask` that does not exist yet —
that link resolves once step 54 lands.
