# Step 58 — VGG16

**Status:** done (architecture; pretrained weights out of scope)

## What was needed first

Step 57 gave `conv2d` the *function*. VGG16 needs `Conv2d` the *layer* — one
that owns its filters — which did not exist yet. So this step is two pieces:

**`layers/conv.rs` — `Conv2dLayer`.** Named that, not `Conv2d`, because
`functions::conv::Conv2d` is already the `Op`. Python reuses the name across
`dezero.layers` and `dezero.functions`; one Rust crate root cannot.

Filters are `(out_channels, in_channels, kh, kw)`, drawn as
`randn * sqrt(1 / (C·KH·KW))` — the same fan-in scaling `Linear` uses, counting
every element of the receptive field. A test pins that a 7×7 filter starts
smaller than a 3×3 one, since dropping the kernel size from that denominator
is an easy and silent mistake.

`in_channels` may be omitted and settled by the first batch, like `Linear`.

**`models/vgg.rs` — `Vgg16`.** Thirteen 3×3 convolutions in five blocks, each
block ending in 2×2 max pooling, then three fully-connected layers with dropout
between them. Only the pooling changes the spatial size, so a 224×224 input is
halved five times to 7×7 and `fc6` receives 7·7·512 = 25088.

Python writes `reshape(x, (N, -1))` to flatten; this port has no `-1`
placeholder (divergence 10), so the length is computed from the shape.

## Not ported

`pretrained=True` downloads a 528 MB `.npz` of ImageNet weights over HTTP.
There is no HTTP client in this crate (divergence 25) and the file is numpy's
format, which this port does not read (divergence 31). `Vgg16::new` builds the
architecture with fresh weights; loading weights saved *by this port* works
normally. Python's `preprocess` (PIL resize, BGR reorder, mean subtraction) is
image handling, not a network, and is also absent. Divergence 32.

## The test-speed problem, and what was done about it

A full VGG16 forward pass in a **debug** build takes tens of seconds. Four such
tests turned a 2-second suite into a 110-second one — slow enough that people
stop running it, which is worse than the tests not existing.

Three things were done:

1. The four full-forward tests are `#[ignore]`d with a stated reason. They run
   under `cargo test -- --ignored`.
2. CI runs them **in release mode**, where the same four take **6.4 seconds**
   instead of 110. A new workflow step does exactly that.
3. `the_input_channel_count_is_settled_by_the_first_batch` was rewritten to
   exercise only the *first* convolution, which is what settles the channel
   count. Same assertion, 0.01 s instead of 20.

The default suite is back to 1.7 seconds and still covers the structural
properties: layer count, channel progression, and lazy channel settling.

## Verification

```
cargo test                        655 passed, 4 ignored, in 1.7s
cargo test --release -- --ignored   4 passed, in 6.4s
cargo clippy --all-targets --all-features -- -D warnings   0 errors
cargo fmt --all --check                                    clean
```

The ignored four assert: a forward pass produces one score per class; five
poolings reduce the side by 32; gradients reach **every one of the sixteen**
weighted layers (a disconnected stack would show up here and nowhere else);
and dropout is live in training mode but inert under `test_mode`.
