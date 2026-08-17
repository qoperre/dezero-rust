# Step 53 — model save / load

**Status:** done

## The bug my own assumption caused

The first version of this module stored weights as JSON **numbers**, on the
stated reasoning that "`serde_json` emits the shortest round-trippable
representation of an `f64`". That reasoning was half right and the conclusion
was wrong. The round-trip test failed on the fourth weight of a two-layer net:

```
saved:  0.23346832377607019
loaded: 0.2334683237760702     <- one ULP away
```

Narrowing it down: `serde_json`'s **writer** is fine — it emits Rust's
shortest-round-trip form. Its **parser** is not correctly rounded. Measured
directly:

| value | Rust `parse` | `serde_json` |
|---|---|---|
| `0.23346832377607019` | exact | **1 ULP off** |
| `0.1 + 0.2` | exact | exact |
| `f64::MIN_POSITIVE` | exact | exact |
| `f64::MAX / 3.0` | exact | **1 ULP off** |

So it is not a rare pathology; it hits ordinary trained weights.

**Fix:** store the numbers as JSON *strings*. Rust's `f64` `Display` is
shortest-round-trip and its `str::parse` is correctly rounded, so going through
a string uses both and is exact. The file stays readable — `"0.1"` is no harder
to read than `0.1`.

This is exactly the class of bug the project's "no errors" bar exists to catch:
silent, one-bit, and it would have shown up much later as a model that scores
differently after a save/load cycle.

## Format

A JSON map from key to `{"shape": [...], "data": ["..."]}` — the same
rank-generic array shape the parity fixtures use, so one reader serves both.

Chosen over hand-rolling an `.npz` writer because an inspectable weights file
is worth a lot while a port is still being debugged. The cost is size, which
for a teaching framework is not the binding constraint.

Written through a `BTreeMap`, so two saves of one model are **byte-identical**
and two checkpoints diff cleanly.

## Keys

Python flattens to `l1/W` using the *field* name, which it gets from the
`__setattr__` interception Rust has no equivalent of. `Layer::named_params`
uses the sub-layer's **index** instead: `W`, `b`, `0/W`, `1/2/b`. Stable for a
given structure, which is all save/load needs. An unnamed parameter falls back
to its position, so keys stay unique either way.

## Behaviour at the edges

| case | behaviour | why |
|---|---|---|
| parameter with no data (lazy `W` before first forward) | skipped on save | matches Python's `if param is not None`; saving an untrained model is not an error |
| key in the file the model lacks | ignored | Python's `npz[key]` raises, but loading a checkpoint with an extra head into a smaller model is normal |
| stored shape disagrees with the parameter | `ShapeMismatch` error | loading wrong-shaped weights is a real mistake, not something to silently reshape around |
| stored shape disagrees with its own data length | `Corrupt` error | |
| a stored value that will not parse | `NotANumber` error | |

## Verification

Eight tests, including a full round trip through an `Mlp` that asserts the
restored model **computes the same output**, and the awkward-float test that
now pins the property the format rests on (`f64::MIN_POSITIVE`,
`f64::MAX / 3.0`, `0.1 + 0.2`, and a value near the smallest normal) with
`assert_eq!` — bit-exact, not merely close.

```
cargo test    645 passed
cargo clippy --all-targets --all-features -- -D warnings   0 errors
cargo fmt --all --check                                    clean
```

`serde` and `serde_json` moved from dev-dependencies to runtime dependencies.
Divergence 31.
