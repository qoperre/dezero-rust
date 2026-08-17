//! Saving and loading a layer's weights (step 53).
//!
//! Port of `Layer.save_weights` / `Layer.load_weights` in
//! `vendor/dezero-python/dezero/layers.py`.
//!
//! # Format
//!
//! Python writes `np.savez_compressed` — a zip of `.npy` members. This port
//! writes **JSON**: a map from [`Layer::named_params`] key to a rank-generic
//! `{"shape": [...], "data": [...]}` array, C order. That is the same shape
//! the project's parity fixtures use, so one reader serves both.
//!
//! JSON was chosen over hand-rolling an `.npz` writer because a weights file
//! you can open in an editor is worth a great deal while a port is still being
//! debugged. The cost is size, which for a teaching framework's models is not
//! the binding constraint.
//!
//! # Why the numbers are strings
//!
//! Weights are stored as JSON **strings**, not JSON numbers, because
//! `serde_json`'s float parser is not correctly rounded: writing `0.23346832377607019`
//! and reading it back yields a value one ULP away. That was measured, not
//! assumed — the first version of this module stored numbers and its
//! round-trip test failed on the fourth weight of a two-layer net.
//!
//! Rust's own `f64` formatting is shortest-round-trip and its `str::parse` is
//! correctly rounded, so going through a string uses both and is exact. A test
//! pins that on deliberately awkward values, including `f64::MIN_POSITIVE` and
//! `f64::MAX / 3.0`.
//!
//! The file stays readable: `"0.1"` is no harder to read than `0.1`.
//!
//! The two formats do not interoperate, which is recorded as divergence 31.
//!
//! # Partially-initialised layers
//!
//! A lazily-shaped [`Linear`](crate::Linear) has a `W` with no data until its
//! first forward pass. Python skips such parameters when saving
//! (`if param is not None`) and this does too, so saving an untrained model is
//! not an error. Loading only writes the keys the file actually holds.

use std::collections::BTreeMap;
use std::path::Path;

use ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};

use crate::layers::Layer;

/// One array, stored rank-generically so a 0-d scalar and a 4-d tensor look
/// the same to the reader.
#[derive(Debug, Serialize, Deserialize)]
struct StoredArray {
    shape: Vec<usize>,
    /// Formatted with Rust's `f64` `Display` and read back with its `parse`,
    /// both of which are exact. See the module docs for why not a JSON number.
    data: Vec<String>,
}

/// Anything that can go wrong saving or loading weights.
#[derive(Debug)]
pub enum WeightsError {
    /// The file could not be read or written.
    Io(std::io::Error),
    /// The file is not valid JSON, or not in this format.
    Format(serde_json::Error),
    /// A stored array's `shape` disagrees with the length of its `data`.
    Corrupt {
        /// The parameter key whose array is inconsistent.
        key: String,
        /// The shape the file claims.
        shape: Vec<usize>,
        /// The number of elements actually stored.
        len: usize,
    },
    /// A stored value is not a number this platform can parse.
    NotANumber {
        /// The parameter key.
        key: String,
        /// The text that failed to parse.
        value: String,
    },
    /// A stored array's shape disagrees with the parameter it would fill.
    ShapeMismatch {
        /// The parameter key.
        key: String,
        /// The shape the parameter already has.
        expected: Vec<usize>,
        /// The shape the file holds.
        found: Vec<usize>,
    },
}

impl std::fmt::Display for WeightsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "weights file could not be read or written: {e}"),
            Self::Format(e) => write!(f, "weights file is not in the expected format: {e}"),
            Self::Corrupt { key, shape, len } => write!(
                f,
                "weights file is corrupt: {key} claims shape {shape:?} but holds {len} elements"
            ),
            Self::NotANumber { key, value } => {
                write!(
                    f,
                    "weights file is corrupt: {key} holds {value:?}, which is not a number"
                )
            }
            Self::ShapeMismatch {
                key,
                expected,
                found,
            } => write!(
                f,
                "weights file does not fit this model: {key} is {expected:?} here \
                 but {found:?} in the file"
            ),
        }
    }
}

impl std::error::Error for WeightsError {}

impl From<std::io::Error> for WeightsError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for WeightsError {
    fn from(e: serde_json::Error) -> Self {
        Self::Format(e)
    }
}

/// Writes every initialised parameter of `layer` to `path` as JSON.
///
/// Parameters that hold no data yet — a lazily-shaped weight before its first
/// forward pass — are skipped rather than treated as an error, matching
/// Python's `if param is not None`.
///
/// A [`BTreeMap`] keeps the output key-ordered, so two saves of the same model
/// produce byte-identical files and a diff between two checkpoints is readable.
///
/// # Errors
///
/// Returns [`WeightsError::Io`] if the file cannot be written, or
/// [`WeightsError::Format`] if serialisation fails.
pub fn save_weights(layer: &dyn Layer, path: impl AsRef<Path>) -> Result<(), WeightsError> {
    let stored: BTreeMap<String, StoredArray> = layer
        .named_params()
        .into_iter()
        .filter_map(|(key, param)| {
            param.data().map(|data| {
                (
                    key,
                    StoredArray {
                        shape: data.shape().to_vec(),
                        data: data.iter().map(ToString::to_string).collect(),
                    },
                )
            })
        })
        .collect();

    std::fs::write(path, serde_json::to_string_pretty(&stored)?)?;
    Ok(())
}

/// Fills `layer`'s parameters from a file written by [`save_weights`].
///
/// Only the keys present in the file are written, so a file saved from a
/// partially-initialised model loads without complaint. A key in the file that
/// this layer does not have is ignored — Python's `npz[key]` would raise, but
/// a checkpoint carrying an extra head is a normal thing to load into a
/// smaller model.
///
/// A parameter that already has data must match the stored shape; loading
/// weights of the wrong shape is a real mistake and is reported rather than
/// silently reshaping the model.
///
/// # Errors
///
/// Returns [`WeightsError::Io`] if the file cannot be read,
/// [`WeightsError::Format`] if it is not valid JSON in this format,
/// [`WeightsError::Corrupt`] if a stored shape disagrees with its own data
/// length, or [`WeightsError::ShapeMismatch`] if it disagrees with the
/// parameter it would fill.
pub fn load_weights(layer: &dyn Layer, path: impl AsRef<Path>) -> Result<(), WeightsError> {
    let text = std::fs::read_to_string(path)?;
    let stored: BTreeMap<String, StoredArray> = serde_json::from_str(&text)?;

    for (key, param) in layer.named_params() {
        let Some(entry) = stored.get(&key) else {
            continue;
        };

        let expected_len: usize = entry.shape.iter().product();
        if expected_len != entry.data.len() {
            return Err(WeightsError::Corrupt {
                key,
                shape: entry.shape.clone(),
                len: entry.data.len(),
            });
        }

        if let Some(current) = param.shape()
            && current != entry.shape
        {
            return Err(WeightsError::ShapeMismatch {
                key,
                expected: current,
                found: entry.shape.clone(),
            });
        }

        let mut values = Vec::with_capacity(entry.data.len());
        for text in &entry.data {
            values.push(text.parse::<f64>().map_err(|_| WeightsError::NotANumber {
                key: key.clone(),
                value: text.clone(),
            })?);
        }

        let array = ArrayD::from_shape_vec(IxDyn(&entry.shape), values)
            .expect("the shape and data length were just checked against each other");
        param.set_data(array);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Variable;
    use crate::layers::Linear;
    use crate::models::Mlp;
    use ndarray::arr2;

    /// A temp path unique to this test binary and call site.
    fn temp_path(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("dezero_weights_{}_{tag}.json", std::process::id()));
        p
    }

    fn input() -> Variable {
        Variable::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn())
    }

    #[test]
    fn a_round_trip_restores_every_weight_exactly() {
        let source = Mlp::new(&[4, 2]);
        source.forward(&input()); // shape the lazy weights

        let path = temp_path("roundtrip");
        save_weights(&source, &path).expect("save");

        let target = Mlp::new(&[4, 2]);
        target.forward(&input());
        load_weights(&target, &path).expect("load");

        for (a, b) in source.params().iter().zip(target.params()) {
            assert_eq!(
                a.data(),
                b.data(),
                "every parameter survives the round trip"
            );
        }
        // ...and the restored model computes the same thing.
        assert_eq!(
            source.forward(&input()).data(),
            target.forward(&input()).data()
        );

        let _ = std::fs::remove_file(&path);
    }

    /// `serde_json` emits the shortest round-trippable form of an `f64`, so
    /// awkward values survive. This is the property the format choice rests on.
    #[test]
    fn awkward_floats_survive_the_json_round_trip() {
        let layer = Linear::with_in_size(1, 4);
        let awkward = [
            f64::MIN_POSITIVE,
            0.1 + 0.2,
            -1.234_567_890_123_456_7e-17,
            f64::MAX / 3.0,
        ];
        layer.weight().set_data(
            ndarray::Array2::from_shape_vec((1, 4), awkward.to_vec())
                .unwrap()
                .into_dyn(),
        );

        let path = temp_path("floats");
        save_weights(&layer, &path).expect("save");

        let target = Linear::with_in_size(1, 4);
        load_weights(&target, &path).expect("load");

        let restored = target.weight().data().expect("data");
        for (got, want) in restored.iter().zip(&awkward) {
            assert_eq!(got, want, "bit-exact, not merely close");
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_uninitialised_parameter_is_skipped_rather_than_failing() {
        let layer = Linear::new(4); // W has no data yet
        assert!(layer.weight().data().is_none());

        let path = temp_path("lazy");
        save_weights(&layer, &path).expect("saving an unshaped layer is fine");

        let text = std::fs::read_to_string(&path).expect("read");
        assert!(text.contains("\"b\""), "the bias exists and is written");
        assert!(!text.contains("\"W\""), "the unshaped weight is skipped");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn nested_layers_get_distinct_path_keys() {
        let model = Mlp::new(&[4, 2]);
        model.forward(&input());

        let keys: Vec<String> = model.named_params().into_iter().map(|(k, _)| k).collect();
        assert_eq!(keys.len(), 4, "two Linear layers, W and b each");
        assert!(keys.iter().any(|k| k == "0/W"));
        assert!(keys.iter().any(|k| k == "1/b"));

        let unique: std::collections::HashSet<&String> = keys.iter().collect();
        assert_eq!(unique.len(), keys.len(), "no key collides with another");
    }

    #[test]
    fn loading_the_wrong_shape_is_reported_not_absorbed() {
        let wide = Linear::with_in_size(3, 4);
        let path = temp_path("mismatch");
        save_weights(&wide, &path).expect("save");

        let narrow = Linear::with_in_size(2, 4);
        let err = load_weights(&narrow, &path).expect_err("the shapes disagree");
        assert!(
            matches!(err, WeightsError::ShapeMismatch { .. }),
            "got {err:?}"
        );
        assert!(err.to_string().contains("does not fit this model"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_corrupt_shape_is_reported() {
        let path = temp_path("corrupt");
        std::fs::write(&path, r#"{"W": {"shape": [2, 3], "data": ["1.0", "2.0"]}}"#)
            .expect("write");

        let layer = Linear::new(3);
        let err = load_weights(&layer, &path).expect_err("6 elements claimed, 2 stored");
        assert!(matches!(err, WeightsError::Corrupt { .. }), "got {err:?}");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_key_the_model_does_not_have_is_ignored() {
        let path = temp_path("extra");
        std::fs::write(
            &path,
            r#"{"b": {"shape": [4], "data": ["1.0", "2.0", "3.0", "4.0"]},
                "somewhere/else": {"shape": [1], "data": ["9.0"]}}"#,
        )
        .expect("write");

        let layer = Linear::new(4);
        load_weights(&layer, &path).expect("the extra key is not this model's problem");
        assert_eq!(
            layer.bias().expect("bias").data(),
            Some(ndarray::arr1(&[1.0, 2.0, 3.0, 4.0]).into_dyn())
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn two_saves_of_one_model_are_byte_identical() {
        let model = Mlp::new(&[4, 2]);
        model.forward(&input());

        let a = temp_path("stable_a");
        let b = temp_path("stable_b");
        save_weights(&model, &a).expect("save");
        save_weights(&model, &b).expect("save");

        assert_eq!(
            std::fs::read_to_string(&a).expect("read"),
            std::fs::read_to_string(&b).expect("read"),
            "a BTreeMap keeps the key order stable, so checkpoints diff cleanly"
        );

        let _ = std::fs::remove_file(&a);
        let _ = std::fs::remove_file(&b);
    }
}
