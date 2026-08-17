//! [`Vgg16`] — the classic 16-layer convolutional network (step 58).
//!
//! Port of `VGG16` in `vendor/dezero-python/dezero/models.py`.
//!
//! # Architecture only
//!
//! Python's constructor takes `pretrained=True`, downloads a 528 MB `.npz` of
//! ImageNet weights over HTTP, and loads it. That is **not** ported: there is
//! no HTTP client in this crate (divergence 25), and the file is in numpy's
//! format, which this port does not read (divergence 31). `Vgg16::new` builds
//! the architecture with freshly initialised weights.
//!
//! Loading weights that were saved *by this port* works normally, through
//! [`load_weights`](crate::load_weights).
//!
//! Python's `preprocess` (PIL resize, BGR reorder, mean subtraction) is also
//! absent — it is image handling, not a network.
//!
//! # Shape
//!
//! Thirteen convolutions in five blocks, each block ending in 2×2 max pooling,
//! then three fully-connected layers. Every convolution is 3×3, stride 1,
//! pad 1, so only the pooling changes the spatial size: a 224×224 input is
//! halved five times to 7×7, and 7·7·512 = 25088 is what `fc6` receives.

use crate::core::parameter::Parameter;
use crate::core::variable::Variable;
use crate::functions::activation::{dropout, relu};
use crate::functions::conv::pooling;
use crate::functions::shape::reshape;
use crate::layers::{Conv2dLayer, Layer, Linear};

/// VGG16 — Python's `dezero.models.VGG16`.
///
/// # Examples
///
/// ```
/// use dezero::{Layer, Vgg16};
///
/// let net = Vgg16::new();
/// // 13 convolutions + 3 fully-connected layers, W and b each.
/// assert_eq!(net.params().len(), 16 * 2);
/// ```
#[derive(Debug)]
pub struct Vgg16 {
    conv: Vec<Conv2dLayer>,
    fc6: Linear,
    fc7: Linear,
    fc8: Linear,
    dropout_ratio: f64,
}

/// Where the five pooling stages fall: after conv 2, 4, 7, 10 and 13.
const BLOCK_ENDS: [usize; 5] = [1, 3, 6, 9, 12];

/// Output channels of each of the thirteen convolutions.
const CHANNELS: [usize; 13] = [
    64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512,
];

impl Vgg16 {
    /// A freshly initialised network, classifying into 1000 categories.
    #[must_use]
    pub fn new() -> Self {
        Self::with_classes(1000)
    }

    /// A network with a different number of output classes — the one thing
    /// worth varying when the pretrained weights are not available anyway.
    #[must_use]
    pub fn with_classes(classes: usize) -> Self {
        Self {
            // Every convolution is 3x3, stride 1, pad 1; input channels are
            // settled by the first batch, so the same model accepts greyscale
            // or RGB without being told which.
            conv: CHANNELS
                .iter()
                .map(|&out| Conv2dLayer::new(out, 3, 1, 1))
                .collect(),
            fc6: Linear::new(4096),
            fc7: Linear::new(4096),
            fc8: Linear::new(classes),
            dropout_ratio: 0.5,
        }
    }

    /// The thirteen convolution layers, in order.
    #[must_use]
    pub fn convolutions(&self) -> &[Conv2dLayer] {
        &self.conv
    }

    /// The three fully-connected layers, in order.
    #[must_use]
    pub fn classifier(&self) -> [&Linear; 3] {
        [&self.fc6, &self.fc7, &self.fc8]
    }
}

impl Default for Vgg16 {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer for Vgg16 {
    fn own_params(&self) -> Vec<Parameter> {
        Vec::new()
    }

    fn sublayers(&self) -> Vec<&dyn Layer> {
        let mut out: Vec<&dyn Layer> = self.conv.iter().map(|c| c as &dyn Layer).collect();
        out.push(&self.fc6);
        out.push(&self.fc7);
        out.push(&self.fc8);
        out
    }

    /// # Panics
    ///
    /// Panics if `x` is not a 4-dimensional `(N, C, H, W)` batch, or if its
    /// spatial size is too small to survive five 2×2 poolings.
    fn forward(&self, x: &Variable) -> Variable {
        let batch = x
            .shape()
            .expect("dezero: VGG16 needs an input that holds data")[0];

        let mut h = x.clone();
        for (index, conv) in self.conv.iter().enumerate() {
            h = relu(&conv.forward(&h));
            if BLOCK_ENDS.contains(&index) {
                h = pooling(&h, 2, 2, 0);
            }
        }

        // Flatten every axis but the batch. Python writes `reshape(x, (N, -1))`;
        // this port has no `-1` placeholder (divergence 10), so the length is
        // computed instead.
        let flat: usize = h
            .shape()
            .expect("the convolution stack produces data")
            .iter()
            .skip(1)
            .product();
        h = reshape(&h, &[batch, flat]);

        h = dropout(&relu(&self.fc6.forward(&h)), self.dropout_ratio);
        h = dropout(&relu(&self.fc7.forward(&h)), self.dropout_ratio);
        self.fc8.forward(&h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_mode;
    use ndarray::{ArrayD, IxDyn};

    /// A batch small enough to run in a test but still large enough to survive
    /// five halvings: 32 -> 16 -> 8 -> 4 -> 2 -> 1.
    fn batch(n: usize, c: usize, side: usize) -> Variable {
        let len = n * c * side * side;
        #[allow(clippy::cast_precision_loss, reason = "small test indices")]
        let data: Vec<f64> = (0..len).map(|i| ((i % 17) as f64) / 17.0).collect();
        Variable::new(
            ArrayD::from_shape_vec(IxDyn(&[n, c, side, side]), data).expect("shape matches length"),
        )
    }

    #[test]
    fn the_stack_is_sixteen_weighted_layers() {
        let net = Vgg16::new();
        assert_eq!(net.convolutions().len(), 13);
        assert_eq!(net.classifier().len(), 3);
        assert_eq!(net.params().len(), 16 * 2, "W and b for each");
    }

    #[test]
    fn the_channel_progression_matches_the_reference() {
        let net = Vgg16::new();
        let widths: Vec<usize> = net
            .convolutions()
            .iter()
            .map(Conv2dLayer::out_channels)
            .collect();
        assert_eq!(
            widths,
            vec![
                64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512
            ]
        );
    }

    #[test]
    #[ignore = "full VGG16 forward is slow in debug builds"]
    fn a_forward_pass_produces_one_score_per_class() {
        let net = Vgg16::with_classes(10);
        let guard = test_mode(); // dropout off, so the result is deterministic
        let y = net.forward(&batch(1, 3, 32));
        drop(guard);

        assert_eq!(y.shape(), Some(vec![1, 10]));
    }

    /// Five 2x2 poolings halve the spatial size five times; nothing else does.
    /// Ignored by default: a full VGG16 forward pass in a debug build takes
    /// tens of seconds, and three of them turn a 2-second suite into a
    /// 2-minute one. Run with `cargo test -- --ignored`; CI does.
    #[test]
    #[ignore = "full VGG16 forward is slow in debug builds"]
    fn five_poolings_reduce_the_side_by_thirty_two() {
        let net = Vgg16::with_classes(4);
        let guard = test_mode();

        // 32 -> 1 after five halvings, so fc6 sees 1*1*512.
        // Deliberately the smallest input that survives five poolings: a 64px
        // batch exercises the same arithmetic and took a minute to run, which
        // is not worth it for a shape assertion.
        let y = net.forward(&batch(2, 3, 32));
        drop(guard);

        assert_eq!(y.shape(), Some(vec![2, 4]));
        assert_eq!(
            net.classifier()[0].in_size(),
            Some(512),
            "the flattened width follows from the pooling stages"
        );
    }

    #[test]
    fn the_input_channel_count_is_settled_by_the_first_batch() {
        let net = Vgg16::with_classes(4);
        assert_eq!(net.convolutions()[0].in_channels(), None);

        // Only the first convolution is exercised: it is what settles the
        // input channel count, and running the whole stack to learn that
        // costs twenty seconds in a debug build.
        net.convolutions()[0].forward(&batch(1, 1, 8)); // greyscale

        assert_eq!(
            net.convolutions()[0].in_channels(),
            Some(1),
            "the same model accepts greyscale without being told"
        );
    }

    /// Ignored by default: a full VGG16 forward pass in a debug build takes
    /// tens of seconds, and three of them turn a 2-second suite into a
    /// 2-minute one. Run with `cargo test -- --ignored`; CI does.
    #[test]
    #[ignore = "full VGG16 forward is slow in debug builds"]
    fn gradients_reach_every_one_of_the_sixteen_layers() {
        let net = Vgg16::with_classes(3);
        let guard = test_mode();
        let y = net.forward(&batch(1, 3, 32));
        drop(guard);

        crate::functions::reduce::sum_all(&y).backward();
        for (index, p) in net.params().iter().enumerate() {
            assert!(
                p.grad().is_some(),
                "parameter {index} took no gradient -- the stack is disconnected"
            );
        }
    }

    /// Dropout is active in training mode, so two passes differ; under
    /// `test_mode` they must not.
    /// Ignored by default: a full VGG16 forward pass in a debug build takes
    /// tens of seconds, and three of them turn a 2-second suite into a
    /// 2-minute one. Run with `cargo test -- --ignored`; CI does.
    #[test]
    #[ignore = "full VGG16 forward is slow in debug builds"]
    fn dropout_is_live_in_training_and_inert_in_test_mode() {
        let net = Vgg16::with_classes(4);
        let x = batch(1, 3, 32);

        let guard = test_mode();
        let a = net.forward(&x);
        let b = net.forward(&x);
        drop(guard);
        assert_eq!(a.data(), b.data(), "test mode is deterministic");

        let c = net.forward(&x);
        let d = net.forward(&x);
        assert_ne!(
            c.data(),
            d.data(),
            "training mode draws a fresh dropout mask each pass"
        );
    }
}
