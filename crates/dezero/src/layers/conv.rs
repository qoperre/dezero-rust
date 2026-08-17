//! The [`Conv2dLayer`]: a convolution that owns its filters (step 57–58).
//!
//! Port of `Conv2d` in `vendor/dezero-python/dezero/layers.py`. The
//! mathematics is [`conv2d`](crate::conv2d); what this adds is the two
//! [`Parameter`]s and the rule for when `W` comes into existence.
//!
//! Named `Conv2dLayer` rather than `Conv2d` because
//! [`Conv2d`](crate::functions::conv::Conv2d) is already the `Op`. Python can
//! reuse the name across `dezero.layers` and `dezero.functions`; a single Rust
//! crate root cannot.
//!
//! # Lazily-shaped filters
//!
//! As with [`Linear`](crate::Linear), `in_channels` may be omitted and settled
//! by the first batch. `W` is `(out_channels, in_channels, kh, kw)`, drawn as
//! `randn * sqrt(1 / (C·KH·KW))` — the same fan-in scaling, counting every
//! element of the receptive field.

use std::cell::Cell;

use ndarray::{ArrayD, IxDyn};

use crate::core::parameter::Parameter;
use crate::core::variable::Variable;
use crate::functions::conv::conv2d;
use crate::layers::Layer;
use crate::utils::conv::Pair;
use crate::utils::random::randn;

/// A 2-D convolution layer — Python's `L.Conv2d`.
#[derive(Debug)]
pub struct Conv2dLayer {
    weight: Parameter,
    bias: Option<Parameter>,
    in_channels: Cell<Option<usize>>,
    out_channels: usize,
    kernel: Pair,
    stride: Pair,
    pad: Pair,
}

impl Conv2dLayer {
    /// A layer whose input channel count is settled by the first batch —
    /// Python's `L.Conv2d(out_channels, kernel_size, stride, pad)`.
    #[must_use]
    pub fn new(
        out_channels: usize,
        kernel: impl Into<Pair>,
        stride: impl Into<Pair>,
        pad: impl Into<Pair>,
    ) -> Self {
        Self {
            weight: Parameter::named(None, "W"),
            bias: Some(Parameter::named(
                Some(ArrayD::zeros(IxDyn(&[out_channels]))),
                "b",
            )),
            in_channels: Cell::new(None),
            out_channels,
            kernel: kernel.into(),
            stride: stride.into(),
            pad: pad.into(),
        }
    }

    /// A layer whose filters exist immediately — Python's
    /// `L.Conv2d(..., in_channels=in_channels)`.
    #[must_use]
    pub fn with_in_channels(
        in_channels: usize,
        out_channels: usize,
        kernel: impl Into<Pair>,
        stride: impl Into<Pair>,
        pad: impl Into<Pair>,
    ) -> Self {
        let layer = Self::new(out_channels, kernel, stride, pad);
        layer.init_weight(in_channels);
        layer
    }

    /// Drops the bias — Python's `nobias=True`.
    #[must_use]
    pub fn without_bias(mut self) -> Self {
        self.bias = None;
        self
    }

    /// The filter bank, `W`, shaped `(out_channels, in_channels, kh, kw)`.
    #[must_use]
    pub fn weight(&self) -> &Parameter {
        &self.weight
    }

    /// The bias, or `None` for a layer built with
    /// [`without_bias`](Self::without_bias).
    #[must_use]
    pub fn bias(&self) -> Option<&Parameter> {
        self.bias.as_ref()
    }

    /// The number of input channels, once it is known.
    #[must_use]
    pub fn in_channels(&self) -> Option<usize> {
        self.in_channels.get()
    }

    /// The number of output channels, fixed at construction.
    #[must_use]
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Fills in `W` — Python's `_init_W`.
    ///
    /// # Panics
    ///
    /// Panics if `in_channels` is 0.
    fn init_weight(&self, in_channels: usize) {
        assert!(
            in_channels > 0,
            "dezero: Conv2d cannot initialise a filter with 0 input channels"
        );
        let (kh, kw) = (self.kernel.height, self.kernel.width);
        #[allow(
            clippy::cast_precision_loss,
            reason = "a receptive field of 2^53 elements is not representable in \
                      memory long before it is representable imprecisely"
        )]
        let scale = (1.0 / (in_channels * kh * kw) as f64).sqrt();
        self.weight
            .set_data(randn(&[self.out_channels, in_channels, kh, kw]).mapv(|v| v * scale));
        self.in_channels.set(Some(in_channels));
    }
}

impl Layer for Conv2dLayer {
    fn own_params(&self) -> Vec<Parameter> {
        match &self.bias {
            Some(bias) => vec![self.weight.clone(), bias.clone()],
            None => vec![self.weight.clone()],
        }
    }

    /// Convolves `x`, initialising the filters on the first call.
    ///
    /// # Panics
    ///
    /// Panics if `x` is not a 4-dimensional `(N, C, H, W)` tensor, if its
    /// channel count disagrees with an already-initialised `W`, or if it holds
    /// no data.
    fn forward(&self, x: &Variable) -> Variable {
        if self.weight.data().is_none() {
            let shape = x
                .shape()
                .expect("dezero: Conv2d needs an input that holds data");
            assert_eq!(
                shape.len(),
                4,
                "dezero: Conv2d expects an (N, C, H, W) tensor, got {shape:?}"
            );
            self.init_weight(shape[1]);
        }
        conv2d(x, &self.weight, self.bias.as_deref(), self.stride, self.pad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functions::reduce::sum_all;

    fn image(n: usize, c: usize, h: usize, w: usize) -> Variable {
        let len = n * c * h * w;
        #[allow(clippy::cast_precision_loss, reason = "small test indices")]
        let data: Vec<f64> = (0..len).map(|i| (i as f64).sin()).collect();
        Variable::new(
            ArrayD::from_shape_vec(IxDyn(&[n, c, h, w]), data).expect("shape matches length"),
        )
    }

    #[test]
    fn an_eager_layer_has_correctly_shaped_filters() {
        let layer = Conv2dLayer::with_in_channels(3, 8, 3, 1, 1);
        assert_eq!(layer.weight().shape(), Some(vec![8, 3, 3, 3]));
        assert_eq!(layer.bias().expect("bias").shape(), Some(vec![8]));
        assert_eq!(layer.in_channels(), Some(3));
    }

    #[test]
    fn a_lazy_layer_settles_on_its_first_batch() {
        let layer = Conv2dLayer::new(8, 3, 1, 1);
        assert!(layer.weight().data().is_none());
        assert_eq!(layer.in_channels(), None);

        let y = layer.forward(&image(2, 3, 5, 5));
        assert_eq!(layer.weight().shape(), Some(vec![8, 3, 3, 3]));
        assert_eq!(layer.in_channels(), Some(3));
        // stride 1, pad 1, kernel 3 keeps the spatial size.
        assert_eq!(y.shape(), Some(vec![2, 8, 5, 5]));
    }

    #[test]
    fn stride_and_pad_shape_the_output() {
        let layer = Conv2dLayer::with_in_channels(3, 4, 3, 2, 1);
        let y = layer.forward(&image(1, 3, 7, 7));
        // (7 + 2*1 - 3) / 2 + 1 = 4
        assert_eq!(y.shape(), Some(vec![1, 4, 4, 4]));
    }

    #[test]
    fn a_bias_free_layer_registers_only_its_filters() {
        let layer = Conv2dLayer::with_in_channels(3, 4, 3, 1, 1).without_bias();
        assert_eq!(layer.params().len(), 1);
        assert!(layer.bias().is_none());
    }

    #[test]
    fn gradients_reach_both_parameters() {
        let layer = Conv2dLayer::with_in_channels(2, 3, 3, 1, 1);
        sum_all(&layer.forward(&image(1, 2, 4, 4))).backward();
        for p in layer.params() {
            assert!(p.grad().is_some());
        }
    }

    /// The fan-in scaling counts every element of the receptive field, not just
    /// the channel count -- a 5x5 filter must come out smaller than a 3x3 one.
    #[test]
    fn the_initialisation_scale_counts_the_whole_receptive_field() {
        let small = Conv2dLayer::with_in_channels(4, 16, 3, 1, 1);
        let large = Conv2dLayer::with_in_channels(4, 16, 7, 1, 3);

        let spread = |layer: &Conv2dLayer| {
            let w = layer.weight().data().expect("data");
            #[allow(clippy::cast_precision_loss, reason = "element counts are small")]
            let n = w.len() as f64;
            (w.iter().map(|v| v * v).sum::<f64>() / n).sqrt()
        };

        assert!(
            spread(&large) < spread(&small),
            "a wider receptive field divides by more, so its weights start smaller"
        );
    }
}
