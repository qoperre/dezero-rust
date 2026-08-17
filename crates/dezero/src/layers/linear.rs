//! The [`Linear`] layer: an affine transform that owns its weights (step 44).
//!
//! Port of `Linear` in `vendor/dezero-python/dezero/layers.py`. The mathematics
//! is [`linear`], one module over; what this adds is the two
//! [`Parameter`]s and the rule for when `W` comes into existence.
//!
//! # Lazily-shaped weights
//!
//! Python lets `in_size` be omitted:
//!
//! ```python
//! self.W = Parameter(None, name='W')
//! if self.in_size is not None:
//!     self._init_W()
//! ...
//! def forward(self, x):
//!     if self.W.data is None:
//!         self.in_size = x.shape[1]
//!         self._init_W(xp)
//! ```
//!
//! so `L.Linear(10)` can be written before anyone knows what feeds it, and the
//! first batch settles the question. The port supports **both** paths:
//! [`Linear::new`] is the lazy one and [`Linear::with_in_size`] the eager one.
//! This is why [`Variable`]'s `data` has been an `Option` since the first
//! commit rather than being retrofitted here (`docs/ARCHITECTURE.md`).
//!
//! Filling `W` in later is safe for everything downstream because the parameter
//! *object* exists from construction: an
//! [`Optimizer`](crate::Optimizer) that registered the layer while `W` was
//! still empty holds the very same `Rc` and sees the weights arrive. An
//! optimizer skips a parameter with no gradient, and an uninitialised `W` has
//! none, so the window between the two is handled rather than merely survived.
//!
//! # Initialisation
//!
//! Python draws `randn(in, out) * sqrt(1/in)` from numpy's global stream. The
//! port draws the same distribution from [`crate::randn`] — a *different*
//! stream, necessarily, since no Rust generator reproduces the Mersenne
//! Twister. Any test that needs exact weights must set them explicitly, which
//! is precisely what the parity fixtures do.

use std::cell::Cell;

use ndarray::{ArrayD, IxDyn};

use crate::core::parameter::Parameter;
use crate::core::variable::Variable;
use crate::functions::matmul::linear;
use crate::layers::Layer;
use crate::utils::random::randn;

/// A fully connected layer, `y = x W + b` — Python's `dezero.layers.Linear`.
///
/// # Examples
///
/// ```
/// use dezero::{Layer, Linear, Variable};
/// use ndarray::arr2;
///
/// // Eager: the weight exists straight away.
/// let eager = Linear::with_in_size(3, 2);
/// assert_eq!(eager.weight().shape(), Some(vec![3, 2]));
///
/// // Lazy: it does not, until a batch reveals the input width.
/// let lazy = Linear::new(2);
/// assert!(lazy.weight().data().is_none());
/// assert_eq!(lazy.in_size(), None);
///
/// let x = Variable::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn());
/// let y = lazy.forward(&x);
///
/// assert_eq!(lazy.in_size(), Some(3));
/// assert_eq!(lazy.weight().shape(), Some(vec![3, 2]));
/// assert_eq!(y.shape(), Some(vec![2, 2]));
/// ```
#[derive(Debug)]
pub struct Linear {
    weight: Parameter,
    bias: Option<Parameter>,
    /// `None` until a forward pass reveals it; a [`Cell`] because `forward`
    /// takes `&self`.
    in_size: Cell<Option<usize>>,
    out_size: usize,
}

impl Linear {
    /// A layer whose weight shape is decided by the first input — Python's
    /// `L.Linear(out_size)`.
    ///
    /// The weight has no data until [`forward`](Layer::forward) runs; the bias,
    /// whose shape depends only on `out_size`, is created immediately and
    /// zero-filled, exactly as Python does.
    #[must_use]
    pub fn new(out_size: usize) -> Self {
        Self {
            weight: Parameter::named(None, "W"),
            bias: Some(Parameter::named(
                Some(ArrayD::zeros(IxDyn(&[out_size]))),
                "b",
            )),
            in_size: Cell::new(None),
            out_size,
        }
    }

    /// A layer with its weight initialised up front — Python's
    /// `L.Linear(out_size, in_size=in_size)`.
    ///
    /// # Panics
    ///
    /// Panics if `in_size` is 0: the initialiser scales by `1/sqrt(in_size)`.
    #[must_use]
    pub fn with_in_size(in_size: usize, out_size: usize) -> Self {
        let layer = Self::new(out_size);
        layer.init_weight(in_size);
        layer
    }

    /// Drops the bias — Python's `L.Linear(out_size, nobias=True)`.
    ///
    /// A builder step, so it chains: `Linear::new(10).without_bias()`.
    #[must_use]
    pub fn without_bias(mut self) -> Self {
        self.bias = None;
        self
    }

    /// The weight parameter, `W`.
    ///
    /// It holds no data before the first forward pass of a layer built by
    /// [`new`](Self::new).
    #[must_use]
    pub fn weight(&self) -> &Parameter {
        &self.weight
    }

    /// The bias parameter, `b`, or `None` for a layer built with
    /// [`without_bias`](Self::without_bias).
    #[must_use]
    pub fn bias(&self) -> Option<&Parameter> {
        self.bias.as_ref()
    }

    /// The number of input features, once it is known.
    #[must_use]
    pub fn in_size(&self) -> Option<usize> {
        self.in_size.get()
    }

    /// The number of output features, fixed at construction.
    #[must_use]
    pub fn out_size(&self) -> usize {
        self.out_size
    }

    /// Fills in `W` — Python's `_init_W`.
    ///
    /// `randn(in, out) * sqrt(1/in)`: the scale keeps the variance of the
    /// output independent of the fan-in, which is what stops a deep stack from
    /// saturating on its first forward pass.
    ///
    /// # Panics
    ///
    /// Panics if `in_size` is 0.
    fn init_weight(&self, in_size: usize) {
        assert!(
            in_size > 0,
            "dezero: Linear cannot initialise a weight with an input size of 0"
        );
        #[allow(
            clippy::cast_precision_loss,
            reason = "a layer with 2^53 inputs is not representable in memory \
                      long before it is representable imprecisely"
        )]
        let scale = (1.0 / in_size as f64).sqrt();
        self.weight
            .set_data(randn(&[in_size, self.out_size]).mapv(|v| v * scale));
        self.in_size.set(Some(in_size));
    }
}

impl Layer for Linear {
    fn own_params(&self) -> Vec<Parameter> {
        match &self.bias {
            Some(bias) => vec![self.weight.clone(), bias.clone()],
            None => vec![self.weight.clone()],
        }
    }

    /// `y = x W + b`, initialising `W` on the first call if it is still empty.
    ///
    /// # Panics
    ///
    /// Panics if `x` is not a 2-dimensional `(batch, features)` matrix, if its
    /// width disagrees with an already-initialised `W`, or if it holds no data.
    fn forward(&self, x: &Variable) -> Variable {
        if self.weight.data().is_none() {
            let shape = x
                .shape()
                .unwrap_or_else(|| panic!("dezero: Linear needs an input that holds data"));
            let [_, features] = shape[..] else {
                panic!(
                    "dezero: Linear needs a 2-dimensional (batch, features) input to size its \
                     weight from, got shape {shape:?}"
                );
            };
            self.init_weight(features);
        }
        linear(x, &self.weight, self.bias.as_deref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functions::reduce::sum_all;
    use crate::utils::random::{Rng, seed};
    use ndarray::{arr1, arr2};

    fn batch() -> Variable {
        Variable::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn())
    }

    // -- construction ------------------------------------------------------

    #[test]
    fn the_eager_path_shapes_everything_up_front() {
        let layer = Linear::with_in_size(3, 4);
        assert_eq!(layer.in_size(), Some(3));
        assert_eq!(layer.out_size(), 4);
        assert_eq!(layer.weight().shape(), Some(vec![3, 4]));
        assert_eq!(
            layer.bias().and_then(|b| b.shape()),
            Some(vec![4]),
            "the bias only ever depended on out_size"
        );
        assert!(
            layer
                .bias()
                .and_then(|b| b.data())
                .is_some_and(|d| d.iter().all(|v| *v == 0.0)),
            "Python zero-fills the bias"
        );
    }

    #[test]
    fn the_lazy_path_defers_only_the_weight() {
        let layer = Linear::new(4);
        assert_eq!(layer.in_size(), None);
        assert!(layer.weight().data().is_none());
        assert_eq!(layer.bias().and_then(|b| b.shape()), Some(vec![4]));

        layer.forward(&batch());
        assert_eq!(layer.in_size(), Some(3));
        assert_eq!(layer.weight().shape(), Some(vec![3, 4]));
    }

    /// The identity that makes lazy initialisation safe: the parameter object
    /// never changes, only its contents. Anything holding the parameter -- an
    /// optimizer, say -- keeps working across the transition.
    #[test]
    fn lazy_initialisation_preserves_parameter_identity() {
        let layer = Linear::new(2);
        let registered = layer.params();
        let weight_id = layer.weight().id();

        layer.forward(&batch());

        assert_eq!(layer.weight().id(), weight_id);
        assert_eq!(registered[0].id(), weight_id);
        assert!(
            registered[0].data().is_some(),
            "the handle taken before initialisation sees the weights arrive"
        );
    }

    #[test]
    fn the_weight_is_initialised_only_once() {
        let layer = Linear::new(2);
        layer.forward(&batch());
        let first = layer.weight().data();
        layer.forward(&batch());
        assert_eq!(layer.weight().data(), first, "the second pass reuses W");
    }

    #[test]
    fn without_bias_drops_the_bias_everywhere() {
        let layer = Linear::with_in_size(3, 4).without_bias();
        assert!(layer.bias().is_none());
        assert_eq!(layer.params().len(), 1);

        let y = layer.forward(&batch());
        assert_eq!(y.shape(), Some(vec![2, 4]));
    }

    // -- registration ------------------------------------------------------

    #[test]
    fn own_params_lists_the_weight_and_the_bias() {
        let layer = Linear::with_in_size(3, 4);
        let params = layer.params();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].id(), layer.weight().id());
        assert_eq!(params[1].id(), layer.bias().expect("bias").id());
        assert_eq!(params[0].name().as_deref(), Some("W"));
        assert_eq!(params[1].name().as_deref(), Some("b"));
        assert!(layer.sublayers().is_empty());
    }

    #[test]
    fn params_are_listed_even_before_the_weight_exists() {
        // The optimizer has to be able to register a lazily-shaped layer.
        let layer = Linear::new(4);
        assert_eq!(layer.params().len(), 2);
        assert!(layer.params()[0].data().is_none());
    }

    // -- forward / backward ------------------------------------------------

    #[test]
    fn forward_matches_the_free_function_on_its_own_parameters() {
        let layer = Linear::with_in_size(3, 4);
        let x = batch();
        let expected = crate::linear(&x, layer.weight(), layer.bias().map(|b| &**b));
        assert_eq!(layer.forward(&x).data(), expected.data());
    }

    #[test]
    fn backward_reaches_both_parameters() {
        let layer = Linear::with_in_size(3, 4);
        let x = batch();
        sum_all(&layer.forward(&x)).backward();

        assert_eq!(
            layer.weight().grad().and_then(|g| g.shape()),
            Some(vec![3, 4])
        );
        assert_eq!(
            layer.bias().and_then(|b| b.grad()).and_then(|g| g.data()),
            Some(arr1(&[2.0, 2.0, 2.0, 2.0]).into_dyn()),
            "one row of the seed per batch element"
        );
        assert_eq!(x.grad().and_then(|g| g.shape()), Some(vec![2, 3]));
    }

    #[test]
    fn cleargrads_clears_both_parameters() {
        let layer = Linear::with_in_size(3, 4);
        sum_all(&layer.forward(&batch())).backward();
        assert!(layer.params().iter().all(|p| p.grad().is_some()));
        layer.cleargrads();
        assert!(layer.params().iter().all(|p| p.grad().is_none()));
    }

    // -- initialisation --------------------------------------------------

    #[test]
    fn the_initialiser_scales_by_one_over_root_fan_in() {
        // With 400 inputs the weights should have a standard deviation of
        // 1/sqrt(400) = 0.05, not 1. Seeded so the moments below are the same
        // in every run, whatever else has drawn from the stream first.
        seed(17);
        let layer = Linear::with_in_size(400, 20);
        let w = layer.weight().data().expect("initialised");
        assert_eq!(w.shape(), &[400, 20]);

        let count = f64::from(u32::try_from(w.len()).expect("fits"));
        let mean = w.sum() / count;
        let variance = w.mapv(|v| (v - mean).powi(2)).sum() / count;
        assert!(mean.abs() < 0.005, "mean was {mean}");
        assert!(
            (variance.sqrt() - 0.05).abs() < 0.005,
            "standard deviation was {}",
            variance.sqrt()
        );
    }

    #[test]
    fn successive_layers_get_different_weights() {
        // They draw from the same global stream, so nothing is duplicated --
        // the failure this guards against is every layer of an MLP starting
        // identical.
        let a = Linear::with_in_size(3, 4);
        let b = Linear::with_in_size(3, 4);
        assert_ne!(a.weight().data(), b.weight().data());
    }

    #[test]
    fn seeding_makes_initialisation_reproducible() {
        seed(11);
        let first = Linear::with_in_size(3, 4).weight().data();
        seed(11);
        let second = Linear::with_in_size(3, 4).weight().data();
        assert_eq!(first, second);

        // ... and the numbers are the global stream's, scaled.
        seed(11);
        let expected = Rng::new(11)
            .randn(&[3, 4])
            .mapv(|v| v * (1.0_f64 / 3.0).sqrt());
        assert_eq!(Linear::with_in_size(3, 4).weight().data(), Some(expected));
    }

    // -- rejections --------------------------------------------------------

    #[test]
    #[should_panic(expected = "2-dimensional (batch, features) input")]
    fn a_lazy_layer_rejects_a_vector_input() {
        let layer = Linear::new(4);
        let _ = layer.forward(&Variable::new(arr1(&[1.0, 2.0, 3.0]).into_dyn()));
    }

    #[test]
    #[should_panic(expected = "input size of 0")]
    fn a_zero_input_size_is_rejected() {
        let _ = Linear::with_in_size(0, 4);
    }

    #[test]
    #[should_panic(expected = "the inner dimensions differ")]
    fn an_input_that_disagrees_with_an_existing_weight_is_rejected() {
        let layer = Linear::with_in_size(5, 4);
        let _ = layer.forward(&batch());
    }
}
