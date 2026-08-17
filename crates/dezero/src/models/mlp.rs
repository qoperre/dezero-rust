//! [`Sequential`] and [`Mlp`], the two networks the book builds by composition
//! (step 45).
//!
//! Port of `Sequential` and `MLP` in
//! `vendor/dezero-python/dezero/models.py`.
//!
//! Both are the same idea from two directions. `Sequential` is heterogeneous —
//! any list of [`Layer`]s, applied in order, no activations of its own.
//! `Mlp` is homogeneous — a list of [`Linear`]s with an activation between
//! consecutive pairs and none after the last, which is exactly what a
//! classifier or a regressor wants.
//!
//! # Registration
//!
//! Python's constructors do the registration by hand too, and for the same
//! reason the Rust ones do:
//!
//! ```python
//! for i, layer in enumerate(layers):
//!     setattr(self, 'l' + str(i), layer)   # only this line registers it
//!     self.layers.append(layer)
//! ```
//!
//! The list alone is invisible to `__setattr__`, so a synthetic attribute name
//! is invented per layer purely to trip the hook. The port needs no such trick:
//! [`sublayers`](Layer::sublayers) returns the list.

use crate::core::parameter::Parameter;
use crate::core::variable::Variable;
use crate::functions::activation::sigmoid;
use crate::layers::{Layer, Linear};

/// A pipeline of layers, applied front to back — Python's `Sequential`.
///
/// The layers are boxed, so they need not share a type.
///
/// # Examples
///
/// ```
/// use dezero::{Layer, Linear, Sequential, Variable};
/// use ndarray::arr2;
///
/// let net = Sequential::new(vec![
///     Box::new(Linear::new(4)) as Box<dyn Layer>,
///     Box::new(Linear::new(1)),
/// ]);
///
/// let x = Variable::new(arr2(&[[1.0, 2.0, 3.0]]).into_dyn());
/// assert_eq!(net.forward(&x).shape(), Some(vec![1, 1]));
/// assert_eq!(net.params().len(), 4, "two weights and two biases");
/// ```
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    /// Builds a pipeline from a list of layers.
    ///
    /// An empty list is allowed and makes the forward pass the identity, which
    /// is what Python's `Sequential()` does.
    #[must_use]
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Self { layers }
    }

    /// Appends a layer to the end of the pipeline.
    pub fn push(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    /// The layers, in order.
    #[must_use]
    pub fn layers(&self) -> &[Box<dyn Layer>] {
        &self.layers
    }
}

impl Layer for Sequential {
    fn own_params(&self) -> Vec<Parameter> {
        Vec::new()
    }

    fn sublayers(&self) -> Vec<&dyn Layer> {
        self.layers.iter().map(|layer| &**layer).collect()
    }

    fn forward(&self, x: &Variable) -> Variable {
        let mut value = x.clone();
        for layer in &self.layers {
            value = layer.forward(&value);
        }
        value
    }
}

/// A multi-layer perceptron — Python's `MLP`.
///
/// One [`Linear`] per entry of `fc_output_sizes`, with `activation` applied
/// between consecutive layers and **not** after the last: the output of a
/// network is a logit or a regression value, and squashing it would be wrong
/// for both.
///
/// Every layer is lazily shaped, so `Mlp::new(&[10, 3])` is complete before
/// anything is known about the input width — the first forward pass settles it.
///
/// # Examples
///
/// ```
/// use dezero::{relu, Layer, Mlp, Variable};
/// use ndarray::arr2;
///
/// // The book's shorthand: MLP((10, 1)).
/// let net = Mlp::new(&[10, 1]);
/// let x = Variable::new(arr2(&[[0.5], [1.5], [2.5]]).into_dyn());
///
/// let y = net.forward(&x);
/// assert_eq!(y.shape(), Some(vec![3, 1]));
/// assert_eq!(net.params().len(), 4);
/// assert_eq!(net.layers()[0].in_size(), Some(1), "sized by the first batch");
///
/// // A different activation is a constructor argument, as in Python.
/// let rectified = Mlp::with_activation(&[10, 1], relu);
/// assert_eq!(rectified.forward(&x).shape(), Some(vec![3, 1]));
/// ```
pub struct Mlp {
    layers: Vec<Linear>,
    activation: fn(&Variable) -> Variable,
}

impl Mlp {
    /// An MLP with [`sigmoid`] between its layers — Python's default.
    ///
    /// # Panics
    ///
    /// Panics if `fc_output_sizes` is empty: a network with no layers has no
    /// output, and Python fails at `self.layers[-1]` for the same reason.
    #[must_use]
    pub fn new(fc_output_sizes: &[usize]) -> Self {
        Self::with_activation(fc_output_sizes, sigmoid)
    }

    /// An MLP with an explicit activation — Python's
    /// `MLP(fc_output_sizes, activation=...)`.
    ///
    /// `activation` is a plain function pointer rather than a boxed closure:
    /// every activation in the book is a free function, and the pointer keeps
    /// the model `Copy`-cheap to construct and free of a lifetime parameter.
    ///
    /// # Panics
    ///
    /// Panics if `fc_output_sizes` is empty.
    #[must_use]
    pub fn with_activation(
        fc_output_sizes: &[usize],
        activation: fn(&Variable) -> Variable,
    ) -> Self {
        assert!(
            !fc_output_sizes.is_empty(),
            "dezero: an MLP needs at least one layer"
        );
        Self {
            layers: fc_output_sizes
                .iter()
                .map(|&out_size| Linear::new(out_size))
                .collect(),
            activation,
        }
    }

    /// The fully connected layers, in order.
    #[must_use]
    pub fn layers(&self) -> &[Linear] {
        &self.layers
    }
}

impl Layer for Mlp {
    fn own_params(&self) -> Vec<Parameter> {
        Vec::new()
    }

    fn sublayers(&self) -> Vec<&dyn Layer> {
        self.layers
            .iter()
            .map(|layer| layer as &dyn Layer)
            .collect()
    }

    fn forward(&self, x: &Variable) -> Variable {
        let Some((last, hidden)) = self.layers.split_last() else {
            panic!("dezero: internal invariant broken — an MLP was built with no layers");
        };

        let mut value = x.clone();
        for layer in hidden {
            value = (self.activation)(&layer.forward(&value));
        }
        last.forward(&value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functions::activation::relu;
    use crate::functions::reduce::sum_all;
    use ndarray::arr2;

    fn batch() -> Variable {
        Variable::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn())
    }

    // -- Sequential --------------------------------------------------------

    #[test]
    fn sequential_applies_its_layers_in_order() {
        let net = Sequential::new(vec![
            Box::new(Linear::with_in_size(3, 5)) as Box<dyn Layer>,
            Box::new(Linear::with_in_size(5, 2)),
        ]);
        assert_eq!(net.forward(&batch()).shape(), Some(vec![2, 2]));
        assert_eq!(net.layers().len(), 2);
    }

    #[test]
    fn sequential_flattens_every_layers_parameters() {
        let net = Sequential::new(vec![
            Box::new(Linear::with_in_size(3, 5)) as Box<dyn Layer>,
            Box::new(Linear::with_in_size(5, 2).without_bias()),
        ]);
        assert_eq!(net.params().len(), 3, "W, b, W");
        assert!(net.own_params().is_empty());
        assert_eq!(net.sublayers().len(), 2);
    }

    #[test]
    fn an_empty_sequential_is_the_identity() {
        let net = Sequential::new(Vec::new());
        let x = batch();
        let y = net.forward(&x);
        assert_eq!(y.id(), x.id(), "no layer means no new node");
        assert!(net.params().is_empty());
    }

    #[test]
    fn sequential_can_be_extended_after_construction() {
        let mut net = Sequential::new(Vec::new());
        net.push(Box::new(Linear::with_in_size(3, 4)));
        net.push(Box::new(Linear::with_in_size(4, 1)));
        assert_eq!(net.params().len(), 4);
        assert_eq!(net.forward(&batch()).shape(), Some(vec![2, 1]));
    }

    #[test]
    fn sequential_nests_inside_itself() {
        let inner = Sequential::new(vec![Box::new(Linear::with_in_size(3, 4)) as Box<dyn Layer>]);
        let outer = Sequential::new(vec![
            Box::new(inner) as Box<dyn Layer>,
            Box::new(Linear::with_in_size(4, 2)),
        ]);
        assert_eq!(
            outer.params().len(),
            4,
            "the recursion goes all the way down"
        );
        assert_eq!(outer.forward(&batch()).shape(), Some(vec![2, 2]));
    }

    // -- MLP ---------------------------------------------------------------

    #[test]
    fn mlp_sizes_itself_from_the_first_batch() {
        let net = Mlp::new(&[10, 4, 1]);
        assert!(net.layers().iter().all(|l| l.in_size().is_none()));

        let y = net.forward(&batch());
        assert_eq!(y.shape(), Some(vec![2, 1]));
        let sizes: Vec<Option<usize>> = net.layers().iter().map(Linear::in_size).collect();
        assert_eq!(sizes, vec![Some(3), Some(10), Some(4)]);
    }

    #[test]
    fn mlp_has_two_parameters_per_layer() {
        let net = Mlp::new(&[10, 4, 1]);
        assert_eq!(net.params().len(), 6);
        assert!(net.own_params().is_empty());
        assert_eq!(net.sublayers().len(), 3);
    }

    /// The output must be raw: an activation after the last layer would clamp
    /// a regressor into `(0, 1)` and destroy a classifier's logits.
    #[test]
    fn mlp_does_not_activate_its_output() {
        let net = Mlp::new(&[4, 1]);
        net.forward(&batch()); // sizes both layers

        // Pin the weights so the numbers below owe nothing to the draw: the
        // hidden layer becomes sigmoid(0) = 0.5 everywhere, and the output
        // layer sums four of those at a gain of 50.
        net.layers()[0]
            .weight()
            .set_data(ndarray::ArrayD::zeros(ndarray::IxDyn(&[3, 4])));
        net.layers()[1]
            .weight()
            .set_data(ndarray::ArrayD::from_elem(ndarray::IxDyn(&[4, 1]), 50.0));

        let y = net.forward(&batch()).data().expect("data");
        assert!(
            (y[[0, 0]] - 100.0).abs() < 1e-12,
            "expected the raw affine output, got {y:?}"
        );
        assert!(
            y.iter().all(|v| *v > 1.0),
            "a sigmoid-capped output could never leave (0, 1)"
        );
    }

    #[test]
    fn a_single_layer_mlp_is_just_that_layer() {
        let net = Mlp::new(&[2]);
        assert_eq!(net.params().len(), 2);
        let y = net.forward(&batch());
        assert_eq!(y.shape(), Some(vec![2, 2]));

        let expected = crate::linear(
            &batch(),
            net.layers()[0].weight(),
            net.layers()[0].bias().map(|b| &**b),
        );
        assert_eq!(y.data(), expected.data());
    }

    #[test]
    fn the_activation_is_configurable() {
        let sigmoidal = Mlp::new(&[6, 3]);
        let rectified = Mlp::with_activation(&[6, 3], relu);

        // Give both the same weights, so only the activation can differ.
        sigmoidal.forward(&batch());
        rectified.forward(&batch());
        for (a, b) in sigmoidal.layers().iter().zip(rectified.layers()) {
            let w = a.weight().data().expect("initialised");
            b.weight().set_data(w);
            if let (Some(ab), Some(bb)) = (a.bias(), b.bias()) {
                bb.set_data(ab.data().expect("initialised"));
            }
        }

        assert_ne!(
            sigmoidal.forward(&batch()).data(),
            rectified.forward(&batch()).data()
        );
    }

    #[test]
    fn backward_reaches_every_layer_of_an_mlp() {
        let net = Mlp::new(&[10, 4, 1]);
        sum_all(&net.forward(&batch())).backward();
        assert!(
            net.params().iter().all(|p| p.grad().is_some()),
            "no layer is left out of the backward pass"
        );

        net.cleargrads();
        assert!(net.params().iter().all(|p| p.grad().is_none()));
    }

    #[test]
    #[should_panic(expected = "at least one layer")]
    fn an_mlp_needs_at_least_one_layer() {
        let _ = Mlp::new(&[]);
    }
}
