//! [`Layer`]: a bundle of [`Parameter`]s with a forward pass (step 44).
//!
//! Port of `Layer` in `vendor/dezero-python/dezero/layers.py`.
//!
//! # The one thing Rust cannot copy
//!
//! Python's `Layer` registers its parameters by **intercepting attribute
//! assignment**:
//!
//! ```python
//! def __setattr__(self, name, value):
//!     if isinstance(value, (Parameter, Layer)):
//!         self._params.add(name)
//!     super().__setattr__(name, value)
//! ```
//!
//! Writing `self.W = Parameter(...)` in `Linear.__init__` therefore *also*
//! enrols `W` in the set `params()` walks, and a sub-layer assigned to a field
//! is enrolled the same way and recursed into. Rust has no `__setattr__` hook,
//! no runtime field iteration, and no `isinstance`. `docs/ARCHITECTURE.md`
//! settles the question: **an explicit trait with a hand-written `params()`**,
//! and no derive macro until the manual pattern has been proven on real layers.
//!
//! # The shape of the trait
//!
//! Registration is split in two so that nesting cannot be half-implemented:
//!
//! * [`own_params`](Layer::own_params) — the parameters this layer holds
//!   *directly*;
//! * [`sublayers`](Layer::sublayers) — the layers it holds, if any;
//! * [`params`](Layer::params) — provided, and the only one anything else
//!   calls. It concatenates the two, recursively.
//!
//! A single `params()` to override would let an author list the local
//! parameters and silently forget to recurse into a child — the failure mode
//! being a network that trains its first layer and leaves the rest frozen. With
//! the split, the recursion is written once, here, and a new layer cannot get it
//! wrong; the worst it can do is forget a field, which the crate's
//! `params()` count tests catch.
//!
//! `forward` takes `&self`, not `&mut self`. A [`Variable`] is an `Rc` over
//! interior-mutable cells, so a layer can still fill in a lazily-shaped weight
//! during its first forward pass — which is exactly what
//! [`Linear`] does — without forcing a mutable borrow up
//! through every caller and out into the training loop.
//!
//! # Example
//!
//! A two-layer network, with the registration written out. This is the whole of
//! what Python's `__setattr__` buys, and it is four lines:
//!
//! ```
//! use dezero::{sigmoid, Layer, Linear, Parameter, Variable};
//! use ndarray::arr2;
//!
//! struct TwoLayer {
//!     first: Linear,
//!     second: Linear,
//! }
//!
//! impl Layer for TwoLayer {
//!     fn own_params(&self) -> Vec<Parameter> {
//!         Vec::new() // no parameters of its own -- they all live in the children
//!     }
//!
//!     fn sublayers(&self) -> Vec<&dyn Layer> {
//!         vec![&self.first, &self.second]
//!     }
//!
//!     fn forward(&self, x: &Variable) -> Variable {
//!         self.second.forward(&sigmoid(&self.first.forward(x)))
//!     }
//! }
//!
//! let net = TwoLayer {
//!     first: Linear::with_in_size(3, 4),
//!     second: Linear::with_in_size(4, 1),
//! };
//!
//! // `params()` flattens both children: two weights and two biases.
//! assert_eq!(net.params().len(), 4);
//!
//! let x = Variable::new(arr2(&[[1.0, 2.0, 3.0]]).into_dyn());
//! let y = net.forward(&x);
//! assert_eq!(y.shape(), Some(vec![1, 1]));
//!
//! // ... and `cleargrads` reaches every one of them.
//! y.backward();
//! assert!(net.params().iter().all(|p| p.grad().is_some()));
//! net.cleargrads();
//! assert!(net.params().iter().all(|p| p.grad().is_none()));
//! ```

pub mod conv;
pub mod linear;
pub mod rnn;
pub mod save;

pub use crate::layers::conv::Conv2dLayer;
pub use crate::layers::linear::Linear;
pub use crate::layers::rnn::{Lstm, Rnn};
pub use crate::layers::save::{WeightsError, load_weights, save_weights};

use crate::core::parameter::Parameter;
use crate::core::variable::Variable;

/// A parameterised, differentiable transformation — Python's `Layer`.
///
/// Implementors supply three things: the parameters they own, the layers they
/// contain, and a forward pass. Everything else is provided.
///
/// The trait is object-safe, so a network can hold `Box<dyn Layer>` children of
/// mixed types — see [`Sequential`](crate::models::Sequential).
pub trait Layer {
    /// The parameters declared directly on this layer, in a stable order.
    ///
    /// This is the hand-written stand-in for Python's `__setattr__`
    /// registration: list every [`Parameter`] field, once. Return an empty
    /// vector for a layer that only composes other layers.
    ///
    /// The returned parameters share identity with the layer's own (a
    /// [`Parameter`] is an `Rc`), so writing through one of them updates the
    /// layer.
    fn own_params(&self) -> Vec<Parameter>;

    /// The layers this layer contains, in a stable order.
    ///
    /// Python discovers these by the same `__setattr__` hook that finds
    /// parameters and recurses into them from `params()`. Override this for any
    /// layer with children; the default — none — is right for a leaf.
    fn sublayers(&self) -> Vec<&dyn Layer> {
        Vec::new()
    }

    /// Runs the layer — Python's `Layer.__call__`.
    ///
    /// Python's is variadic; every layer in the book takes a single input and
    /// produces a single output, so the port says so in the signature.
    fn forward(&self, x: &Variable) -> Variable;

    /// Every parameter in this layer and, recursively, in its children —
    /// Python's `Layer.params()`.
    ///
    /// The order is deterministic: own parameters first, then each sub-layer's
    /// in turn. (Python's is *not*: `_params` is a `set`, so its iteration
    /// order is hash-dependent. Nothing in DeZero depends on it, and a stable
    /// order is strictly easier to test against.)
    fn params(&self) -> Vec<Parameter> {
        let mut all = self.own_params();
        for sublayer in self.sublayers() {
            all.extend(sublayer.params());
        }
        all
    }

    /// Clears the gradient of every parameter, recursively — Python's
    /// `Layer.cleargrads()`.
    ///
    /// Call this at the top of each training iteration:
    /// [`backward`](Variable::backward) *accumulates* into `grad`, so a run
    /// that skips it sums the gradients of every iteration so far.
    fn cleargrads(&self) {
        for param in self.params() {
            param.cleargrad();
        }
    }

    /// Every parameter paired with a path that identifies it — Python's
    /// `Layer._flatten_params`.
    ///
    /// Keys look like `W`, `b`, `0/W`, `1/2/b`: a sub-layer contributes its
    /// **index** as a path segment, and the leaf segment is the parameter's own
    /// name.
    ///
    /// Python uses the *field* name for the sub-layer segment (`l1/W`), which
    /// it gets from `__setattr__` interception that Rust has no equivalent of.
    /// The index is stable for a given layer structure, which is all
    /// save/load needs. It does mean a file written here does not carry
    /// Python's key names — but the formats differ anyway (divergence 31), so
    /// the two were never going to exchange files.
    ///
    /// An unnamed parameter falls back to its position, so keys stay unique
    /// even for a layer that never named its weights.
    fn named_params(&self) -> Vec<(String, Parameter)> {
        let mut out = Vec::new();
        for (index, param) in self.own_params().into_iter().enumerate() {
            let leaf = param.name().unwrap_or_else(|| index.to_string());
            out.push((leaf, param));
        }
        for (index, sublayer) in self.sublayers().into_iter().enumerate() {
            for (key, param) in sublayer.named_params() {
                out.push((format!("{index}/{key}"), param));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functions::activation::sigmoid;
    use ndarray::{ArrayD, arr1, arr2};

    /// A leaf layer with two parameters and a trivially checkable forward.
    struct Scale {
        gain: Parameter,
        offset: Parameter,
    }

    impl Scale {
        fn new(gain: f64, offset: f64) -> Self {
            Self {
                gain: Parameter::named(Some(ndarray::arr0(gain).into_dyn()), "gain"),
                offset: Parameter::named(Some(ndarray::arr0(offset).into_dyn()), "offset"),
            }
        }
    }

    impl Layer for Scale {
        fn own_params(&self) -> Vec<Parameter> {
            vec![self.gain.clone(), self.offset.clone()]
        }

        fn forward(&self, x: &Variable) -> Variable {
            &(x * &*self.gain) + &*self.offset
        }
    }

    /// One level of nesting: two `Scale`s and a parameter of its own.
    struct Middle {
        weight: Parameter,
        left: Scale,
        right: Scale,
    }

    impl Layer for Middle {
        fn own_params(&self) -> Vec<Parameter> {
            vec![self.weight.clone()]
        }

        fn sublayers(&self) -> Vec<&dyn Layer> {
            vec![&self.left, &self.right]
        }

        fn forward(&self, x: &Variable) -> Variable {
            &self.left.forward(x) * &(&self.right.forward(x) * &*self.weight)
        }
    }

    /// Two levels of nesting, plus a leaf sibling.
    struct Outer {
        middle: Middle,
        tail: Scale,
    }

    impl Layer for Outer {
        fn own_params(&self) -> Vec<Parameter> {
            Vec::new()
        }

        fn sublayers(&self) -> Vec<&dyn Layer> {
            vec![&self.middle, &self.tail]
        }

        fn forward(&self, x: &Variable) -> Variable {
            self.tail.forward(&sigmoid(&self.middle.forward(x)))
        }
    }

    fn nested() -> Outer {
        Outer {
            middle: Middle {
                weight: Parameter::named(Some(ndarray::arr0(3.0).into_dyn()), "weight"),
                left: Scale::new(2.0, 1.0),
                right: Scale::new(0.5, -1.0),
            },
            tail: Scale::new(1.5, 0.25),
        }
    }

    #[test]
    fn params_flattens_the_whole_tree() {
        let net = nested();
        // 1 (Middle.weight) + 2 + 2 (its Scales) + 2 (the tail Scale).
        assert_eq!(net.params().len(), 7);

        let names: Vec<String> = net
            .params()
            .iter()
            .map(|p| p.name().unwrap_or_default())
            .collect();
        assert_eq!(
            names,
            vec![
                "weight", "gain", "offset", "gain", "offset", "gain", "offset"
            ],
            "own parameters first, then each sub-layer in order"
        );
    }

    #[test]
    fn params_returns_the_layers_own_parameters_not_copies() {
        let net = nested();
        let before = net.middle.weight.data();

        // Writing through the flattened handle must be visible on the layer.
        let handle = net
            .params()
            .into_iter()
            .find(|p| p.id() == net.middle.weight.id())
            .expect("the weight is in the flattened list");
        handle.set_data(ndarray::arr0(99.0).into_dyn());

        assert_ne!(net.middle.weight.data(), before);
        assert_eq!(
            net.middle.weight.data(),
            Some(ndarray::arr0(99.0).into_dyn())
        );
    }

    #[test]
    fn every_flattened_parameter_is_distinct() {
        let net = nested();
        let mut ids: Vec<usize> = net.params().iter().map(|p| p.id()).collect();
        ids.sort_unstable();
        let count = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), count, "no parameter is listed twice");
    }

    #[test]
    fn a_leaf_layer_has_no_sublayers_by_default() {
        let leaf = Scale::new(1.0, 0.0);
        assert!(leaf.sublayers().is_empty());
        assert_eq!(leaf.params().len(), 2);
    }

    #[test]
    fn cleargrads_reaches_every_nested_parameter() {
        let net = nested();
        let x = Variable::new(arr1(&[0.5, 1.0]).into_dyn());

        crate::sum_all(&net.forward(&x)).backward();
        assert!(
            net.params().iter().all(|p| p.grad().is_some()),
            "every parameter took part in the forward pass"
        );

        net.cleargrads();
        assert!(net.params().iter().all(|p| p.grad().is_none()));
    }

    /// Without `cleargrads` the gradients of successive passes sum, which is
    /// the bug the call exists to prevent.
    #[test]
    fn gradients_accumulate_across_passes_until_cleared() {
        let net = nested();
        let x = Variable::new(arr1(&[0.5, 1.0]).into_dyn());

        let gradient_of = |net: &Outer| -> ArrayD<f64> {
            net.middle
                .weight
                .grad()
                .and_then(|g| g.data())
                .expect("gradient")
        };

        crate::sum_all(&net.forward(&x)).backward();
        let once = gradient_of(&net);

        crate::sum_all(&net.forward(&x)).backward();
        let twice = gradient_of(&net);
        assert_eq!(twice, once.mapv(|v| v * 2.0));

        net.cleargrads();
        crate::sum_all(&net.forward(&x)).backward();
        assert_eq!(gradient_of(&net), once);
    }

    #[test]
    fn a_layer_works_behind_a_trait_object() {
        let net = nested();
        let erased: &dyn Layer = &net;
        assert_eq!(erased.params().len(), 7);

        let x = Variable::new(arr2(&[[0.5, 1.0]]).into_dyn());
        assert_eq!(erased.forward(&x).shape(), Some(vec![1, 2]));
    }
}
