//! [`Model`]: the [`Layer`] at the root of a network, and the two the book
//! ships with — [`Sequential`] and [`Mlp`] (step 45).
//!
//! Port of `vendor/dezero-python/dezero/models.py`.
//!
//! # Why `Model` is (still) empty
//!
//! Python's is too:
//!
//! ```python
//! class Model(Layer):
//!     def plot(self, *inputs, to_file='model.png'):
//!         y = self.forward(*inputs)
//!         return utils.plot_dot_graph(y, verbose=True, to_file=to_file)
//! ```
//!
//! One method, and it is the step-26 Graphviz writer, which this port has not
//! reached. So `Model` carries nothing a [`Layer`] does not already carry, and
//! the honest way to say that in Rust is a supertrait alias with a blanket
//! implementation: every `Layer` *is* a `Model`, the name exists to mark the
//! role, and `plot` lands here when the DOT writer does.
//!
//! Nothing takes `&dyn Model` as a parameter — an [`Optimizer`](crate::Optimizer)
//! takes `&dyn Layer`, because that is all it needs. Modelling the relationship
//! the other way (a `Model` trait that layers do *not* satisfy) would force
//! every leaf layer to be wrapped before it could be trained on its own, which
//! Python does not require either.

pub mod mlp;
pub mod vgg;

pub use crate::models::mlp::{Mlp, Sequential};
pub use crate::models::vgg::Vgg16;

use crate::layers::Layer;

/// A [`Layer`] used as a whole network — Python's `Model`.
///
/// Implemented automatically for every `Layer`; do not implement it by hand.
///
/// # Examples
///
/// ```
/// use dezero::{Layer, Linear, Model, Variable};
/// use ndarray::arr2;
///
/// // Any layer already satisfies it, including a bare `Linear`.
/// fn parameter_count(model: &dyn Model) -> usize {
///     model.params().len()
/// }
///
/// assert_eq!(parameter_count(&Linear::with_in_size(3, 2)), 2);
/// ```
pub trait Model: Layer {}

impl<L: Layer + ?Sized> Model for L {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::variable::Variable;
    use crate::functions::reduce::sum_all;
    use crate::layers::Linear;
    use crate::optim::{Optimizer, Sgd};
    use ndarray::arr2;

    /// A model three levels deep: an `Mlp` and a bare `Linear` inside a
    /// `Sequential`. Nothing here overrides `params` or `cleargrads`; the
    /// recursion is the trait's.
    fn deep() -> Sequential {
        Sequential::new(vec![
            Box::new(Mlp::new(&[6, 4])) as Box<dyn Layer>,
            Box::new(Sequential::new(vec![
                Box::new(Linear::new(3)) as Box<dyn Layer>,
                Box::new(Linear::new(1)),
            ])),
        ])
    }

    fn batch() -> Variable {
        Variable::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn())
    }

    #[test]
    fn every_layer_is_a_model() {
        fn count(model: &dyn Model) -> usize {
            model.params().len()
        }
        assert_eq!(count(&Linear::with_in_size(3, 2)), 2);
        assert_eq!(count(&Mlp::new(&[4, 1])), 4);
        assert_eq!(count(&deep()), 8, "four layers, two parameters each");
    }

    #[test]
    fn cleargrads_reaches_every_nested_parameter() {
        let model = deep();
        sum_all(&model.forward(&batch())).backward();
        assert!(
            model.params().iter().all(|p| p.grad().is_some()),
            "the backward pass reaches every layer at every depth"
        );

        // `cleargrads` is the provided method on `Layer`, called here through
        // the `Model` view of the same object.
        let as_model: &dyn Model = &model;
        as_model.cleargrads();
        assert!(model.params().iter().all(|p| p.grad().is_none()));
    }

    #[test]
    fn a_model_can_be_trained_through_the_trait_object() {
        let model = deep();
        let as_model: &dyn Model = &model;

        let mut optimizer = Sgd::new(0.05);
        optimizer.setup(as_model);
        assert_eq!(optimizer.params().len(), 8);

        let before: Vec<_> = model.params().iter().map(|p| p.data()).collect();
        sum_all(&as_model.forward(&batch())).backward();
        optimizer.update();

        let after: Vec<_> = model.params().iter().map(|p| p.data()).collect();
        assert_ne!(before, after, "the update reached the nested parameters");
    }
}
