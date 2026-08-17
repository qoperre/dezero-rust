//! [`Optimizer`]: the rule that turns gradients into weight updates (step 46).
//!
//! Port of `Optimizer` in `vendor/dezero-python/dezero/optimizers.py`.
//!
//! Python's base class is four methods and one of them is a stub:
//!
//! ```python
//! def setup(self, target):  self.target = target; return self
//! def update(self):
//!     params = [p for p in self.target.params() if p.grad is not None]
//!     for f in self.hooks: f(params)
//!     for param in params: self.update_one(param)
//! def update_one(self, param): raise NotImplementedError()
//! ```
//!
//! The port keeps that split exactly: [`setup`](Optimizer::setup) registers a
//! network, [`update`](Optimizer::update) walks the parameters that actually
//! have a gradient, and [`update_one`](Optimizer::update_one) is the one method
//! a new optimizer writes.
//!
//! # What `setup` stores
//!
//! Python stores the *target object* and re-reads `target.params()` on every
//! update. Doing that in Rust would put a lifetime on the optimizer
//! (`Sgd<'a> { target: &'a dyn Layer }`), which then propagates into every
//! struct that owns one and every function that returns one. The port stores
//! the **parameter list** instead — a `Vec<Parameter>` of `Rc` handles, which
//! is cheap, needs no lifetime, and keeps the parameters alive for as long as
//! they are registered.
//!
//! That is equivalent for every layer in the book, and deliberately so: a
//! layer's parameter *set* is fixed at construction. A lazily-shaped
//! [`Linear`](crate::Linear) creates its `W` immediately and only fills in the
//! data later ([`Parameter::empty`](crate::Parameter::empty)), so registering
//! before the first forward pass registers the right object. What the snapshot
//! does not track is a layer that *replaces* a parameter after `setup`; no
//! layer does, and one that wanted to would call `setup` again.
//!
//! # Per-parameter state
//!
//! Optimizers with momentum key their state on `id(param)`. The port uses
//! [`Variable::id`](crate::Variable::id), which is `Rc::as_ptr` — the same
//! identity for the same reason. Reusing a freed address cannot alias one
//! parameter's state onto another's, because the optimizer holds a strong
//! handle to every parameter it has state for.

pub mod momentum;
pub mod sgd;

pub use crate::optim::momentum::MomentumSgd;
pub use crate::optim::sgd::Sgd;

use ndarray::ArrayD;

use crate::core::parameter::Parameter;
use crate::layers::Layer;

/// An update rule — Python's `Optimizer`.
///
/// A new optimizer implements three methods: two one-liners that hand the
/// trait its parameter list, and [`update_one`](Optimizer::update_one), which
/// is the rule itself.
///
/// # Examples
///
/// ```
/// use dezero::{Layer, Linear, Optimizer, Sgd, Variable, mean_squared_error};
/// use ndarray::arr2;
///
/// let model = Linear::with_in_size(1, 1);
/// let mut optimizer = Sgd::new(0.1);
/// optimizer.setup(&model);
///
/// let x = Variable::new(arr2(&[[1.0], [2.0]]).into_dyn());
/// let target = Variable::new(arr2(&[[3.0], [5.0]]).into_dyn());
///
/// let before = mean_squared_error(&model.forward(&x), &target);
/// let before = before.data().expect("loss").sum();
///
/// for _ in 0..50 {
///     let loss = mean_squared_error(&model.forward(&x), &target);
///     model.cleargrads();
///     loss.backward();
///     optimizer.update();
/// }
///
/// let after = mean_squared_error(&model.forward(&x), &target);
/// assert!(after.data().expect("loss").sum() < before);
/// ```
pub trait Optimizer {
    /// The parameters registered by [`setup`](Optimizer::setup).
    fn params(&self) -> &[Parameter];

    /// Replaces the registered parameter list.
    ///
    /// The primitive [`setup`](Optimizer::setup) is written in terms of; call
    /// it directly to optimize a bare list of parameters with no layer around
    /// them.
    fn set_params(&mut self, params: Vec<Parameter>);

    /// Applies the update rule to one parameter — Python's `update_one`.
    ///
    /// [`update`](Optimizer::update) has already checked that the parameter has
    /// a gradient.
    fn update_one(&mut self, param: &Parameter);

    /// Registers a network's parameters — Python's `Optimizer.setup`.
    ///
    /// Replaces whatever was registered before. Any per-parameter state an
    /// implementation keeps is *not* cleared: a parameter that survives a
    /// re-`setup` keeps its momentum, exactly as Python's `id`-keyed
    /// dictionaries do.
    fn setup(&mut self, target: &dyn Layer) {
        self.set_params(target.params());
    }

    /// Updates every registered parameter that has a gradient — Python's
    /// `Optimizer.update`.
    ///
    /// Parameters without one are skipped rather than treated as having a zero
    /// gradient. That distinction matters twice over: a lazily-shaped weight
    /// that has not been through a forward pass yet has no gradient and must
    /// not be touched, and for a momentum method "no gradient" must not be
    /// allowed to decay the velocity of a parameter that simply was not part of
    /// this batch.
    fn update(&mut self) {
        // The filtered list is collected first so that the borrow of `self`
        // taken by `params()` is released before `update_one` needs `&mut self`.
        let pending: Vec<Parameter> = self
            .params()
            .iter()
            .filter(|param| param.grad().is_some())
            .cloned()
            .collect();

        for param in &pending {
            self.update_one(param);
        }
    }
}

/// A parameter's data and gradient, checked to agree in shape.
///
/// Every update rule needs both and none of them can do anything sensible if
/// they disagree, so the check lives here once. Returns `None` when either is
/// missing, which [`Optimizer::update`] has already ruled out for the gradient
/// and which is the correct no-op for an uninitialised weight.
///
/// # Panics
///
/// Panics if the gradient's shape differs from the data's — a silent broadcast
/// would move the wrong weights by the wrong amounts.
fn data_and_grad(param: &Parameter) -> Option<(ArrayD<f64>, ArrayD<f64>)> {
    let data = param.data()?;
    let grad = param.grad().and_then(|g| g.data())?;
    assert!(
        data.shape() == grad.shape(),
        "dezero: a parameter of shape {:?} has a gradient of shape {:?}",
        data.shape(),
        grad.shape()
    );
    Some((data, grad))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::variable::Variable;
    use ndarray::{arr1, arr2};

    /// The smallest possible optimizer: records which parameters it was asked
    /// to update, and moves each one by a fixed amount.
    #[derive(Default)]
    struct Recorder {
        params: Vec<Parameter>,
        touched: Vec<usize>,
    }

    impl Optimizer for Recorder {
        fn params(&self) -> &[Parameter] {
            &self.params
        }

        fn set_params(&mut self, params: Vec<Parameter>) {
            self.params = params;
        }

        fn update_one(&mut self, param: &Parameter) {
            self.touched.push(param.id());
            if let Some(data) = param.data() {
                param.set_data(data.mapv(|v| v + 1.0));
            }
        }
    }

    fn with_grad(values: &[f64], grad: &[f64]) -> Parameter {
        let p = Parameter::new(arr1(values).into_dyn());
        p.set_grad(Some(Variable::new(arr1(grad).into_dyn())));
        p
    }

    #[test]
    fn setup_registers_a_layers_parameters() {
        let model = crate::Linear::with_in_size(3, 2);
        let mut optimizer = Recorder::default();
        optimizer.setup(&model);

        assert_eq!(optimizer.params().len(), 2);
        assert_eq!(optimizer.params()[0].id(), model.weight().id());
    }

    #[test]
    fn update_skips_parameters_without_a_gradient() {
        let with = with_grad(&[1.0], &[0.5]);
        let without = Parameter::new(arr1(&[1.0]).into_dyn());

        let mut optimizer = Recorder::default();
        optimizer.set_params(vec![with.clone(), without.clone()]);
        optimizer.update();

        assert_eq!(optimizer.touched, vec![with.id()]);
        assert_eq!(with.data(), Some(arr1(&[2.0]).into_dyn()));
        assert_eq!(without.data(), Some(arr1(&[1.0]).into_dyn()), "untouched");
    }

    #[test]
    fn update_visits_every_parameter_that_has_one() {
        let a = with_grad(&[1.0], &[1.0]);
        let b = with_grad(&[2.0], &[1.0]);
        let mut optimizer = Recorder::default();
        optimizer.set_params(vec![a.clone(), b.clone()]);
        optimizer.update();
        assert_eq!(optimizer.touched, vec![a.id(), b.id()]);
    }

    #[test]
    fn a_second_setup_replaces_the_registration() {
        let first = crate::Linear::with_in_size(3, 2);
        let second = crate::Linear::with_in_size(3, 2).without_bias();

        let mut optimizer = Recorder::default();
        optimizer.setup(&first);
        assert_eq!(optimizer.params().len(), 2);
        optimizer.setup(&second);
        assert_eq!(optimizer.params().len(), 1);
        assert_eq!(optimizer.params()[0].id(), second.weight().id());
    }

    #[test]
    fn data_and_grad_reports_a_missing_half() {
        assert!(data_and_grad(&Parameter::empty()).is_none());

        let no_gradient = Parameter::new(arr1(&[1.0]).into_dyn());
        assert!(data_and_grad(&no_gradient).is_none());

        let complete = with_grad(&[1.0], &[2.0]);
        let (data, grad) = data_and_grad(&complete).expect("both halves");
        assert_eq!(data, arr1(&[1.0]).into_dyn());
        assert_eq!(grad, arr1(&[2.0]).into_dyn());
    }

    #[test]
    #[should_panic(expected = "has a gradient of shape")]
    fn a_mismatched_gradient_shape_is_rejected() {
        let p = Parameter::new(arr1(&[1.0, 2.0]).into_dyn());
        p.set_grad(Some(Variable::new(arr2(&[[1.0, 2.0]]).into_dyn())));
        let _ = data_and_grad(&p);
    }
}
