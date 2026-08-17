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
//!
//! # Hooks (step 50)
//!
//! Python's `update` runs a list of callables over the parameter list before
//! stepping any of it, and [`hooks`](mod@crate::optim::hooks) ports the three
//! the book ships. Storage for them is a *required* trait method rather than a
//! default, for the same reason [`Layer`]'s registration is split in two: a new
//! optimizer that quietly dropped its hooks would train a network that ignores
//! its own weight decay, with nothing anywhere to notice.

pub mod hooks;
pub mod momentum;
pub mod sgd;

pub use crate::optim::hooks::{ClipGrad, FreezeParam, Hook, Hooks, WeightDecay};
pub use crate::optim::momentum::MomentumSgd;
pub use crate::optim::sgd::Sgd;

use std::rc::Rc;

use ndarray::ArrayD;

use crate::core::parameter::Parameter;
use crate::layers::Layer;

/// An update rule — Python's `Optimizer`.
///
/// A new optimizer implements five methods: four one-liners that hand the
/// trait its parameter list and its [`Hook`] list, and
/// [`update_one`](Optimizer::update_one), which is the rule itself. Everything
/// else — [`setup`](Optimizer::setup), [`update`](Optimizer::update),
/// [`add_hook`](Optimizer::add_hook) — is provided.
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

    /// The hooks registered by [`add_hook`](Optimizer::add_hook), in order.
    fn hooks(&self) -> &[Rc<dyn Hook>];

    /// The hook list, for mutation.
    ///
    /// Required rather than defaulted so that an optimizer must decide where
    /// its hooks live. The alternative — a default that silently accepts and
    /// discards them — is the failure mode this trait's shape exists to
    /// prevent.
    fn hooks_mut(&mut self) -> &mut Hooks;

    /// Applies the update rule to one parameter — Python's `update_one`.
    ///
    /// [`update`](Optimizer::update) has already checked that the parameter had
    /// a gradient *before the hooks ran*. A hook may since have cleared it, so
    /// an implementation still has to cope with a gradient that is not there —
    /// which is what `data_and_grad` returning `None` is for.
    fn update_one(&mut self, param: &Parameter);

    /// Registers a gradient rewrite — Python's `Optimizer.add_hook`.
    ///
    /// Hooks run in the order they were added, before every update.
    ///
    /// # Examples
    ///
    /// ```
    /// use dezero::{Optimizer, Parameter, Sgd, Variable, WeightDecay};
    /// use ndarray::arr1;
    ///
    /// let w = Parameter::new(arr1(&[4.0]).into_dyn());
    /// w.set_grad(Some(Variable::new(arr1(&[1.0]).into_dyn())));
    ///
    /// let mut optimizer = Sgd::new(0.1);
    /// optimizer.set_params(vec![w.clone()]);
    /// optimizer.add_hook(WeightDecay::new(0.25));
    /// optimizer.update();
    ///
    /// // The gradient became 1 + 0.25 * 4 = 2 before the step.
    /// assert_eq!(w.data(), Some(arr1(&[3.8]).into_dyn()));
    /// ```
    fn add_hook<H: Hook + 'static>(&mut self, hook: H)
    where
        Self: Sized,
    {
        self.hooks_mut().push(Rc::new(hook));
    }

    /// Registers a hook that is already shared — the object-safe half of
    /// [`add_hook`](Optimizer::add_hook).
    ///
    /// Use it to give the same [`FreezeParam`] to a `dyn Optimizer`, or to two
    /// optimizers at once.
    fn add_shared_hook(&mut self, hook: Rc<dyn Hook>) {
        self.hooks_mut().push(hook);
    }

    /// Discards every registered hook.
    fn clear_hooks(&mut self) {
        self.hooks_mut().clear();
    }

    /// Registers a network's parameters — Python's `Optimizer.setup`.
    ///
    /// Replaces whatever was registered before. Any per-parameter state an
    /// implementation keeps is *not* cleared: a parameter that survives a
    /// re-`setup` keeps its momentum, exactly as Python's `id`-keyed
    /// dictionaries do.
    fn setup(&mut self, target: &dyn Layer) {
        self.set_params(target.params());
    }

    /// Runs every hook, then updates every registered parameter that has a
    /// gradient — Python's `Optimizer.update`.
    ///
    /// Parameters without one are skipped rather than treated as having a zero
    /// gradient. That distinction matters twice over: a lazily-shaped weight
    /// that has not been through a forward pass yet has no gradient and must
    /// not be touched, and for a momentum method "no gradient" must not be
    /// allowed to decay the velocity of a parameter that simply was not part of
    /// this batch.
    ///
    /// The three-step order is Python's exactly — filter, hook, step — and each
    /// step depends on the one before it:
    ///
    /// ```python
    /// params = [p for p in self.target.params() if p.grad is not None]
    /// for f in self.hooks: f(params)
    /// for param in params: self.update_one(param)
    /// ```
    ///
    /// In particular the list is snapshotted *before* the hooks run, so a
    /// [`FreezeParam`] that clears a gradient does not remove its parameter
    /// from this pass; `update_one` reaches it and finds nothing to do.
    fn update(&mut self) {
        // The filtered list is collected first so that the borrow of `self`
        // taken by `params()` is released before `update_one` needs `&mut self`.
        let pending: Vec<Parameter> = self
            .params()
            .iter()
            .filter(|param| param.grad().is_some())
            .cloned()
            .collect();

        for hook in self.hooks() {
            hook.call(&pending);
        }

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
    use std::cell::RefCell;

    /// The smallest possible optimizer: records which parameters it was asked
    /// to update, and moves each one by a fixed amount.
    #[derive(Debug, Default)]
    struct Recorder {
        params: Vec<Parameter>,
        hooks: Hooks,
        touched: Vec<usize>,
    }

    impl Optimizer for Recorder {
        fn params(&self) -> &[Parameter] {
            &self.params
        }

        fn set_params(&mut self, params: Vec<Parameter>) {
            self.params = params;
        }

        fn hooks(&self) -> &[Rc<dyn Hook>] {
            &self.hooks
        }

        fn hooks_mut(&mut self) -> &mut Hooks {
            &mut self.hooks
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

    // -- hooks -------------------------------------------------------------

    /// Records the parameter list it was handed, and when.
    #[derive(Debug, Default)]
    struct Spy {
        seen: RefCell<Vec<Vec<usize>>>,
    }

    impl Hook for Spy {
        fn call(&self, params: &[Parameter]) {
            // `id` reaches `Parameter` through `Deref`, so it needs a closure
            // rather than a path.
            self.seen
                .borrow_mut()
                .push(params.iter().map(|param| param.id()).collect());
        }
    }

    #[test]
    fn an_optimizer_starts_with_no_hooks() {
        let optimizer = Recorder::default();
        assert!(optimizer.hooks().is_empty());
    }

    #[test]
    fn add_hook_registers_in_order_and_clear_hook_removes_them() {
        let mut optimizer = Recorder::default();
        optimizer.add_hook(WeightDecay::new(0.1));
        optimizer.add_hook(ClipGrad::new(1.0));
        assert_eq!(optimizer.hooks().len(), 2);
        assert!(format!("{:?}", optimizer.hooks()[0]).contains("WeightDecay"));
        assert!(format!("{:?}", optimizer.hooks()[1]).contains("ClipGrad"));

        optimizer.clear_hooks();
        assert!(optimizer.hooks().is_empty());
    }

    /// Python filters, then hooks, then steps. A hook must therefore see the
    /// gradient-bearing parameters and only those, and see them before any of
    /// them has moved.
    #[test]
    fn hooks_see_the_filtered_list_before_any_update() {
        let with = with_grad(&[1.0], &[0.5]);
        let without = Parameter::new(arr1(&[1.0]).into_dyn());

        let spy = Rc::new(Spy::default());
        let mut optimizer = Recorder::default();
        optimizer.set_params(vec![with.clone(), without.clone()]);
        optimizer.add_shared_hook(Rc::clone(&spy) as Rc<dyn Hook>);
        optimizer.update();

        assert_eq!(
            *spy.seen.borrow(),
            vec![vec![with.id()]],
            "called once, with the parameters that have a gradient"
        );
        assert!(
            optimizer.touched.iter().all(|id| *id == with.id()),
            "and the step happened after"
        );
    }

    #[test]
    fn a_hook_runs_once_per_update_not_once_per_parameter() {
        let a = with_grad(&[1.0], &[1.0]);
        let b = with_grad(&[2.0], &[1.0]);

        let spy = Rc::new(Spy::default());
        let mut optimizer = Recorder::default();
        optimizer.set_params(vec![a, b]);
        optimizer.add_shared_hook(Rc::clone(&spy) as Rc<dyn Hook>);

        optimizer.update();
        optimizer.update();

        let seen = spy.seen.borrow();
        assert_eq!(seen.len(), 2, "two updates, two calls");
        assert_eq!(seen[0].len(), 2, "both parameters, in one call");
    }

    /// The snapshot is taken before the hooks, so a hook that clears a gradient
    /// does not shorten the list being iterated — `update_one` still gets the
    /// parameter and has to cope with the gradient being gone.
    #[test]
    fn a_hook_that_clears_a_gradient_does_not_shrink_the_pass() {
        let w = with_grad(&[1.0], &[1.0]);
        let mut optimizer = Recorder::default();
        optimizer.set_params(vec![w.clone()]);
        optimizer.add_hook(FreezeParam::new(vec![w.clone()]));
        optimizer.update();

        assert_eq!(
            optimizer.touched,
            vec![w.id()],
            "update_one was still called for it"
        );
        assert!(w.grad().is_none());
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
