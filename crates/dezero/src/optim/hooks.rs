//! [`Hook`]s: the gradient rewrites an [`Optimizer`](crate::Optimizer) applies
//! before it steps (step 50).
//!
//! Port of `WeightDecay`, `ClipGrad` and `FreezeParam` in
//! `vendor/dezero-python/dezero/optimizers.py`.
//!
//! Python's hooks are bare callables invoked from `Optimizer.update`:
//!
//! ```python
//! params = [p for p in self.target.params() if p.grad is not None]
//! for f in self.hooks:
//!     f(params)
//! for param in params:
//!     self.update_one(param)
//! ```
//!
//! Three things about that order are load-bearing and the port keeps all three:
//!
//! 1. hooks see the **whole** parameter list at once, not one parameter at a
//!    time — [`ClipGrad`] could not compute a global norm otherwise;
//! 2. hooks run **before** any update, so a rule reads gradients no step has
//!    yet consumed;
//! 3. the list is snapshotted **before** the hooks run, so a hook that clears a
//!    gradient does not shorten the list being iterated.
//!
//! # Why `Hook` requires `Debug`
//!
//! [`Sgd`](crate::Sgd) and [`MomentumSgd`](crate::MomentumSgd) derive `Debug`,
//! and an optimizer that printed everything about itself *except* the rules
//! rewriting its gradients would be actively misleading in a debug session. A
//! `#[derive(Debug)]` on the hook is the whole cost, and it is why hooks are
//! named types here rather than Python's anonymous callables.

use std::fmt;
use std::rc::Rc;

use crate::core::parameter::Parameter;
use crate::core::variable::Variable;
use crate::layers::Layer;
use crate::optim::data_and_grad;

/// The hooks an optimizer holds, in the order they were added.
///
/// `Rc` rather than `Box` so that [`Sgd`](crate::Sgd) and friends stay
/// `Clone`: a hook is a stateless rule, so sharing one between a cloned pair of
/// optimizers is exactly right.
pub type Hooks = Vec<Rc<dyn Hook>>;

/// A rule applied to every gradient before an update — Python's hook
/// callables.
///
/// Takes the parameters the optimizer is about to step, all of them at once,
/// and rewrites their gradients in place.
///
/// # Examples
///
/// A hook is a struct with one method:
///
/// ```
/// use dezero::{Hook, Optimizer, Parameter, Sgd, Variable};
/// use ndarray::arr1;
///
/// /// Throws away the sign of every gradient, keeping only its direction.
/// #[derive(Debug)]
/// struct SignOnly;
///
/// impl Hook for SignOnly {
///     fn call(&self, params: &[Parameter]) {
///         for param in params {
///             if let Some(grad) = param.grad() {
///                 if let Some(values) = grad.data() {
///                     grad.set_data(values.mapv(f64::signum));
///                 }
///             }
///         }
///     }
/// }
///
/// let w = Parameter::new(arr1(&[0.0, 0.0]).into_dyn());
/// w.set_grad(Some(Variable::new(arr1(&[100.0, -0.001]).into_dyn())));
///
/// let mut optimizer = Sgd::new(0.5);
/// optimizer.set_params(vec![w.clone()]);
/// optimizer.add_hook(SignOnly);
/// optimizer.update();
///
/// assert_eq!(w.data(), Some(arr1(&[-0.5, 0.5]).into_dyn()));
/// ```
pub trait Hook: fmt::Debug {
    /// Rewrites the gradients of `params` — Python's `__call__`.
    ///
    /// Every parameter in the slice has a gradient at the moment the optimizer
    /// calls this; a hook that clears one must expect to see it again in the
    /// same slice if another hook follows.
    fn call(&self, params: &[Parameter]);
}

/// A parameter's gradient node together with its values, or `None` if either
/// half is missing.
fn grad_parts(param: &Parameter) -> Option<(Variable, ndarray::ArrayD<f64>)> {
    let node = param.grad()?;
    let values = node.data()?;
    Some((node, values))
}

/// L2 regularisation folded into the gradient — Python's `WeightDecay`.
///
/// ```python
/// param.grad.data += self.rate * param.data
/// ```
///
/// Adding `rate * w` to `dL/dw` is the gradient of `rate/2 * ||w||^2`, so the
/// step becomes one on `L + rate/2 * ||w||^2` without the loss ever being told.
/// That is the trick, and also the catch: the reported loss no longer includes
/// the penalty.
///
/// # Examples
///
/// ```
/// use dezero::{Hook, Parameter, Variable, WeightDecay};
/// use ndarray::arr1;
///
/// let w = Parameter::new(arr1(&[2.0, -4.0]).into_dyn());
/// w.set_grad(Some(Variable::new(arr1(&[1.0, 1.0]).into_dyn())));
///
/// WeightDecay::new(0.5).call(&[w.clone()]);
///
/// // grad += 0.5 * data
/// assert_eq!(
///     w.grad().and_then(|g| g.data()),
///     Some(arr1(&[2.0, -1.0]).into_dyn())
/// );
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightDecay {
    rate: f64,
}

impl WeightDecay {
    /// The decay coefficient — Python's `rate`.
    #[must_use]
    pub fn new(rate: f64) -> Self {
        Self { rate }
    }

    /// The decay coefficient.
    #[must_use]
    pub fn rate(&self) -> f64 {
        self.rate
    }
}

impl Hook for WeightDecay {
    /// # Panics
    ///
    /// Panics if a parameter's gradient has a different shape from its data —
    /// the sum would otherwise broadcast into a silently wrong gradient.
    fn call(&self, params: &[Parameter]) {
        for param in params {
            // `data_and_grad` is the shared shape check; it returns `None` for
            // a parameter that has no data yet, which is the right no-op.
            let Some((data, _)) = data_and_grad(param) else {
                continue;
            };
            if let Some((node, grad)) = grad_parts(param) {
                node.set_data(grad + data * self.rate);
            }
        }
    }
}

/// Global gradient-norm clipping — Python's `ClipGrad`.
///
/// ```python
/// total_norm = math.sqrt(sum((p.grad.data ** 2).sum() for p in params))
/// rate = self.max_norm / (total_norm + 1e-6)
/// if rate < 1:
///     for param in params: param.grad.data *= rate
/// ```
///
/// The norm is taken over **every parameter at once**, and every gradient is
/// scaled by the same factor, so the update's direction is untouched and only
/// its length is capped. Clipping each parameter separately — the obvious
/// mistake — would bend the direction instead.
///
/// The `1e-6` guards the division when every gradient is zero. It also means
/// the rule is very slightly conservative, which is Python's behaviour and is
/// reproduced exactly.
///
/// # Examples
///
/// ```
/// use dezero::{ClipGrad, Hook, Parameter, Variable};
/// use ndarray::arr1;
///
/// let a = Parameter::new(arr1(&[0.0]).into_dyn());
/// let b = Parameter::new(arr1(&[0.0]).into_dyn());
/// a.set_grad(Some(Variable::new(arr1(&[3.0]).into_dyn())));
/// b.set_grad(Some(Variable::new(arr1(&[4.0]).into_dyn())));
///
/// // The joint norm is 5, so a cap of 1 scales both by about 1/5.
/// ClipGrad::new(1.0).call(&[a.clone(), b.clone()]);
///
/// let scaled = |p: &Parameter| p.grad().and_then(|g| g.data()).expect("grad")[[0]];
/// assert!((scaled(&a) - 0.6).abs() < 1e-6);
/// assert!((scaled(&b) - 0.8).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipGrad {
    max_norm: f64,
}

impl ClipGrad {
    /// The largest joint gradient norm to allow — Python's `max_norm`.
    #[must_use]
    pub fn new(max_norm: f64) -> Self {
        Self { max_norm }
    }

    /// The norm cap.
    #[must_use]
    pub fn max_norm(&self) -> f64 {
        self.max_norm
    }

    /// The joint L2 norm of every gradient in `params`, as the hook computes
    /// it.
    ///
    /// Exposed because the *decision* the hook makes is worth being able to
    /// inspect and test without also performing it.
    #[must_use]
    pub fn total_norm(params: &[Parameter]) -> f64 {
        params
            .iter()
            .filter_map(|param| param.grad().and_then(|g| g.data()))
            .map(|grad| grad.iter().map(|v| v * v).sum::<f64>())
            .sum::<f64>()
            .sqrt()
    }
}

impl Hook for ClipGrad {
    fn call(&self, params: &[Parameter]) {
        let rate = self.max_norm / (Self::total_norm(params) + 1e-6);
        if rate >= 1.0 {
            return;
        }
        for param in params {
            if let Some((node, grad)) = grad_parts(param) {
                node.set_data(grad * rate);
            }
        }
    }
}

/// Holds a set of parameters still — Python's `FreezeParam`.
///
/// Used for transfer learning: freeze a pretrained trunk and train only the
/// head. The hook ignores the list it is handed and clears the gradient of
/// every parameter it was *constructed* with, which is what makes it able to
/// freeze a subset.
///
/// # `grad = None`, not `grad = 0`
///
/// Python's rule is `p.grad = None`, and the port matches it. The difference is
/// not cosmetic: [`Optimizer::update_one`](crate::Optimizer::update_one) skips
/// a parameter with no gradient entirely, whereas a zero gradient still lets
/// [`MomentumSgd`](crate::MomentumSgd) move the weight by its decaying
/// velocity. Only the first of those is *frozen*.
///
/// Python cannot actually run this: `update()` snapshots the parameter list
/// before the hooks, so `update_one` reaches a frozen parameter and evaluates
/// `param.grad.data` on `None`, which raises `AttributeError`. The port keeps
/// the snapshot order — it is what [`ClipGrad`] needs — and survives, because
/// every `update_one` here starts from a `data_and_grad` that returns `None`.
/// See `docs/DIVERGENCES.md`.
///
/// # Examples
///
/// ```
/// use dezero::{FreezeParam, Layer, Linear, Optimizer, Sgd, Variable, sum_all};
/// use ndarray::arr2;
///
/// let trunk = Linear::with_in_size(2, 2);
/// let head = Linear::with_in_size(2, 1);
///
/// let mut optimizer = Sgd::new(0.1);
/// optimizer.set_params([trunk.params(), head.params()].concat());
/// optimizer.add_hook(FreezeParam::from_layer(&trunk));
///
/// let frozen = trunk.weight().data().expect("initialised");
/// let trainable = head.weight().data().expect("initialised");
///
/// let x = Variable::new(arr2(&[[1.0, 2.0]]).into_dyn());
/// sum_all(&head.forward(&trunk.forward(&x))).backward();
/// optimizer.update();
///
/// assert_eq!(trunk.weight().data(), Some(frozen), "the trunk did not move");
/// assert_ne!(head.weight().data(), Some(trainable), "the head did");
/// ```
#[derive(Debug, Clone, Default)]
pub struct FreezeParam {
    frozen: Vec<Parameter>,
}

impl FreezeParam {
    /// Freezes an explicit list of parameters.
    #[must_use]
    pub fn new(params: Vec<Parameter>) -> Self {
        Self { frozen: params }
    }

    /// Freezes every parameter of a layer, recursively — Python's
    /// `FreezeParam(layer)`.
    #[must_use]
    pub fn from_layer(layer: &dyn Layer) -> Self {
        Self::new(layer.params())
    }

    /// Adds another layer's parameters — Python's variadic
    /// `FreezeParam(*layers)`.
    #[must_use]
    pub fn and_layer(mut self, layer: &dyn Layer) -> Self {
        self.frozen.extend(layer.params());
        self
    }

    /// Adds one more parameter.
    #[must_use]
    pub fn and_param(mut self, param: Parameter) -> Self {
        self.frozen.push(param);
        self
    }

    /// The parameters this hook freezes.
    #[must_use]
    pub fn params(&self) -> &[Parameter] {
        &self.frozen
    }
}

impl Hook for FreezeParam {
    /// Clears the gradient of every frozen parameter, ignoring `_params` —
    /// exactly as Python's `__call__` ignores its argument.
    fn call(&self, _params: &[Parameter]) {
        for param in &self.frozen {
            param.cleargrad();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::variable::Variable;
    use crate::layers::Linear;
    use crate::optim::{Optimizer, Sgd};
    use ndarray::{ArrayD, arr1, arr2};

    fn param(values: &[f64], grad: &[f64]) -> Parameter {
        let p = Parameter::new(arr1(values).into_dyn());
        p.set_grad(Some(Variable::new(arr1(grad).into_dyn())));
        p
    }

    fn grad_of(param: &Parameter) -> ArrayD<f64> {
        param.grad().and_then(|g| g.data()).expect("a gradient")
    }

    // -- WeightDecay -------------------------------------------------------

    #[test]
    fn weight_decay_adds_a_multiple_of_the_weight() {
        let w = param(&[1.0, -2.0], &[0.1, 0.2]);
        WeightDecay::new(0.5).call(std::slice::from_ref(&w));
        assert_eq!(grad_of(&w), arr1(&[0.6, -0.8]).into_dyn());
    }

    #[test]
    fn weight_decay_of_zero_changes_nothing() {
        let w = param(&[1.0, -2.0], &[0.1, 0.2]);
        WeightDecay::new(0.0).call(std::slice::from_ref(&w));
        assert_eq!(grad_of(&w), arr1(&[0.1, 0.2]).into_dyn());
    }

    #[test]
    fn weight_decay_visits_every_parameter() {
        let a = param(&[1.0], &[0.0]);
        let b = param(&[2.0], &[0.0]);
        WeightDecay::new(1.0).call(&[a.clone(), b.clone()]);
        assert_eq!(grad_of(&a), arr1(&[1.0]).into_dyn());
        assert_eq!(grad_of(&b), arr1(&[2.0]).into_dyn());
    }

    #[test]
    fn weight_decay_works_on_a_matrix_parameter() {
        let w = Parameter::new(arr2(&[[1.0, -2.0], [3.0, -4.0]]).into_dyn());
        w.set_grad(Some(Variable::new(
            arr2(&[[0.1, 0.2], [0.3, 0.4]]).into_dyn(),
        )));
        WeightDecay::new(0.1).call(std::slice::from_ref(&w));
        // The last value is `0.3 + 0.1 * 3.0` computed in binary, not `0.6`;
        // the `hook_weight_decay` fixture records the same bits from numpy.
        assert_eq!(
            grad_of(&w),
            arr2(&[[0.2, 0.0], [0.3 + 0.1 * 3.0, 0.0]]).into_dyn()
        );
    }

    #[test]
    fn weight_decay_skips_a_parameter_with_no_data() {
        let w = Parameter::empty();
        w.set_grad(Some(Variable::new(arr1(&[1.0]).into_dyn())));
        WeightDecay::new(0.5).call(std::slice::from_ref(&w));
        assert_eq!(grad_of(&w), arr1(&[1.0]).into_dyn(), "left alone");
    }

    #[test]
    fn weight_decay_skips_a_parameter_with_no_gradient() {
        let w = Parameter::new(arr1(&[1.0]).into_dyn());
        WeightDecay::new(0.5).call(std::slice::from_ref(&w));
        assert!(w.grad().is_none(), "a hook does not invent a gradient");
    }

    #[test]
    #[should_panic(expected = "has a gradient of shape")]
    fn weight_decay_rejects_a_mismatched_gradient() {
        let w = Parameter::new(arr1(&[1.0, 2.0]).into_dyn());
        w.set_grad(Some(Variable::new(arr2(&[[1.0, 2.0]]).into_dyn())));
        WeightDecay::new(0.1).call(&[w]);
    }

    #[test]
    fn the_decay_rate_is_readable() {
        assert!((WeightDecay::new(0.25).rate() - 0.25).abs() < f64::EPSILON);
    }

    // -- ClipGrad ----------------------------------------------------------

    #[test]
    fn clipping_scales_every_gradient_by_the_same_factor() {
        let a = param(&[0.0], &[3.0]);
        let b = param(&[0.0], &[4.0]);
        ClipGrad::new(1.0).call(&[a.clone(), b.clone()]);

        let (ga, gb) = (grad_of(&a)[[0]], grad_of(&b)[[0]]);
        assert!((ga - 0.6).abs() < 1e-6, "{ga}");
        assert!((gb - 0.8).abs() < 1e-6, "{gb}");
        assert!(
            (ga / gb - 3.0 / 4.0).abs() < 1e-12,
            "the direction is unchanged"
        );
    }

    #[test]
    fn clipping_below_the_cap_is_a_no_op() {
        let a = param(&[0.0], &[3.0]);
        let b = param(&[0.0], &[4.0]);
        ClipGrad::new(100.0).call(&[a.clone(), b.clone()]);
        assert_eq!(grad_of(&a), arr1(&[3.0]).into_dyn());
        assert_eq!(grad_of(&b), arr1(&[4.0]).into_dyn());
    }

    #[test]
    fn the_norm_is_joint_not_per_parameter() {
        // Two gradients of norm 1 each: jointly sqrt(2), so a cap of 1.2 fires
        // even though neither one exceeds it on its own.
        let a = param(&[0.0], &[1.0]);
        let b = param(&[0.0], &[1.0]);
        assert!((ClipGrad::total_norm(&[a.clone(), b.clone()]) - 2.0_f64.sqrt()).abs() < 1e-12);

        ClipGrad::new(1.2).call(&[a.clone(), b.clone()]);
        assert!(
            grad_of(&a)[[0]] < 1.0,
            "clipped despite being under the cap"
        );
    }

    #[test]
    fn clipping_an_all_zero_gradient_does_not_divide_by_zero() {
        let w = param(&[1.0], &[0.0]);
        ClipGrad::new(0.5).call(std::slice::from_ref(&w));
        assert_eq!(grad_of(&w), arr1(&[0.0]).into_dyn());
        assert!(grad_of(&w)[[0]].is_finite());
    }

    /// The epsilon makes the cap very slightly conservative: at exactly
    /// `max_norm` the rate is a hair below 1, so the gradient *is* scaled.
    #[test]
    fn a_norm_exactly_at_the_cap_is_scaled_by_the_epsilon() {
        let w = param(&[0.0], &[5.0]);
        ClipGrad::new(5.0).call(std::slice::from_ref(&w));
        let scaled = grad_of(&w)[[0]];
        assert!(scaled < 5.0 && scaled > 4.999_99, "{scaled}");
    }

    #[test]
    fn clipping_ignores_a_parameter_without_a_gradient() {
        let with = param(&[0.0], &[10.0]);
        let without = Parameter::new(arr1(&[1.0]).into_dyn());
        ClipGrad::new(1.0).call(&[with.clone(), without.clone()]);
        assert!(grad_of(&with)[[0]] < 1.001);
        assert!(without.grad().is_none());
    }

    #[test]
    fn clipping_handles_a_matrix_gradient() {
        let w = Parameter::new(arr2(&[[0.0, 0.0], [0.0, 0.0]]).into_dyn());
        w.set_grad(Some(Variable::new(
            arr2(&[[3.0, 0.0], [0.0, 4.0]]).into_dyn(),
        )));
        ClipGrad::new(1.0).call(std::slice::from_ref(&w));
        let g = grad_of(&w);
        assert!((g[[0, 0]] - 0.6).abs() < 1e-6);
        assert!((g[[1, 1]] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn the_cap_is_readable() {
        assert!((ClipGrad::new(7.5).max_norm() - 7.5).abs() < f64::EPSILON);
        assert_eq!(ClipGrad::total_norm(&[]), 0.0, "no gradients, no norm");
    }

    // -- FreezeParam -------------------------------------------------------

    #[test]
    fn freezing_clears_the_gradients_it_was_given() {
        let frozen = param(&[1.0], &[5.0]);
        let free = param(&[1.0], &[5.0]);

        FreezeParam::new(vec![frozen.clone()]).call(&[frozen.clone(), free.clone()]);

        assert!(frozen.grad().is_none());
        assert_eq!(grad_of(&free), arr1(&[5.0]).into_dyn(), "untouched");
    }

    #[test]
    fn freezing_ignores_the_list_it_is_handed() {
        // Python's `__call__` never looks at its argument; nor does this.
        let frozen = param(&[1.0], &[5.0]);
        FreezeParam::new(vec![frozen.clone()]).call(&[]);
        assert!(frozen.grad().is_none());
    }

    #[test]
    fn a_layer_freezes_all_of_its_parameters() {
        let layer = Linear::with_in_size(2, 3);
        let hook = FreezeParam::from_layer(&layer);
        assert_eq!(hook.params().len(), 2, "W and b");

        for p in layer.params() {
            p.set_grad(Some(Variable::new(arr1(&[1.0]).into_dyn())));
        }
        hook.call(&[]);
        assert!(layer.params().iter().all(|p| p.grad().is_none()));
    }

    #[test]
    fn freezing_accumulates_layers_and_parameters() {
        let first = Linear::with_in_size(2, 2);
        let second = Linear::with_in_size(2, 2);
        let loose = Parameter::new(arr1(&[1.0]).into_dyn());

        let hook = FreezeParam::from_layer(&first)
            .and_layer(&second)
            .and_param(loose.clone());
        assert_eq!(hook.params().len(), 5, "2 + 2 + 1");
    }

    #[test]
    fn an_empty_freeze_is_harmless() {
        let w = param(&[1.0], &[2.0]);
        FreezeParam::default().call(std::slice::from_ref(&w));
        assert_eq!(grad_of(&w), arr1(&[2.0]).into_dyn());
    }

    /// The reason freezing clears rather than zeroes: a frozen parameter must
    /// not move, and only a missing gradient guarantees that under every rule.
    #[test]
    fn a_frozen_parameter_does_not_move_under_sgd() {
        let frozen = param(&[10.0], &[1.0]);
        let free = param(&[10.0], &[1.0]);

        let mut optimizer = Sgd::new(0.5);
        optimizer.set_params(vec![frozen.clone(), free.clone()]);
        optimizer.add_hook(FreezeParam::new(vec![frozen.clone()]));
        optimizer.update();

        assert_eq!(frozen.data(), Some(arr1(&[10.0]).into_dyn()));
        assert_eq!(free.data(), Some(arr1(&[9.5]).into_dyn()));
    }

    // -- composition -------------------------------------------------------

    #[test]
    fn hooks_run_in_the_order_they_were_added() {
        // Decay then clip is not the same as clip then decay; the order the
        // optimizer applies them in must be the insertion order.
        let decay_then_clip = param(&[10.0], &[1.0]);
        let clip_then_decay = param(&[10.0], &[1.0]);

        let mut first = Sgd::new(0.0);
        first.set_params(vec![decay_then_clip.clone()]);
        first.add_hook(WeightDecay::new(1.0));
        first.add_hook(ClipGrad::new(1.0));
        first.update();

        let mut second = Sgd::new(0.0);
        second.set_params(vec![clip_then_decay.clone()]);
        second.add_hook(ClipGrad::new(1.0));
        second.add_hook(WeightDecay::new(1.0));
        second.update();

        // (1 + 10) clipped to ~1 vs 1 clipped to ~1 then + 10.
        assert!(grad_of(&decay_then_clip)[[0]] < 1.001);
        assert!(grad_of(&clip_then_decay)[[0]] > 10.9);
    }

    #[test]
    fn a_hook_can_be_shared_between_optimizers() {
        let shared: Rc<dyn Hook> = Rc::new(WeightDecay::new(0.5));
        let w = param(&[2.0], &[0.0]);

        let mut a = Sgd::new(0.0);
        a.set_params(vec![w.clone()]);
        a.add_shared_hook(Rc::clone(&shared));

        let mut b = Sgd::new(0.0);
        b.set_params(vec![w.clone()]);
        b.add_shared_hook(shared);

        a.update();
        b.update();
        assert_eq!(grad_of(&w), arr1(&[2.0]).into_dyn(), "0.5 * 2, twice");
    }

    #[test]
    fn hooks_render_inside_an_optimizers_debug_output() {
        let mut optimizer = Sgd::new(0.1);
        optimizer.add_hook(ClipGrad::new(2.0));
        let rendered = format!("{optimizer:?}");
        assert!(rendered.contains("ClipGrad"), "{rendered}");
    }
}
