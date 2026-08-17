//! [`MomentumSgd`]: gradient descent with velocity (step 46).
//!
//! Port of `MomentumSGD` in `vendor/dezero-python/dezero/optimizers.py`:
//!
//! ```python
//! v = self.vs[id(param)]      # zeros_like(param.data) the first time
//! v *= self.momentum
//! v -= self.lr * param.grad.data
//! param.data += v
//! ```
//!
//! The velocity is a running, exponentially decayed sum of past gradients. It
//! is what carries an update across a flat stretch of the loss surface and what
//! damps the zig-zag across a narrow valley — the failure mode plain
//! [`Sgd`](crate::Sgd) shows on the book's own Rosenbrock example.

use std::collections::HashMap;

use ndarray::ArrayD;

use crate::core::parameter::Parameter;
use crate::optim::{Optimizer, data_and_grad};

/// Gradient descent with momentum — Python's `MomentumSGD`.
///
/// Named `MomentumSgd` rather than `MomentumSGD` to follow Rust's convention
/// for acronyms in type names.
///
/// # Examples
///
/// ```
/// use dezero::{MomentumSgd, Optimizer, Parameter, Variable};
/// use ndarray::arr1;
///
/// let w = Parameter::new(arr1(&[0.0]).into_dyn());
/// w.set_grad(Some(Variable::new(arr1(&[1.0]).into_dyn())));
///
/// let mut optimizer = MomentumSgd::new(0.1, 0.9);
/// optimizer.set_params(vec![w.clone()]);
///
/// // First step: no velocity yet, so it is plain gradient descent.
/// optimizer.update();
/// assert_eq!(w.data(), Some(arr1(&[-0.1]).into_dyn()));
///
/// // Second: the velocity carries 0.9 of the first step into this one.
/// optimizer.update();
/// let moved = w.data().expect("data").sum();
/// assert!((moved + 0.29).abs() < 1e-12, "{moved}");
/// ```
#[derive(Debug, Clone)]
pub struct MomentumSgd {
    params: Vec<Parameter>,
    learning_rate: f64,
    momentum: f64,
    /// One velocity per parameter, keyed by [`Variable::id`](crate::Variable::id)
    /// — Python's `self.vs[id(param)]`.
    ///
    /// Keying on a pointer is safe because `self.params` holds a strong handle
    /// to every registered parameter, so none of them can be freed and have its
    /// address reused while its velocity is live.
    velocities: HashMap<usize, ArrayD<f64>>,
}

impl MomentumSgd {
    /// Creates the optimizer. Python's defaults are `lr=0.01, momentum=0.9`.
    #[must_use]
    pub fn new(learning_rate: f64, momentum: f64) -> Self {
        Self {
            params: Vec::new(),
            learning_rate,
            momentum,
            velocities: HashMap::new(),
        }
    }

    /// The learning rate.
    #[must_use]
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// The momentum coefficient.
    #[must_use]
    pub fn momentum(&self) -> f64 {
        self.momentum
    }

    /// The accumulated velocity of one parameter, if it has taken a step.
    #[must_use]
    pub fn velocity(&self, param: &Parameter) -> Option<&ArrayD<f64>> {
        self.velocities.get(&param.id())
    }

    /// Discards every accumulated velocity, restarting from rest.
    pub fn clear_velocities(&mut self) {
        self.velocities.clear();
    }
}

impl Optimizer for MomentumSgd {
    fn params(&self) -> &[Parameter] {
        &self.params
    }

    fn set_params(&mut self, params: Vec<Parameter>) {
        self.params = params;
    }

    /// `v = momentum * v - lr * grad; param.data += v`.
    ///
    /// # Panics
    ///
    /// Panics if the parameter's gradient has a different shape from its data.
    fn update_one(&mut self, param: &Parameter) {
        let Some((data, grad)) = data_and_grad(param) else {
            return;
        };

        let velocity = self
            .velocities
            .entry(param.id())
            .or_insert_with(|| ArrayD::zeros(data.raw_dim()));

        // Python mutates the stored array in place; so does this, which is what
        // makes the velocity persist across calls.
        *velocity *= self.momentum;
        *velocity -= &(grad * self.learning_rate);

        param.set_data(data + &*velocity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::variable::Variable;
    use crate::layers::{Layer, Linear};
    use crate::mean_squared_error;
    use ndarray::{arr1, arr2};

    fn with_grad(values: &[f64], grad: &[f64]) -> Parameter {
        let p = Parameter::new(arr1(values).into_dyn());
        p.set_grad(Some(Variable::new(arr1(grad).into_dyn())));
        p
    }

    #[test]
    fn the_first_step_is_plain_gradient_descent() {
        let w = with_grad(&[1.0], &[2.0]);
        let mut optimizer = MomentumSgd::new(0.5, 0.9);
        optimizer.set_params(vec![w.clone()]);
        optimizer.update();
        assert_eq!(w.data(), Some(arr1(&[0.0]).into_dyn()), "1 - 0.5 * 2");
    }

    #[test]
    fn velocity_accumulates_across_steps() {
        // lr = 0.1, momentum = 0.9, constant gradient of 1:
        //   v1 = -0.1                    -> x = -0.1
        //   v2 = 0.9(-0.1) - 0.1 = -0.19 -> x = -0.29
        //   v3 = 0.9(-0.19) - 0.1 = -0.271 -> x = -0.561
        let w = with_grad(&[0.0], &[1.0]);
        let mut optimizer = MomentumSgd::new(0.1, 0.9);
        optimizer.set_params(vec![w.clone()]);

        for expected in [-0.1, -0.29, -0.561] {
            optimizer.update();
            let value = w.data().expect("data").sum();
            assert!((value - expected).abs() < 1e-12, "{value} vs {expected}");
        }
    }

    #[test]
    fn zero_momentum_is_exactly_plain_descent() {
        let momentum_free = with_grad(&[0.0], &[1.0]);
        let mut optimizer = MomentumSgd::new(0.25, 0.0);
        optimizer.set_params(vec![momentum_free.clone()]);

        let plain = with_grad(&[0.0], &[1.0]);
        let mut reference = crate::Sgd::new(0.25);
        reference.set_params(vec![plain.clone()]);

        for _ in 0..5 {
            optimizer.update();
            reference.update();
            assert_eq!(momentum_free.data(), plain.data());
        }
    }

    #[test]
    fn velocity_carries_the_parameter_after_the_gradient_vanishes() {
        // Two steps of a real gradient, then a zero one: momentum keeps moving.
        let w = with_grad(&[0.0], &[1.0]);
        let mut optimizer = MomentumSgd::new(0.1, 0.9);
        optimizer.set_params(vec![w.clone()]);
        optimizer.update();
        optimizer.update();
        let coasting_from = w.data().expect("data").sum();

        w.set_grad(Some(Variable::new(arr1(&[0.0]).into_dyn())));
        optimizer.update();
        let after = w.data().expect("data").sum();
        assert!(
            after < coasting_from,
            "it should coast: {coasting_from} -> {after}"
        );
        // v was -0.19; decayed by 0.9 it is -0.171.
        assert!((after - (coasting_from - 0.171)).abs() < 1e-12);
    }

    #[test]
    fn each_parameter_keeps_its_own_velocity() {
        let fast = with_grad(&[0.0], &[1.0]);
        let slow = with_grad(&[0.0], &[0.1]);
        let mut optimizer = MomentumSgd::new(0.1, 0.9);
        optimizer.set_params(vec![fast.clone(), slow.clone()]);
        optimizer.update();
        optimizer.update();

        let fast_velocity = optimizer.velocity(&fast).expect("velocity").sum();
        let slow_velocity = optimizer.velocity(&slow).expect("velocity").sum();
        assert!((fast_velocity + 0.19).abs() < 1e-12);
        assert!((slow_velocity + 0.019).abs() < 1e-12);
        assert!(optimizer.velocity(&Parameter::empty()).is_none());
    }

    #[test]
    fn clearing_velocities_restarts_from_rest() {
        let w = with_grad(&[0.0], &[1.0]);
        let mut optimizer = MomentumSgd::new(0.1, 0.9);
        optimizer.set_params(vec![w.clone()]);
        optimizer.update();
        optimizer.update();

        optimizer.clear_velocities();
        assert!(optimizer.velocity(&w).is_none());

        let before = w.data().expect("data").sum();
        optimizer.update();
        let step = w.data().expect("data").sum() - before;
        assert!((step + 0.1).abs() < 1e-12, "a fresh first step, {step}");
    }

    #[test]
    fn the_velocity_has_the_parameters_shape() {
        let w = Parameter::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn());
        w.set_grad(Some(Variable::new(
            arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn(),
        )));
        let mut optimizer = MomentumSgd::new(0.5, 0.5);
        optimizer.set_params(vec![w.clone()]);
        optimizer.update();

        assert_eq!(optimizer.velocity(&w).expect("velocity").shape(), &[2, 2]);
        assert_eq!(w.data(), Some(arr2(&[[0.5, 2.0], [3.0, 3.5]]).into_dyn()));
    }

    #[test]
    fn an_uninitialised_parameter_gets_no_velocity() {
        let w = Parameter::empty();
        w.set_grad(Some(Variable::new(arr1(&[1.0]).into_dyn())));
        let mut optimizer = MomentumSgd::new(0.1, 0.9);
        optimizer.set_params(vec![w.clone()]);
        optimizer.update();
        assert!(w.data().is_none());
        assert!(optimizer.velocity(&w).is_none());
    }

    #[test]
    fn accessors_report_the_hyperparameters() {
        let optimizer = MomentumSgd::new(0.01, 0.95);
        assert!((optimizer.learning_rate() - 0.01).abs() < f64::EPSILON);
        assert!((optimizer.momentum() - 0.95).abs() < f64::EPSILON);
    }

    /// End to end, against the same problem [`Sgd`] solves: momentum reaches a
    /// lower loss in the same number of steps at the same learning rate.
    #[test]
    fn momentum_converges_faster_than_plain_descent() {
        let x = Variable::new(arr2(&[[1.0], [2.0], [3.0]]).into_dyn());
        let target = Variable::new(arr2(&[[3.0], [5.0], [7.0]]).into_dyn());

        let train = |model: &Linear, optimizer: &mut dyn Optimizer, steps: usize| -> f64 {
            optimizer.setup(model);
            for _ in 0..steps {
                let loss = mean_squared_error(&model.forward(&x), &target);
                model.cleargrads();
                loss.backward();
                optimizer.update();
            }
            mean_squared_error(&model.forward(&x), &target)
                .data()
                .expect("loss")
                .sum()
        };

        // Identical, *pinned* starting weights: only the update rule differs,
        // and the comparison owes nothing to which weight the initialiser drew.
        let plain = Linear::with_in_size(1, 1);
        let accelerated = Linear::with_in_size(1, 1);
        for model in [&plain, &accelerated] {
            model.weight().set_data(arr2(&[[0.0]]).into_dyn());
        }

        let plain_loss = train(&plain, &mut crate::Sgd::new(0.01), 100);
        let momentum_loss = train(&accelerated, &mut MomentumSgd::new(0.01, 0.9), 100);

        assert!(
            momentum_loss < plain_loss,
            "momentum {momentum_loss} should beat plain {plain_loss}"
        );
    }
}
