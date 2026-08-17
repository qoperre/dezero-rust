//! [`Sgd`]: plain gradient descent (step 46).
//!
//! Port of `SGD` in `vendor/dezero-python/dezero/optimizers.py`, whose whole
//! rule is one line:
//!
//! ```python
//! param.data -= self.lr * param.grad.data
//! ```

use std::rc::Rc;

use crate::core::parameter::Parameter;
use crate::optim::{Hook, Hooks, Optimizer, data_and_grad};

/// Stochastic gradient descent — Python's `SGD`.
///
/// Named `Sgd` rather than `SGD` to follow Rust's convention for acronyms in
/// type names; it is the reference's `optimizers.SGD` in every other respect.
///
/// # Examples
///
/// ```
/// use dezero::{Optimizer, Parameter, Sgd, Variable};
/// use ndarray::arr1;
///
/// let w = Parameter::new(arr1(&[1.0, 2.0]).into_dyn());
/// w.set_grad(Some(Variable::new(arr1(&[10.0, -10.0]).into_dyn())));
///
/// let mut optimizer = Sgd::new(0.1);
/// optimizer.set_params(vec![w.clone()]);
/// optimizer.update();
///
/// // Each weight moved by -lr * grad.
/// assert_eq!(w.data(), Some(arr1(&[0.0, 3.0]).into_dyn()));
/// ```
#[derive(Debug, Clone)]
pub struct Sgd {
    params: Vec<Parameter>,
    hooks: Hooks,
    learning_rate: f64,
}

impl Sgd {
    /// Creates the optimizer with a learning rate.
    ///
    /// Python's default is `lr=0.01`; there is no default here, because an
    /// unstated learning rate is the single most common reason a training run
    /// silently does nothing.
    #[must_use]
    pub fn new(learning_rate: f64) -> Self {
        Self {
            params: Vec::new(),
            hooks: Hooks::new(),
            learning_rate,
        }
    }

    /// The learning rate.
    #[must_use]
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Changes the learning rate, for schedules that decay it over training.
    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }
}

impl Optimizer for Sgd {
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

    /// `param.data -= lr * param.grad.data`.
    ///
    /// # Panics
    ///
    /// Panics if the parameter's gradient has a different shape from its data.
    fn update_one(&mut self, param: &Parameter) {
        let Some((data, grad)) = data_and_grad(param) else {
            return;
        };
        param.set_data(data - grad * self.learning_rate);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::variable::Variable;
    use crate::layers::{Layer, Linear};
    use crate::{Sequential, mean_squared_error};
    use ndarray::{arr1, arr2};

    fn with_grad(values: &[f64], grad: &[f64]) -> Parameter {
        let p = Parameter::new(arr1(values).into_dyn());
        p.set_grad(Some(Variable::new(arr1(grad).into_dyn())));
        p
    }

    #[test]
    fn one_step_moves_against_the_gradient() {
        let w = with_grad(&[1.0, 2.0, 3.0], &[1.0, 0.0, -2.0]);
        let mut optimizer = Sgd::new(0.5);
        optimizer.set_params(vec![w.clone()]);
        optimizer.update();
        assert_eq!(w.data(), Some(arr1(&[0.5, 2.0, 4.0]).into_dyn()));
    }

    #[test]
    fn a_zero_learning_rate_changes_nothing() {
        let w = with_grad(&[1.0, 2.0], &[5.0, 5.0]);
        let mut optimizer = Sgd::new(0.0);
        optimizer.set_params(vec![w.clone()]);
        optimizer.update();
        assert_eq!(w.data(), Some(arr1(&[1.0, 2.0]).into_dyn()));
    }

    #[test]
    fn sgd_is_stateless_across_steps() {
        // Two steps with the same gradient move exactly twice as far as one:
        // nothing accumulates.
        let w = with_grad(&[0.0], &[1.0]);
        let mut optimizer = Sgd::new(0.25);
        optimizer.set_params(vec![w.clone()]);
        optimizer.update();
        optimizer.update();
        assert_eq!(w.data(), Some(arr1(&[-0.5]).into_dyn()));
    }

    #[test]
    fn the_learning_rate_can_be_rescheduled() {
        let w = with_grad(&[0.0], &[1.0]);
        let mut optimizer = Sgd::new(1.0);
        optimizer.set_params(vec![w.clone()]);
        optimizer.update();
        assert_eq!(w.data(), Some(arr1(&[-1.0]).into_dyn()));

        optimizer.set_learning_rate(0.5);
        assert!((optimizer.learning_rate() - 0.5).abs() < f64::EPSILON);
        optimizer.update();
        assert_eq!(w.data(), Some(arr1(&[-1.5]).into_dyn()));
    }

    #[test]
    fn an_uninitialised_parameter_is_left_alone() {
        let w = Parameter::empty();
        w.set_grad(Some(Variable::new(arr1(&[1.0]).into_dyn())));
        let mut optimizer = Sgd::new(0.1);
        optimizer.set_params(vec![w.clone()]);
        optimizer.update();
        assert!(w.data().is_none());
    }

    #[test]
    fn a_matrix_parameter_updates_elementwise() {
        let w = Parameter::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn());
        w.set_grad(Some(Variable::new(
            arr2(&[[1.0, 1.0], [2.0, 2.0]]).into_dyn(),
        )));
        let mut optimizer = Sgd::new(0.5);
        optimizer.set_params(vec![w.clone()]);
        optimizer.update();
        assert_eq!(w.data(), Some(arr2(&[[0.5, 1.5], [2.0, 3.0]]).into_dyn()));
    }

    /// End to end: a real network, a real loss, and a parameter that visibly
    /// moves in the direction that lowers it.
    #[test]
    fn training_a_layer_reduces_its_loss() {
        let model = Linear::with_in_size(1, 1);
        let x = Variable::new(arr2(&[[1.0], [2.0], [3.0]]).into_dyn());
        let target = Variable::new(arr2(&[[3.0], [5.0], [7.0]]).into_dyn()); // y = 2x + 1

        let loss_of = |model: &Linear| {
            mean_squared_error(&model.forward(&x), &target)
                .data()
                .expect("loss")
                .sum()
        };

        let before = loss_of(&model);
        let weight_before = model.weight().data();

        let mut optimizer = Sgd::new(0.05);
        optimizer.setup(&model);
        for _ in 0..2000 {
            let loss = mean_squared_error(&model.forward(&x), &target);
            model.cleargrads();
            loss.backward();
            optimizer.update();
        }

        let after = loss_of(&model);
        assert!(after < before, "the loss must fall: {before} -> {after}");
        assert_ne!(model.weight().data(), weight_before, "the weight moved");
        assert!(after < 1e-6, "and the fit is essentially exact: {after}");
    }

    #[test]
    fn setup_reaches_the_parameters_of_a_nested_model() {
        let model = Sequential::new(vec![
            Box::new(Linear::with_in_size(2, 3)) as Box<dyn Layer>,
            Box::new(Linear::with_in_size(3, 1)),
        ]);
        let mut optimizer = Sgd::new(0.1);
        optimizer.setup(&model);
        assert_eq!(optimizer.params().len(), 4);

        let x = Variable::new(arr2(&[[1.0, 2.0]]).into_dyn());
        let before: Vec<_> = optimizer.params().iter().map(|p| p.data()).collect();

        crate::sum_all(&model.forward(&x)).backward();
        optimizer.update();

        let after: Vec<_> = optimizer.params().iter().map(|p| p.data()).collect();
        assert!(
            before.iter().zip(&after).all(|(b, a)| b != a),
            "every parameter of every layer moved"
        );
    }
}
