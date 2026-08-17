//! Activations: [`sigmoid`], [`relu`] and [`softmax`] (steps 43 and 47).
//!
//! Port of the "activation function" section of
//! `vendor/dezero-python/dezero/functions.py`.
//!
//! All three forwards are numerically deliberate, and the port keeps the
//! reference's exact arithmetic rather than the textbook formula:
//!
//! * `sigmoid` is `tanh(x/2)/2 + 1/2`, not `1/(1 + exp(-x))`. The two are
//!   algebraically identical; the second overflows `exp` for large negative
//!   `x`, the first cannot.
//! * `softmax` subtracts the per-row maximum before exponentiating. Again
//!   algebraically a no-op — every term of the ratio is scaled by the same
//!   constant — and again the difference between a finite answer and `inf/inf`.
//!
//! `sigmoid` and `softmax` differentiate from their **output**, which is free
//! (the forward result is already in the graph); `relu` differentiates from its
//! **input**, because the sign it needs is exactly what the forward threw away.

use ndarray::ArrayD;

use crate::core::function::{Op, apply1};
use crate::core::ops::{mul, one, scalar, sub};
use crate::core::variable::Variable;
use crate::functions::reduce::sum;
use crate::utils::array::{max_keepdims, sum_keepdims};
use crate::utils::shape::normalize_axis;

/// The data a variable must hold.
///
/// # Panics
///
/// Panics if `v` holds no data; [`apply`](crate::apply) has already rejected
/// that case for anything reaching an `Op`.
fn data_of(v: &Variable, op: &str) -> ArrayD<f64> {
    v.data()
        .unwrap_or_else(|| panic!("dezero: {op} needs a variable that holds data"))
}

// ---------------------------------------------------------------------------
// Sigmoid
// ---------------------------------------------------------------------------

/// `y = 1 / (1 + exp(-x))`, elementwise.
#[derive(Debug, Clone, Copy)]
pub struct Sigmoid;

impl Op for Sigmoid {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = one(xs, "Sigmoid", "input");
        // Python: `y = xp.tanh(x * 0.5) * 0.5 + 0.5  # Better implementation`.
        // Reproduced operation for operation, not merely up to algebra: a
        // different-but-equivalent formula would drift from the reference in the
        // last bits and take the parity fixtures with it.
        vec![x.mapv(|v| (v * 0.5).tanh() * 0.5 + 0.5)]
    }

    fn backward(
        &self,
        _inputs: &[Variable],
        outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let y = one(outputs, "Sigmoid", "output");
        let gy = one(gys, "Sigmoid", "output gradient");
        // gx = gy * y * (1 - y)
        vec![mul(&mul(gy, y), &sub(&scalar(1.0), y))]
    }
}

/// The logistic sigmoid, elementwise — Python's `dezero.functions.sigmoid`.
///
/// # Examples
///
/// ```
/// use dezero::{sigmoid, Variable};
/// use ndarray::arr1;
///
/// let x = Variable::new(arr1(&[-1.0, 0.0, 1.0]).into_dyn());
/// let y = sigmoid(&x);
///
/// let values = y.data().expect("data");
/// assert!((values[[1]] - 0.5).abs() < 1e-15);
/// assert!((values[[0]] + values[[2]] - 1.0).abs() < 1e-15, "sigmoid is odd about (0, 1/2)");
///
/// // y' = y (1 - y), largest at the origin.
/// y.backward();
/// let slope = x.grad().and_then(|g| g.data()).expect("gradient");
/// assert!((slope[[1]] - 0.25).abs() < 1e-15);
/// ```
///
/// # Panics
///
/// Panics if `x` holds no data.
#[must_use]
pub fn sigmoid(x: &Variable) -> Variable {
    apply1(Sigmoid, &[x])
}

// ---------------------------------------------------------------------------
// ReLU
// ---------------------------------------------------------------------------

/// `y = max(x, 0)`, elementwise.
#[derive(Debug, Clone, Copy)]
pub struct ReLU;

impl Op for ReLU {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = one(xs, "ReLU", "input");
        vec![x.mapv(|v| v.max(0.0))]
    }

    fn backward(
        &self,
        inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let x = one(inputs, "ReLU", "input");
        let gy = one(gys, "ReLU", "output gradient");
        // Python: `mask = x.data > 0; gx = gy * mask`. The mask is a *constant*
        // -- a detached variable with no creator -- so the multiplication is a
        // graph node while the threshold itself is not differentiated. At x = 0
        // the derivative is taken to be 0, matching the strict `>`.
        let mask = Variable::new(data_of(x, "ReLU").mapv(|v| f64::from(v > 0.0)));
        vec![mul(gy, &mask)]
    }
}

/// The rectifier `max(x, 0)`, elementwise — Python's
/// `dezero.functions.relu`.
///
/// The derivative at exactly `x = 0` is defined to be 0 (Python's `x.data > 0`
/// is strict); the function is not differentiable there, so any convention is
/// a choice, and this is the reference's.
///
/// # Examples
///
/// ```
/// use dezero::{relu, Variable};
/// use ndarray::arr1;
///
/// let x = Variable::new(arr1(&[-2.0, 0.0, 3.0]).into_dyn());
/// let y = relu(&x);
/// assert_eq!(y.data(), Some(arr1(&[0.0, 0.0, 3.0]).into_dyn()));
///
/// y.backward();
/// assert_eq!(
///     x.grad().and_then(|g| g.data()),
///     Some(arr1(&[0.0, 0.0, 1.0]).into_dyn()),
///     "the gradient passes through only where x was positive"
/// );
/// ```
///
/// # Panics
///
/// Panics if `x` holds no data.
#[must_use]
pub fn relu(x: &Variable) -> Variable {
    apply1(ReLU, &[x])
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

/// `y = exp(x) / sum(exp(x), axis)`, computed stably.
#[derive(Debug, Clone, Copy)]
pub struct Softmax {
    axis: isize,
}

impl Softmax {
    /// Creates the op for a given axis. Negative axes count back from the last,
    /// as in numpy.
    #[must_use]
    pub fn new(axis: isize) -> Self {
        Self { axis }
    }
}

/// The stable softmax of a raw array, statement for statement as Python has it:
///
/// ```text
/// y = x - x.max(axis=axis, keepdims=True)
/// y = xp.exp(y)
/// y /= y.sum(axis=axis, keepdims=True)
/// ```
fn softmax_array(x: &ArrayD<f64>, axis: usize) -> ArrayD<f64> {
    let mut y = x.clone();
    y -= &max_keepdims(x, axis);
    y.mapv_inplace(f64::exp);
    let totals = sum_keepdims(&y, axis);
    y /= &totals;
    y
}

impl Op for Softmax {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = one(xs, "Softmax", "input");
        let axis = normalize_axis(self.axis, x.ndim());
        vec![softmax_array(x, axis)]
    }

    fn backward(
        &self,
        _inputs: &[Variable],
        outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let y = one(outputs, "Softmax", "output");
        let gy = one(gys, "Softmax", "output gradient");
        // The Jacobian is `diag(y) - y y^T` per row, which never has to be
        // materialised: contracting it with `gy` is
        //   gx = y * gy - y * sum(y * gy)
        // -- Python's three lines, in Variable arithmetic.
        let gx = mul(y, gy);
        let sumdx = sum(&gx, self.axis, true);
        vec![sub(&gx, &mul(y, &sumdx))]
    }
}

/// The softmax over the columns of each row — Python's
/// `dezero.functions.softmax(x)` with its default `axis=1`.
///
/// This is the classification case: rows are samples, columns are classes, and
/// each row of the result sums to 1. Use [`softmax_axis`] to reduce a different
/// axis.
///
/// # Examples
///
/// ```
/// use dezero::{softmax, Variable};
/// use ndarray::arr2;
///
/// let x = Variable::new(arr2(&[[0.0, 0.0, 0.0], [0.0, 1000.0, 0.0]]).into_dyn());
/// let y = softmax(&x).data().expect("data");
///
/// // Uniform logits give a uniform distribution ...
/// assert!((y[[0, 0]] - 1.0 / 3.0).abs() < 1e-15);
/// // ... and a logit large enough to overflow `exp` still comes out finite.
/// assert_eq!(y[[1, 1]], 1.0);
///
/// for row in 0..2 {
///     let total: f64 = (0..3).map(|c| y[[row, c]]).sum();
///     assert!((total - 1.0).abs() < 1e-15, "each row is a distribution");
/// }
/// ```
///
/// # Panics
///
/// Panics if `x` is 0-dimensional (there is no axis 1 to reduce) or holds no
/// data.
#[must_use]
pub fn softmax(x: &Variable) -> Variable {
    softmax_axis(x, 1)
}

/// The softmax over an explicit axis — Python's
/// `dezero.functions.softmax(x, axis)`.
///
/// Negative axes count back from the last, as in numpy.
///
/// # Examples
///
/// ```
/// use dezero::{softmax, softmax_axis, Variable};
/// use ndarray::arr2;
///
/// let x = Variable::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn());
/// assert_eq!(softmax_axis(&x, -1).data(), softmax(&x).data());
///
/// // Down the columns instead: each *column* now sums to 1.
/// let y = softmax_axis(&x, 0).data().expect("data");
/// assert!((y[[0, 0]] + y[[1, 0]] - 1.0).abs() < 1e-15);
/// ```
///
/// # Panics
///
/// Panics if `axis` is out of bounds for `x`, or if `x` holds no data.
#[must_use]
pub fn softmax_axis(x: &Variable, axis: isize) -> Variable {
    apply1(Softmax::new(axis), &[x])
}

// ---------------------------------------------------------------------------
// Dropout (step 54)
// ---------------------------------------------------------------------------

/// `y = x * mask / (1 - ratio)` with a caller-supplied mask.
///
/// This is the testable half of [`dropout`]. The mask is a **constant** — a
/// detached variable with no creator — so the multiplication is a graph node
/// while the mask itself is never differentiated, exactly as in the reference.
///
/// Inverted dropout: scaling by `1/(1 - ratio)` during training is what lets
/// test time be a plain identity instead of a rescale.
///
/// # Panics
///
/// Panics if `x` holds no data, if `mask`'s shape differs from `x`'s, or if
/// `ratio` is outside `[0, 1)`. `ratio == 1` would drop everything and divide
/// by zero.
///
/// # Examples
///
/// ```
/// use dezero::{dropout_with_mask, Variable};
/// use ndarray::arr1;
///
/// let x = Variable::new(arr1(&[1.0, 2.0, 3.0, 4.0]).into_dyn());
/// let mask = arr1(&[1.0, 0.0, 1.0, 0.0]).into_dyn();
///
/// // Survivors are scaled by 1/(1 - 0.5) = 2; the dropped ones vanish.
/// let y = dropout_with_mask(&x, 0.5, &mask);
/// assert_eq!(y.data(), Some(arr1(&[2.0, 0.0, 6.0, 0.0]).into_dyn()));
///
/// // The gradient follows the same mask.
/// y.backward();
/// assert_eq!(
///     x.grad().and_then(|g| g.data()),
///     Some(arr1(&[2.0, 0.0, 2.0, 0.0]).into_dyn())
/// );
/// ```
#[must_use]
pub fn dropout_with_mask(x: &Variable, ratio: f64, mask: &ArrayD<f64>) -> Variable {
    assert!(
        (0.0..1.0).contains(&ratio),
        "dezero: dropout ratio must be in [0, 1), got {ratio}"
    );
    let data = data_of(x, "dropout");
    assert_eq!(
        data.shape(),
        mask.shape(),
        "dezero: dropout mask shape {:?} does not match the input {:?}",
        mask.shape(),
        data.shape()
    );

    // Built from existing differentiable ops rather than a bespoke `Op`: there
    // is no new derivative to define, so a new node type would only be more
    // code to get wrong. `scale` folds into the mask so the graph stays two
    // nodes deep instead of three.
    let scaled = Variable::new(mask.mapv(|m| m / (1.0 - ratio)));
    mul(x, &scaled)
}

/// Inverted dropout — Python's `dezero.functions.dropout`.
///
/// In training mode (the default) this draws a fresh Bernoulli mask and
/// returns [`dropout_with_mask`]. Under [`test_mode`](crate::test_mode) it is
/// the **identity**, returning `x` itself with no graph node added — matching
/// the reference, which returns `x` unchanged rather than multiplying by ones.
///
/// The mask comes from the crate's own RNG, whose stream cannot match numpy's
/// (see [`rand`](crate::rand)). Any test that needs a *particular* mask must
/// supply it through [`dropout_with_mask`]; that is what the parity fixture
/// does.
///
/// # Panics
///
/// Panics if `x` holds no data, or if `ratio` is outside `[0, 1)`.
///
/// # Examples
///
/// ```
/// use dezero::{dropout, test_mode, Variable};
/// use ndarray::arr1;
///
/// let x = Variable::new(arr1(&[1.0, 2.0, 3.0]).into_dyn());
///
/// // Under test mode dropout is exactly the identity.
/// let guard = test_mode();
/// let y = dropout(&x, 0.5);
/// assert_eq!(y.data(), x.data());
/// drop(guard);
///
/// // In training mode every surviving element is scaled by 1/(1 - ratio).
/// let y = dropout(&x, 0.5);
/// for (out, inp) in y.data().expect("data").iter().zip(x.data().expect("data").iter()) {
///     assert!(*out == 0.0 || (*out - 2.0 * inp).abs() < 1e-12);
/// }
/// ```
#[must_use]
pub fn dropout(x: &Variable, ratio: f64) -> Variable {
    if !crate::core::config::is_train() {
        return x.clone();
    }
    let shape: Vec<usize> = data_of(x, "dropout").shape().to_vec();
    let mask = crate::utils::random::rand(&shape).mapv(|u| f64::from(u > ratio));
    dropout_with_mask(x, ratio, &mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functions::reduce::sum_all;
    use crate::utils::gradient_check;
    use ndarray::{arr1, arr2};

    const EPS: f64 = 1e-4;
    const RTOL: f64 = 1e-4;
    const ATOL: f64 = 1e-5;

    fn logits() -> Variable {
        Variable::new(arr2(&[[-2.0, -0.5, 0.0], [0.5, 1.0, 2.0]]).into_dyn())
    }

    fn data(v: &Variable) -> ArrayD<f64> {
        v.data().expect("variable holds data")
    }

    fn grad(v: &Variable) -> ArrayD<f64> {
        data(&v.grad().expect("variable has a gradient"))
    }

    // -- sigmoid -----------------------------------------------------------

    #[test]
    fn sigmoid_matches_the_textbook_formula() {
        let points = [-8.0_f64, -1.0, -0.25, 0.0, 0.25, 1.0, 8.0];
        let y = data(&sigmoid(&Variable::new(arr1(&points).into_dyn())));
        for (actual, point) in y.iter().zip(points.iter()) {
            let expected = 1.0 / (1.0 + (-point).exp());
            assert!(
                (actual - expected).abs() < 1e-15,
                "sigmoid({point}) was {actual}, expected {expected}"
            );
        }
    }

    /// The reason the reference uses the `tanh` spelling: `exp(-x)` for a large
    /// negative `x` overflows to `inf`, and `1/(1 + inf)` is 0 rather than the
    /// gradual underflow the stable form gives.
    #[test]
    fn sigmoid_survives_saturating_inputs() {
        let x = Variable::new(arr1(&[-800.0, 800.0]).into_dyn());
        let y = data(&sigmoid(&x));
        assert!(y.iter().all(|v| v.is_finite()));
        assert_eq!(y[[0]], 0.0);
        assert_eq!(y[[1]], 1.0);
        assert!(
            (800.0_f64).exp().is_infinite(),
            "the naive formula really would overflow"
        );
    }

    #[test]
    fn sigmoid_backward_is_y_times_one_minus_y() {
        let x = logits();
        let y = sigmoid(&x);
        let values = data(&y);
        y.backward();

        let expected = values.mapv(|v| v * (1.0 - v));
        for (actual, expected) in grad(&x).iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-15);
        }
    }

    #[test]
    fn sigmoid_gradient_matches_numerical_diff() {
        gradient_check(sigmoid, &logits(), EPS, RTOL, ATOL).expect("sigmoid");

        let weights = Variable::new(arr2(&[[1.0, -2.0, 0.5], [3.0, 0.25, -1.0]]).into_dyn());
        gradient_check(
            |x| sum_all(&(&sigmoid(x) * &weights)),
            &logits(),
            EPS,
            RTOL,
            ATOL,
        )
        .expect("weighted sigmoid");
    }

    // -- relu --------------------------------------------------------------

    #[test]
    fn relu_clamps_at_zero() {
        let x = Variable::new(arr1(&[-2.0, -0.0, 0.0, 0.5, 3.0]).into_dyn());
        assert_eq!(data(&relu(&x)), arr1(&[0.0, 0.0, 0.0, 0.5, 3.0]).into_dyn());
    }

    #[test]
    fn relu_backward_is_the_positive_mask() {
        let x = Variable::new(arr1(&[-2.0, 0.0, 0.5, 3.0]).into_dyn());
        relu(&x).backward();
        assert_eq!(
            grad(&x),
            arr1(&[0.0, 0.0, 1.0, 1.0]).into_dyn(),
            "x = 0 is on the closed side, matching Python's strict `>`"
        );
    }

    #[test]
    fn relu_gradient_matches_numerical_diff() {
        // Points chosen away from the kink: a finite difference straddling
        // x = 0 measures the average of the two one-sided slopes and would
        // "fail" against any convention.
        let x = Variable::new(arr1(&[-2.0, -0.5, 0.5, 3.0]).into_dyn());
        gradient_check(relu, &x, EPS, RTOL, ATOL).expect("relu");

        let weights = Variable::new(arr1(&[1.0, -2.0, 0.5, 3.0]).into_dyn());
        gradient_check(|x| sum_all(&(&relu(x) * &weights)), &x, EPS, RTOL, ATOL)
            .expect("weighted relu");
    }

    #[test]
    fn relu_mask_is_a_constant_not_a_graph_node() {
        // The mask must not be differentiated: it is a step function, and its
        // "derivative" would be zero everywhere and infinite at the origin.
        let x = Variable::new(arr1(&[1.0, -1.0]).into_dyn());
        relu(&x).backward_with(false, true);
        let gx = x.grad().expect("gradient");
        assert_eq!(data(&gx), arr1(&[1.0, 0.0]).into_dyn());
    }

    // -- softmax -----------------------------------------------------------

    #[test]
    fn softmax_rows_are_probability_distributions() {
        let y = data(&softmax(&logits()));
        assert_eq!(y.shape(), &[2, 3]);
        for row in 0..2 {
            let total: f64 = (0..3).map(|c| y[[row, c]]).sum();
            assert!((total - 1.0).abs() < 1e-15, "row {row} summed to {total}");
            assert!((0..3).all(|c| y[[row, c]] > 0.0));
        }
    }

    #[test]
    fn softmax_is_invariant_to_a_per_row_shift() {
        // Adding a constant to a whole row cancels out of the ratio; that this
        // holds *numerically* is what subtracting the row maximum buys.
        let shifted = Variable::new(arr2(&[[98.0, 99.5, 100.0], [-99.5, -99.0, -98.0]]).into_dyn());
        let a = data(&softmax(&logits()));
        let b = data(&softmax(&shifted));
        for (a, b) in a.iter().zip(b.iter()) {
            assert!((a - b).abs() < 1e-14, "{a} vs {b}");
        }
    }

    #[test]
    fn softmax_does_not_overflow_on_large_logits() {
        let x = Variable::new(arr2(&[[1000.0, 0.0]]).into_dyn());
        let y = data(&softmax(&x));
        assert_eq!(y[[0, 0]], 1.0);
        assert_eq!(y[[0, 1]], 0.0);
    }

    #[test]
    fn softmax_over_axis_zero_normalises_the_columns() {
        let y = data(&softmax_axis(&logits(), 0));
        for column in 0..3 {
            let total = y[[0, column]] + y[[1, column]];
            assert!((total - 1.0).abs() < 1e-15, "column {column}");
        }
    }

    #[test]
    fn a_negative_axis_counts_from_the_end() {
        assert_eq!(
            data(&softmax_axis(&logits(), -1)),
            data(&softmax(&logits()))
        );
        assert_eq!(
            data(&softmax_axis(&logits(), -2)),
            data(&softmax_axis(&logits(), 0))
        );
    }

    /// With a seed of ones the gradient of a softmax is exactly zero — every
    /// row already sums to 1, so nothing can change it. That makes the
    /// all-ones case useless as a test and is why the checks below weight the
    /// output first.
    #[test]
    fn softmax_backward_of_a_uniform_seed_vanishes() {
        let x = logits();
        softmax(&x).backward();
        for value in grad(&x).iter() {
            assert!(value.abs() < 1e-15, "expected 0, got {value}");
        }
    }

    #[test]
    fn softmax_gradient_matches_numerical_diff() {
        let weights = Variable::new(arr2(&[[1.0, -2.0, 0.5], [3.0, 0.25, -1.0]]).into_dyn());
        gradient_check(
            |x| sum_all(&(&softmax(x) * &weights)),
            &logits(),
            EPS,
            RTOL,
            ATOL,
        )
        .expect("weighted softmax over rows");

        gradient_check(
            |x| sum_all(&(&softmax_axis(x, 0) * &weights)),
            &logits(),
            EPS,
            RTOL,
            ATOL,
        )
        .expect("weighted softmax over columns");
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn softmax_rejects_an_axis_the_input_does_not_have() {
        let _ = softmax(&Variable::new(arr1(&[1.0, 2.0]).into_dyn()));
    }

    // -- higher-order ------------------------------------------------------

    /// Every backward here is `Variable` arithmetic, so an activation's
    /// gradient can be differentiated again.
    #[test]
    fn activation_gradients_stay_differentiable() {
        // sigmoid'' = y(1-y)(1-2y); at x = 0, y = 1/2 and the value is 0.
        let x = Variable::from_scalar(0.0);
        sigmoid(&x).backward_with(false, true);
        let gx = x.grad().expect("y'");
        assert!(gx.creator().is_some());

        x.cleargrad();
        gx.backward();
        assert!(data(&x.grad().expect("y''")).sum().abs() < 1e-15);

        // Away from the origin it is not zero, so the test above is not passing
        // by accident.
        let x = Variable::from_scalar(1.0);
        sigmoid(&x).backward_with(false, true);
        let gx = x.grad().expect("y'");
        x.cleargrad();
        gx.backward();
        let y = 1.0 / (1.0 + (-1.0_f64).exp());
        let expected = y * (1.0 - y) * (1.0 - 2.0 * y);
        assert!((data(&x.grad().expect("y''")).sum() - expected).abs() < 1e-12);
    }
}
