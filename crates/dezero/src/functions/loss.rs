//! Loss functions: [`mean_squared_error`] and [`softmax_cross_entropy`]
//! (steps 42 and 47).
//!
//! Port of the "loss function" section of
//! `vendor/dezero-python/dezero/functions.py`.
//!
//! Both are written as fused ops rather than as compositions, exactly as the
//! reference is. Python keeps the naive spellings alongside
//! (`mean_squared_error_simple`, `softmax_cross_entropy_simple`) to show what
//! the fusion buys; the port ships only the fused versions, since the simple
//! ones are already expressible in one line of the arithmetic that exists.
//!
//! # The gather / scatter this step really costs
//!
//! `softmax_cross_entropy`'s forward is one line of numpy —
//! `log_p[np.arange(N), t]` — and `ndarray` has no equivalent. That line and
//! its adjoint are [`gather_rows`] and [`scatter_add_rows`], written out by
//! hand in [`crate::utils::array`] and unit-tested there as an adjoint pair,
//! because getting the row/column order wrong produces a plausible number and a
//! silently wrong gradient.

use ndarray::{ArrayD, IxDyn, arr0};

use crate::core::function::{Op, apply1};
use crate::core::ops::{mul, neg, one, scalar, sub, two};
use crate::core::variable::Variable;
use crate::functions::activation::softmax;
use crate::utils::array::{gather_rows, logsumexp, scatter_add_rows};

/// The length of the leading axis — Python's `len(x)` on an array.
///
/// # Panics
///
/// Panics if `x` is 0-dimensional, which is numpy's
/// `TypeError: len() of unsized object`.
fn batch_size(x: &ArrayD<f64>, op: &str) -> usize {
    match x.shape().first() {
        Some(&n) => n,
        None => panic!("dezero: {op} needs a batched input, but got a 0-dimensional one"),
    }
}

/// `batch_size` as an `f64` divisor.
#[allow(
    clippy::cast_precision_loss,
    reason = "a batch large enough to lose precision here (2^53 rows) cannot be \
              held in memory"
)]
fn batch_divisor(n: usize) -> f64 {
    n as f64
}

// ---------------------------------------------------------------------------
// MeanSquaredError
// ---------------------------------------------------------------------------

/// `y = sum((x0 - x1)^2) / len(x0)`.
///
/// Note the divisor: Python's `len(diff)` is the length of the *leading* axis,
/// not the element count. For the usual `(batch, features)` input this is the
/// mean over samples of the *summed* squared error per sample, which is what
/// the book's training loops report.
#[derive(Debug, Clone, Copy)]
pub struct MeanSquaredError;

impl Op for MeanSquaredError {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let (x0, x1) = two(xs, "MeanSquaredError", "inputs");
        assert!(
            x0.shape() == x1.shape(),
            "dezero: MeanSquaredError needs two operands of the same shape, got {:?} and {:?}",
            x0.shape(),
            x1.shape()
        );
        let n = batch_divisor(batch_size(x0, "MeanSquaredError"));
        let total: f64 = x0
            .iter()
            .zip(x1.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        vec![arr0(total / n).into_dyn()]
    }

    fn backward(
        &self,
        inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let (x0, x1) = two(inputs, "MeanSquaredError", "inputs");
        let gy = one(gys, "MeanSquaredError", "output gradient");

        // Python: `gx0 = gy * diff * (2. / len(diff))`, `gx1 = -gx0`. `gy` is
        // 0-dimensional and broadcasts over the difference.
        let n = batch_divisor(
            x0.shape()
                .and_then(|s| s.first().copied())
                .unwrap_or_else(|| {
                    panic!("dezero: MeanSquaredError needs operands that hold batched data")
                }),
        );
        let diff = sub(x0, x1);
        let gx0 = mul(&mul(gy, &diff), &scalar(2.0 / n));
        let gx1 = neg(&gx0);
        vec![gx0, gx1]
    }
}

/// Mean squared error — Python's `dezero.functions.mean_squared_error`.
///
/// `sum((x0 - x1)^2) / len(x0)`, where `len` is the length of the leading axis.
/// The two operands must have the **same shape**: Python's version would
/// broadcast them in the forward pass and then hand back gradients shaped like
/// the broadcast difference rather than like the operands, so the port rejects
/// the case outright instead of returning a silently wrong gradient (the same
/// choice `utils::sum_to`'s target check makes).
///
/// # Examples
///
/// ```
/// use dezero::{mean_squared_error, Variable};
/// use ndarray::arr2;
///
/// let prediction = Variable::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn());
/// let target = Variable::new(arr2(&[[0.0, 2.0], [3.0, 6.0]]).into_dyn());
///
/// // Squared errors 1, 0, 0, 4; summed and divided by the two rows.
/// let loss = mean_squared_error(&prediction, &target);
/// assert_eq!(loss.data(), Some(ndarray::arr0(2.5).into_dyn()));
///
/// // d/dx0 = 2 (x0 - x1) / n
/// loss.backward();
/// assert_eq!(
///     prediction.grad().and_then(|g| g.data()),
///     Some(arr2(&[[1.0, 0.0], [0.0, -2.0]]).into_dyn())
/// );
/// ```
///
/// # Panics
///
/// Panics if the operands have different shapes, if they are 0-dimensional, or
/// if either holds no data.
#[must_use]
pub fn mean_squared_error(x0: &Variable, x1: &Variable) -> Variable {
    apply1(MeanSquaredError, &[x0, x1])
}

// ---------------------------------------------------------------------------
// SoftmaxCrossEntropy
// ---------------------------------------------------------------------------

/// `y = -mean(log(softmax(x)[i, t[i]]))`, the classification loss.
///
/// The labels are part of the **op**, not an input variable. Python passes them
/// as a second `Variable` and then quietly drops their gradient (its `backward`
/// returns one value for two inputs, and `zip` truncates). Integer class indices
/// are not differentiable, so the port says so in the type: `t` is a
/// `Vec<usize>` the op owns, and the op has exactly one input and one gradient.
#[derive(Debug, Clone)]
pub struct SoftmaxCrossEntropy {
    labels: Vec<usize>,
}

impl SoftmaxCrossEntropy {
    /// Creates the op for a batch of class indices, one per row of the logits.
    #[must_use]
    pub fn new(labels: &[usize]) -> Self {
        Self {
            labels: labels.to_vec(),
        }
    }

    /// Checks the logits against the label list and returns `(rows, classes)`.
    ///
    /// # Panics
    ///
    /// Panics if the logits are not 2-dimensional or the label count disagrees.
    fn dimensions(&self, shape: &[usize]) -> (usize, usize) {
        let [rows, classes] = shape else {
            panic!(
                "dezero: SoftmaxCrossEntropy needs 2-dimensional logits (batch, classes), \
                 got shape {shape:?}"
            );
        };
        assert!(
            self.labels.len() == *rows,
            "dezero: SoftmaxCrossEntropy got {} labels for {rows} rows of logits",
            self.labels.len()
        );
        (*rows, *classes)
    }
}

/// `log(softmax(x))` over the rows, written as `x - logsumexp(x, 1)` so that no
/// exponential is ever formed — Python's
/// `log_z = utils.logsumexp(x, axis=1); log_p = x - log_z`.
fn log_softmax_rows(x: &ArrayD<f64>) -> ArrayD<f64> {
    let mut log_p = x.clone();
    log_p -= &logsumexp(x, 1);
    log_p
}

impl Op for SoftmaxCrossEntropy {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = one(xs, "SoftmaxCrossEntropy", "input");
        let (rows, _) = self.dimensions(x.shape());

        // Python:
        //   log_p = log_p[np.arange(N), t.ravel()]
        //   y = -log_p.sum() / N
        let picked = gather_rows(&log_softmax_rows(x), &self.labels);

        vec![arr0(-picked.sum() / batch_divisor(rows)).into_dyn()]
    }

    fn backward(
        &self,
        inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let x = one(inputs, "SoftmaxCrossEntropy", "input");
        let gy = one(gys, "SoftmaxCrossEntropy", "output gradient");
        let shape = x.shape().unwrap_or_else(|| {
            panic!("dezero: SoftmaxCrossEntropy needs a variable that holds data")
        });
        let (rows, classes) = self.dimensions(&shape);

        // d/dx of the loss is `(softmax(x) - onehot(t)) / N`, the tidy closed
        // form of "scatter -1/N onto the true class, then push it back through
        // the log-sum-exp". The one-hot matrix is the scatter, and it is a
        // *constant*: class indices carry no gradient.
        let onehot = Variable::new(scatter_add_rows(
            &self.labels,
            &ArrayD::from_elem(IxDyn(&[rows]), 1.0),
            classes,
        ));
        let scaled = mul(gy, &scalar(1.0 / batch_divisor(rows)));
        vec![mul(&sub(&softmax(x), &onehot), &scaled)]
    }
}

/// Softmax followed by cross entropy — Python's
/// `dezero.functions.softmax_cross_entropy`.
///
/// `logits` is `(batch, classes)` and `labels` holds one class index per row.
/// The two halves are fused for the usual reason: `log(softmax(x))` written out
/// is `x - logsumexp(x)`, which never forms the exponentials that would
/// overflow, and the combined gradient collapses to `softmax(x) - onehot(t)`.
///
/// # Examples
///
/// ```
/// use dezero::{softmax_cross_entropy, Variable};
/// use ndarray::arr2;
///
/// // Two samples, three classes; the model is confident and right about the
/// // first and confident and wrong about the second.
/// let logits = Variable::new(arr2(&[[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]).into_dyn());
/// let loss = softmax_cross_entropy(&logits, &[0, 2]);
///
/// // The second sample contributes essentially all of the loss.
/// assert!((loss.data().expect("data").sum() - 5.0).abs() < 1e-3);
///
/// // The gradient pushes probability mass off class 0 and onto class 2.
/// loss.backward();
/// let g = logits.grad().and_then(|g| g.data()).expect("gradient");
/// assert!(g[[1, 0]] > 0.0 && g[[1, 2]] < 0.0);
/// ```
///
/// # Panics
///
/// Panics if `logits` is not 2-dimensional, if `labels` does not have one entry
/// per row, if a label is not a valid class index, or if `logits` holds no data.
#[must_use]
pub fn softmax_cross_entropy(logits: &Variable, labels: &[usize]) -> Variable {
    apply1(SoftmaxCrossEntropy::new(labels), &[logits])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::gradient_check;
    use ndarray::{arr1, arr2};

    const EPS: f64 = 1e-4;
    const RTOL: f64 = 1e-4;
    const ATOL: f64 = 1e-5;

    fn prediction() -> Variable {
        Variable::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn())
    }

    fn target() -> Variable {
        Variable::new(arr2(&[[0.5, 2.5], [2.0, 5.0]]).into_dyn())
    }

    fn logits() -> Variable {
        Variable::new(arr2(&[[0.5, -1.0, 2.0], [1.5, 0.25, -0.75], [-2.0, 0.0, 1.0]]).into_dyn())
    }

    fn data(v: &Variable) -> ArrayD<f64> {
        v.data().expect("variable holds data")
    }

    fn grad(v: &Variable) -> ArrayD<f64> {
        data(&v.grad().expect("variable has a gradient"))
    }

    fn scalar_value(v: &Variable) -> f64 {
        let d = data(v);
        assert_eq!(d.len(), 1, "expected a 0-dimensional loss");
        d.sum()
    }

    // -- mean_squared_error ------------------------------------------------

    #[test]
    fn mse_divides_by_the_batch_not_the_element_count() {
        // Differences 0.5, -0.5, 1.0, -1.0 -> squares 0.25, 0.25, 1, 1 -> 2.5,
        // over *two rows*, not four elements.
        let loss = mean_squared_error(&prediction(), &target());
        assert!((scalar_value(&loss) - 1.25).abs() < 1e-15);
        assert_eq!(loss.shape(), Some(vec![]), "the loss is a scalar");
    }

    #[test]
    fn mse_is_zero_exactly_when_the_operands_agree() {
        let x = prediction();
        assert_eq!(scalar_value(&mean_squared_error(&x, &x)), 0.0);
    }

    #[test]
    fn mse_is_symmetric_in_value_and_antisymmetric_in_gradient() {
        let (a, b) = (prediction(), target());
        let forward = mean_squared_error(&a, &b);
        forward.backward();
        let (ga, gb) = (grad(&a), grad(&b));
        assert_eq!(ga, -gb);

        let (c, d) = (prediction(), target());
        let reversed = mean_squared_error(&d, &c);
        assert_eq!(data(&forward), data(&reversed));
    }

    #[test]
    fn mse_backward_is_two_over_n_times_the_difference() {
        let (a, b) = (prediction(), target());
        mean_squared_error(&a, &b).backward();
        // 2/2 * (x0 - x1)
        assert_eq!(grad(&a), arr2(&[[0.5, -0.5], [1.0, -1.0]]).into_dyn());
    }

    #[test]
    fn mse_gradient_matches_numerical_diff() {
        gradient_check(
            |x| mean_squared_error(x, &target()),
            &prediction(),
            EPS,
            RTOL,
            ATOL,
        )
        .expect("d/dx0");
        gradient_check(
            |x| mean_squared_error(&prediction(), x),
            &target(),
            EPS,
            RTOL,
            ATOL,
        )
        .expect("d/dx1");
    }

    #[test]
    fn mse_works_on_a_one_dimensional_batch() {
        let a = Variable::new(arr1(&[1.0, 2.0, 3.0]).into_dyn());
        let b = Variable::new(arr1(&[0.0, 2.0, 5.0]).into_dyn());
        // (1 + 0 + 4) / 3
        assert!((scalar_value(&mean_squared_error(&a, &b)) - 5.0 / 3.0).abs() < 1e-15);
        gradient_check(|x| mean_squared_error(x, &b), &a, EPS, RTOL, ATOL).expect("1-d");
    }

    #[test]
    #[should_panic(expected = "two operands of the same shape")]
    fn mse_rejects_mismatched_shapes() {
        let row = Variable::new(arr1(&[1.0, 2.0]).into_dyn());
        let _ = mean_squared_error(&prediction(), &row);
    }

    #[test]
    #[should_panic(expected = "0-dimensional")]
    fn mse_rejects_scalars() {
        let _ = mean_squared_error(&Variable::from_scalar(1.0), &Variable::from_scalar(2.0));
    }

    // -- softmax_cross_entropy ---------------------------------------------

    /// The fused loss must equal the naive `-mean(log(softmax(x)[i, t[i]]))`
    /// built out of the ops that already exist. Nothing in the two paths is
    /// shared, so agreeing to 15 digits is a real cross-check.
    #[test]
    fn cross_entropy_matches_the_naive_composition() {
        let x = logits();
        let labels = [2_usize, 0, 1];

        let fused = scalar_value(&softmax_cross_entropy(&x, &labels));

        let probabilities = data(&crate::softmax(&x));
        let naive = -labels
            .iter()
            .enumerate()
            .map(|(row, &label)| probabilities[[row, label]].ln())
            .sum::<f64>()
            / 3.0;

        assert!(
            (fused - naive).abs() < 1e-14,
            "fused {fused} vs naive {naive}"
        );
    }

    #[test]
    fn a_confident_correct_prediction_costs_almost_nothing() {
        let x = Variable::new(arr2(&[[20.0, 0.0, 0.0]]).into_dyn());
        assert!(scalar_value(&softmax_cross_entropy(&x, &[0])) < 1e-8);
        // ... and a confident wrong one costs about the logit gap.
        let wrong = scalar_value(&softmax_cross_entropy(&x, &[1]));
        assert!((wrong - 20.0).abs() < 1e-6, "{wrong}");
    }

    #[test]
    fn uniform_logits_cost_the_log_of_the_class_count() {
        let x = Variable::new(ArrayD::zeros(IxDyn(&[4, 5])));
        let loss = scalar_value(&softmax_cross_entropy(&x, &[0, 1, 2, 3]));
        assert!((loss - 5.0_f64.ln()).abs() < 1e-14, "{loss}");
    }

    #[test]
    fn cross_entropy_backward_is_softmax_minus_one_hot_over_n() {
        let x = logits();
        let labels = [2_usize, 0, 1];
        softmax_cross_entropy(&x, &labels).backward();

        let expected = data(&crate::softmax(&logits()));
        let gx = grad(&x);
        for row in 0..3 {
            for column in 0..3 {
                let onehot = f64::from(labels[row] == column);
                let want = (expected[[row, column]] - onehot) / 3.0;
                assert!(
                    (gx[[row, column]] - want).abs() < 1e-15,
                    "gradient at ({row}, {column})"
                );
            }
        }
    }

    /// Every row of the gradient sums to zero: shifting all of a row's logits
    /// by a constant leaves the loss unchanged, so the loss cannot have a
    /// gradient along that direction.
    #[test]
    fn cross_entropy_gradient_rows_sum_to_zero() {
        let x = logits();
        softmax_cross_entropy(&x, &[0, 1, 2]).backward();
        let gx = grad(&x);
        for row in 0..3 {
            let total: f64 = (0..3).map(|c| gx[[row, c]]).sum();
            assert!(total.abs() < 1e-15, "row {row} summed to {total}");
        }
    }

    #[test]
    fn cross_entropy_gradient_matches_numerical_diff() {
        for labels in [[0_usize, 1, 2], [2, 2, 2], [1, 0, 0]] {
            gradient_check(
                |x| softmax_cross_entropy(x, &labels),
                &logits(),
                EPS,
                RTOL,
                ATOL,
            )
            .unwrap_or_else(|e| panic!("labels {labels:?}: {e}"));
        }
    }

    /// The overflow case the fusion exists for: `exp(800)` is `inf`, so a
    /// `log(softmax(x))` built as two separate steps would give `nan` here.
    #[test]
    fn cross_entropy_survives_logits_that_would_overflow() {
        let x = Variable::new(arr2(&[[800.0, 0.0], [0.0, 800.0]]).into_dyn());
        let loss = softmax_cross_entropy(&x, &[0, 1]);
        assert!(scalar_value(&loss).is_finite());
        assert!(scalar_value(&loss) < 1e-8);

        loss.backward();
        assert!(grad(&x).iter().all(|v| v.is_finite()));
    }

    #[test]
    #[should_panic(expected = "2-dimensional logits")]
    fn cross_entropy_rejects_non_matrix_logits() {
        let _ = softmax_cross_entropy(&Variable::new(arr1(&[1.0, 2.0]).into_dyn()), &[0]);
    }

    #[test]
    #[should_panic(expected = "2 labels for 3 rows")]
    fn cross_entropy_rejects_a_label_count_mismatch() {
        let _ = softmax_cross_entropy(&logits(), &[0, 1]);
    }

    #[test]
    #[should_panic(expected = "out of bounds for 3 columns")]
    fn cross_entropy_rejects_an_out_of_range_label() {
        let _ = softmax_cross_entropy(&logits(), &[0, 1, 3]);
    }
}
