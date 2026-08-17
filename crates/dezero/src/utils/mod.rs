//! Helpers that support the core but are not part of the graph.
//!
//! * this module — finite-difference differentiation, ported from
//!   `vendor/dezero-python/dezero/utils.py` (`numerical_grad`,
//!   `gradient_check`) and `steps/step04.py` (`numerical_diff`);
//! * [`shape`] — the array-level shape arithmetic broadcasting is built from
//!   (`sum_to`, `reshape_sum_backward`), also from `dezero/utils.py`;
//! * [`array`](mod@array) — keepdims reductions, `logsumexp`, and the gather/scatter pair
//!   that stands in for numpy's fancy indexing (steps 41–48);
//! * [`conv`](mod@conv) — `pair`, the convolution output-size arithmetic of step 56, and
//!   the `im2col`/`col2im` adjoint pair of step 57;
//! * [`random`] — a self-contained generator for weight initialisation and
//!   dropout masks, since no Rust stream can reproduce numpy's.

pub mod array;
pub mod conv;
pub mod random;
pub mod shape;

pub use crate::utils::conv::{Pair, get_conv_outsize, get_deconv_outsize, pair};

use std::error::Error;
use std::fmt;

use ndarray::{ArrayD, IxDyn};

use crate::core::config::no_grad;
use crate::core::variable::Variable;

/// Rebuilds an array of shape `dim` from values in row-major order.
///
/// # Panics
///
/// Cannot fail: `values` is always produced by flattening an array of exactly
/// this shape.
fn reshaped(dim: &IxDyn, values: Vec<f64>) -> ArrayD<f64> {
    ArrayD::from_shape_vec(dim.clone(), values)
        .expect("the buffer was flattened from an array of this exact shape")
}

/// Sums every element of a variable's data.
///
/// # Panics
///
/// Panics if `v` holds no data, which for a function's output means the
/// function under test returned an unfilled [`Variable::empty`].
fn total(v: &Variable) -> f64 {
    v.data()
        .unwrap_or_else(|| panic!("dezero: numerical differentiation needs f(x) to produce data"))
        .sum()
}

/// Differentiates `f` at `x` by central finite differences.
///
/// This is the general form used by `dezero/utils.py::numerical_grad`: each
/// element of `x` is perturbed on its own by `±eps` and the *sum* of the
/// outputs is differenced, so the result is the gradient of `sum(f(x))` with
/// respect to `x` — the same quantity [`Variable::backward`] computes when
/// seeded with ones. For an elementwise `f` it reduces to `step04.py`'s
/// scalar `numerical_diff`.
///
/// `eps = 1e-4` is the value DeZero uses; much smaller values lose more to
/// floating-point cancellation than they gain in truncation error.
///
/// Probes run under [`no_grad`], so gradient checking never allocates a graph.
///
/// # Examples
///
/// ```
/// use dezero::{numerical_diff, square, Variable};
/// use ndarray::arr1;
///
/// let x = Variable::new(arr1(&[2.0, 3.0]).into_dyn());
/// let grad = numerical_diff(square, &x, 1e-4);
/// assert!((grad[[0]] - 4.0).abs() < 1e-6);
/// assert!((grad[[1]] - 6.0).abs() < 1e-6);
/// ```
///
/// # Panics
///
/// Panics if `x` holds no data, or if `f` returns a variable that holds none.
pub fn numerical_diff<F>(f: F, x: &Variable, eps: f64) -> ArrayD<f64>
where
    F: Fn(&Variable) -> Variable,
{
    let x_data = x
        .data()
        .unwrap_or_else(|| panic!("dezero: numerical_diff needs a variable that holds data"));
    let dim = x_data.raw_dim();
    let base: Vec<f64> = x_data.iter().copied().collect();

    let _guard = no_grad();
    let mut grad = Vec::with_capacity(base.len());
    for (index, original) in base.iter().enumerate() {
        let mut plus = base.clone();
        plus[index] = original + eps;
        let mut minus = base.clone();
        minus[index] = original - eps;

        let y_plus = total(&f(&Variable::new(reshaped(&dim, plus))));
        let y_minus = total(&f(&Variable::new(reshaped(&dim, minus))));
        grad.push((y_plus - y_minus) / (2.0 * eps));
    }

    reshaped(&dim, grad)
}

/// Why a [`gradient_check`] failed.
#[derive(Debug, Clone, PartialEq)]
pub enum GradientCheckError {
    /// The variable under test holds no data, so it cannot be perturbed.
    NoData,
    /// Backpropagation left no gradient on the variable — usually a sign that
    /// `f` did not actually use its argument, or that it was run under
    /// [`no_grad`].
    NoGradient,
    /// Backpropagation produced a gradient of the wrong shape.
    ShapeMismatch {
        /// Shape reported by finite differences (the shape of `x`).
        numerical: Vec<usize>,
        /// Shape reported by backpropagation.
        analytical: Vec<usize>,
    },
    /// The two gradients differ by more than the tolerance allows.
    ///
    /// Boxed to keep `Result<(), GradientCheckError>` cheap to return: the two
    /// gradient arrays would otherwise make every `Ok` as wide as the worst
    /// `Err`.
    ValueMismatch(Box<GradientMismatch>),
}

/// The two gradients a failed [`gradient_check`] compared.
#[derive(Debug, Clone, PartialEq)]
pub struct GradientMismatch {
    /// Gradient from finite differences.
    pub numerical: ArrayD<f64>,
    /// Gradient from backpropagation.
    pub analytical: ArrayD<f64>,
    /// The largest absolute difference between corresponding elements.
    pub max_abs_diff: f64,
}

impl fmt::Display for GradientCheckError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoData => write!(f, "gradient check: the variable holds no data"),
            Self::NoGradient => write!(
                f,
                "gradient check: backpropagation produced no gradient for the input"
            ),
            Self::ShapeMismatch {
                numerical,
                analytical,
            } => write!(
                f,
                "gradient check: numerical gradient has shape {numerical:?} but \
                 backpropagation produced {analytical:?}"
            ),
            Self::ValueMismatch(mismatch) => write!(
                f,
                "gradient check: gradients differ by up to {:e}\n numerical: {}\n analytical: {}",
                mismatch.max_abs_diff, mismatch.numerical, mismatch.analytical
            ),
        }
    }
}

impl Error for GradientCheckError {}

/// Compares backpropagation against finite differences — Python's
/// `dezero.utils.gradient_check`.
///
/// `f` is evaluated on a private copy of `x`, so the caller's variable keeps
/// whatever gradient it already had. Elements are compared with numpy's
/// `allclose` rule, `|a - b| <= atol + rtol * |b|`.
///
/// DeZero's own defaults are `eps = 1e-4`, `rtol = 1e-4`, `atol = 1e-5`;
/// finite differences are not accurate enough for tighter bounds.
///
/// # Errors
///
/// Returns [`GradientCheckError`] describing which way the check failed:
/// missing data, a missing or mis-shaped gradient, or values outside the
/// tolerance.
///
/// # Examples
///
/// ```
/// use dezero::{gradient_check, square, Variable};
/// use ndarray::arr1;
///
/// let x = Variable::new(arr1(&[0.5, 1.0, 2.0]).into_dyn());
/// gradient_check(square, &x, 1e-4, 1e-4, 1e-5).expect("square is differentiable");
/// ```
pub fn gradient_check<F>(
    f: F,
    x: &Variable,
    eps: f64,
    rtol: f64,
    atol: f64,
) -> Result<(), GradientCheckError>
where
    F: Fn(&Variable) -> Variable,
{
    let Some(data) = x.data() else {
        return Err(GradientCheckError::NoData);
    };

    let numerical = numerical_diff(&f, x, eps);

    // A private copy: gradient checking must not clobber the caller's graph.
    let probe = Variable::new(data);
    f(&probe).backward();
    let Some(analytical) = probe.grad().and_then(|g| g.data()) else {
        return Err(GradientCheckError::NoGradient);
    };

    if numerical.shape() != analytical.shape() {
        return Err(GradientCheckError::ShapeMismatch {
            numerical: numerical.shape().to_vec(),
            analytical: analytical.shape().to_vec(),
        });
    }

    let max_abs_diff = numerical
        .iter()
        .zip(analytical.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    let within_tolerance = numerical
        .iter()
        .zip(analytical.iter())
        .all(|(a, b)| (a - b).abs() <= atol + rtol * b.abs());

    if within_tolerance {
        Ok(())
    } else {
        Err(GradientCheckError::ValueMismatch(Box::new(
            GradientMismatch {
                numerical,
                analytical,
                max_abs_diff,
            },
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Variable, add, exp, mul, square};
    use ndarray::{arr0, arr1, arr2};

    #[test]
    fn scalar_numerical_diff_matches_step04() {
        // step04.py: numerical_diff(Square(), Variable(np.array(2.0))) -> 4.0
        let x = Variable::from_scalar(2.0);
        let grad = numerical_diff(square, &x, 1e-4);
        assert_eq!(grad.shape(), arr0(0.0).into_dyn().shape());
        assert!((grad.sum() - 4.0).abs() < 1e-8);
    }

    #[test]
    fn composed_numerical_diff_matches_step04() {
        // step04.py: f(x) = C(B(A(x))) at x = 0.5 -> 3.2974426293330694
        let x = Variable::from_scalar(0.5);
        let grad = numerical_diff(|x| square(&exp(&square(x))), &x, 1e-4);
        assert!((grad.sum() - 3.297_442_541_400_256).abs() < 1e-6);
    }

    #[test]
    fn numerical_diff_is_elementwise_for_arrays() {
        let x = Variable::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn());
        let grad = numerical_diff(square, &x, 1e-4);
        let expected = arr2(&[[2.0, 4.0], [6.0, 8.0]]).into_dyn();
        assert_eq!(grad.shape(), expected.shape());
        for (actual, expected) in grad.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn numerical_diff_does_not_build_a_graph() {
        let x = Variable::new(arr1(&[1.0, 2.0]).into_dyn());
        let _ = numerical_diff(square, &x, 1e-4);
        assert!(crate::enable_backprop(), "the guard is scoped");
        assert!(x.grad().is_none(), "the input is never differentiated");
    }

    #[test]
    fn gradient_check_leaves_the_caller_untouched() {
        let x = Variable::new(arr1(&[1.0, 2.0]).into_dyn());
        gradient_check(square, &x, 1e-4, 1e-4, 1e-5).expect("square");
        assert!(x.grad().is_none());
        assert!(x.creator().is_none());
    }

    #[test]
    fn gradient_check_accepts_a_multi_argument_closure() {
        let a = Variable::new(arr1(&[3.0, -1.0]).into_dyn());
        gradient_check(
            |x| add(&mul(x, &a), x),
            &Variable::new(arr1(&[1.0, 2.0]).into_dyn()),
            1e-4,
            1e-4,
            1e-5,
        )
        .expect("mul + add");
    }

    #[test]
    fn gradient_check_reports_a_missing_gradient() {
        let constant = Variable::new(arr1(&[1.0, 2.0]).into_dyn());
        let result = gradient_check(
            |_| square(&constant),
            &Variable::new(arr1(&[1.0, 2.0]).into_dyn()),
            1e-4,
            1e-4,
            1e-5,
        );
        assert_eq!(result, Err(GradientCheckError::NoGradient));
    }

    #[test]
    fn gradient_check_reports_missing_data() {
        let result = gradient_check(square, &Variable::empty(), 1e-4, 1e-4, 1e-5);
        assert_eq!(result, Err(GradientCheckError::NoData));
    }

    #[test]
    fn gradient_check_rejects_a_wrong_gradient() {
        // A deliberately broken op: forward squares, backward claims d/dx = 1.
        #[derive(Debug)]
        struct WrongSquare;

        impl crate::Op for WrongSquare {
            fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
                vec![xs[0].mapv(|v| v * v)]
            }

            fn backward(
                &self,
                _inputs: &[Variable],
                _outputs: &[Variable],
                gys: &[Variable],
            ) -> Vec<Variable> {
                vec![gys[0].clone()]
            }
        }

        let result = gradient_check(
            |x| crate::apply1(WrongSquare, &[x]),
            &Variable::new(arr1(&[3.0, 4.0]).into_dyn()),
            1e-4,
            1e-4,
            1e-5,
        );
        let Err(GradientCheckError::ValueMismatch(mismatch)) = result else {
            panic!("expected a value mismatch, got {result:?}");
        };
        assert!(mismatch.max_abs_diff > 1.0);
        // The error renders something a developer can act on.
        assert!(
            format!(
                "{}",
                GradientCheckError::ShapeMismatch {
                    numerical: vec![2],
                    analytical: vec![2, 1],
                }
            )
            .contains("[2, 1]")
        );
    }
}
