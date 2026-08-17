//! Elementary elementwise functions.
//!
//! [`square`] is the book's very first `Function` (step 2) and [`exp`] its
//! companion (step 3). `exp` is ported from
//! `vendor/dezero-python/dezero/functions.py`, which computes its backward
//! from the *output* (`gx = gy * y`) rather than recomputing `exp(x)`.

use ndarray::ArrayD;

use crate::core::function::{Op, apply1};
use crate::core::ops::{constant_like, mul};
use crate::core::variable::Variable;

/// `y = x ** 2`, elementwise.
#[derive(Debug, Clone, Copy)]
pub struct Square;

impl Op for Square {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let [x] = xs else {
            panic!("dezero: Square expects exactly 1 input, got {}", xs.len());
        };
        vec![x.mapv(|v| v * v)]
    }

    fn backward(
        &self,
        inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let [x] = inputs else {
            panic!(
                "dezero: Square expects exactly 1 input, got {}",
                inputs.len()
            );
        };
        let [gy] = gys else {
            panic!(
                "dezero: Square expects exactly 1 output gradient, got {}",
                gys.len()
            );
        };
        // gx = 2 * x * gy, built from Variables so it can be differentiated
        // again.
        vec![mul(&mul(&constant_like(2.0, x), x), gy)]
    }
}

/// Squares a variable elementwise — Python's `dezero.functions.square`.
///
/// # Examples
///
/// ```
/// use dezero::{square, Variable};
/// use ndarray::arr1;
///
/// let x = Variable::new(arr1(&[1.0, 2.0, 3.0]).into_dyn());
/// let y = square(&x);
/// assert_eq!(y.data(), Some(arr1(&[1.0, 4.0, 9.0]).into_dyn()));
///
/// y.backward();
/// assert_eq!(
///     x.grad().and_then(|g| g.data()),
///     Some(arr1(&[2.0, 4.0, 6.0]).into_dyn())
/// );
/// ```
#[must_use]
pub fn square(x: &Variable) -> Variable {
    apply1(Square, &[x])
}

/// `y = e ** x`, elementwise.
#[derive(Debug, Clone, Copy)]
pub struct Exp;

impl Op for Exp {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let [x] = xs else {
            panic!("dezero: Exp expects exactly 1 input, got {}", xs.len());
        };
        vec![x.mapv(f64::exp)]
    }

    fn backward(
        &self,
        _inputs: &[Variable],
        outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let [y] = outputs else {
            panic!(
                "dezero: Exp expects exactly 1 output, got {}",
                outputs.len()
            );
        };
        let [gy] = gys else {
            panic!(
                "dezero: Exp expects exactly 1 output gradient, got {}",
                gys.len()
            );
        };
        // gx = gy * y: reuse the forward result instead of recomputing exp(x),
        // exactly as `dezero/functions.py` does.
        vec![mul(gy, y)]
    }
}

/// Exponentiates a variable elementwise — Python's `dezero.functions.exp`.
#[must_use]
pub fn exp(x: &Variable) -> Variable {
    apply1(Exp, &[x])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::gradient_check;
    use ndarray::{arr1, arr2};

    const EPS: f64 = 1e-4;
    const RTOL: f64 = 1e-4;
    const ATOL: f64 = 1e-5;

    fn var(values: &[f64]) -> Variable {
        Variable::new(arr1(values).into_dyn())
    }

    #[test]
    fn square_forward() {
        let x = Variable::new(arr2(&[[1.0, -2.0], [3.0, 0.5]]).into_dyn());
        assert_eq!(
            square(&x).data(),
            Some(arr2(&[[1.0, 4.0], [9.0, 0.25]]).into_dyn())
        );
    }

    #[test]
    fn square_backward_is_two_x() {
        let x = var(&[3.0, -4.0]);
        let y = square(&x);
        y.backward();
        assert_eq!(
            x.grad().and_then(|g| g.data()),
            Some(arr1(&[6.0, -8.0]).into_dyn())
        );
    }

    #[test]
    fn square_gradient_matches_numerical_diff() {
        gradient_check(square, &var(&[0.5, 1.0, 2.0, -3.0]), EPS, RTOL, ATOL)
            .expect("square gradient");
    }

    #[test]
    fn square_second_derivative_is_two() {
        let x = Variable::from_scalar(5.0);
        let y = square(&x);
        y.backward_with(false, true);
        let gx = x.grad().expect("first derivative");
        x.cleargrad();
        gx.backward();
        assert_eq!(
            x.grad().and_then(|g| g.data()),
            Some(ndarray::arr0(2.0).into_dyn())
        );
    }

    #[test]
    fn exp_forward() {
        let x = var(&[0.0, 1.0, 2.0]);
        let y = exp(&x).data().expect("data");
        for (actual, expected) in y
            .iter()
            .zip([1.0, std::f64::consts::E, 7.389_056_098_930_65])
        {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn exp_backward_is_the_output_itself() {
        let x = var(&[0.0, 1.0, 2.0]);
        let y = exp(&x);
        let values = y.data().expect("data");
        y.backward();
        assert_eq!(x.grad().and_then(|g| g.data()), Some(values));
    }

    #[test]
    fn exp_gradient_matches_numerical_diff() {
        gradient_check(exp, &var(&[0.5, 1.0, -1.5]), EPS, RTOL, ATOL).expect("exp gradient");
    }

    #[test]
    fn exp_higher_order_derivatives_repeat() {
        // Every derivative of exp is exp itself.
        let x = Variable::from_scalar(1.0);
        let y = exp(&x);
        y.backward_with(false, true);
        let gx = x.grad().expect("first derivative");

        x.cleargrad();
        gx.backward();
        let gx2 = x.grad().expect("second derivative");

        let e = std::f64::consts::E;
        assert!((gx.data().expect("data").sum() - e).abs() < 1e-12);
        assert!((gx2.data().expect("data").sum() - e).abs() < 1e-12);
    }

    #[test]
    fn step03_composition_matches_the_book() {
        // y = square(exp(square(x))) at x = 0.5.
        let x = Variable::from_scalar(0.5);
        let y = square(&exp(&square(&x)));
        let value = y.data().expect("data").sum();
        assert!((value - 1.648_721_270_700_128_2).abs() < 1e-12);

        y.backward();
        let g = x.grad().and_then(|g| g.data()).expect("gradient").sum();
        assert!((g - 3.297_442_541_400_256).abs() < 1e-10);
    }
}
