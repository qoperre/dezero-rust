//! Elementary elementwise functions.
//!
//! [`square`] is the book's very first `Function` (step 2) and [`exp`] its
//! companion (step 3). [`sin`], [`cos`] and [`tanh`] arrive with steps 27–35,
//! the phase that makes higher-order derivatives work; they are the
//! "Basic functions: sin / cos / tanh / exp / log" section of
//! `vendor/dezero-python/dezero/functions.py`.
//!
//! Two backward shapes appear here, both taken from the reference:
//!
//! * from the **input** — `sin'(x) = cos(x)`, `cos'(x) = -sin(x)`,
//!   `square'(x) = 2x`;
//! * from the **output** — `exp'(x) = y`, `tanh'(x) = 1 - y²`, which reuse the
//!   forward result instead of recomputing it.
//!
//! Every one of them is written in [`Variable`] arithmetic, so differentiating
//! a gradient produces the next derivative rather than a dead end. That is what
//! steps 33–35 rest on; see `docs/ARCHITECTURE.md`.

use ndarray::ArrayD;

use crate::core::function::{Op, apply1};
use crate::core::ops::{mul, neg, one, scalar, sub};
use crate::core::variable::Variable;

/// `y = x ** 2`, elementwise.
#[derive(Debug, Clone, Copy)]
pub struct Square;

impl Op for Square {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = one(xs, "Square", "input");
        vec![x.mapv(|v| v * v)]
    }

    fn backward(
        &self,
        inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let x = one(inputs, "Square", "input");
        let gy = one(gys, "Square", "output gradient");
        // gx = 2 * x * gy, built from Variables so it can be differentiated
        // again. The 2 stays 0-d and broadcasts, as Python's plain `2` does.
        vec![mul(&mul(&scalar(2.0), x), gy)]
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
        let x = one(xs, "Exp", "input");
        vec![x.mapv(f64::exp)]
    }

    fn backward(
        &self,
        _inputs: &[Variable],
        outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let y = one(outputs, "Exp", "output");
        let gy = one(gys, "Exp", "output gradient");
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

// ---------------------------------------------------------------------------
// Sin / Cos  (step 27, differentiable to any order from step 34)
// ---------------------------------------------------------------------------

/// `y = sin(x)`, elementwise.
#[derive(Debug, Clone, Copy)]
pub struct Sin;

impl Op for Sin {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = one(xs, "Sin", "input");
        vec![x.mapv(f64::sin)]
    }

    fn backward(
        &self,
        inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let x = one(inputs, "Sin", "input");
        let gy = one(gys, "Sin", "output gradient");
        // gx = gy * cos(x). Calling `cos` — not `x.data().mapv(f64::cos)` —
        // is what puts the derivative in the graph, so differentiating it
        // again yields -sin(x) instead of a dead constant.
        vec![mul(gy, &cos(x))]
    }
}

/// Sine of a variable, elementwise — Python's `dezero.functions.sin`.
///
/// # Examples
///
/// ```
/// use dezero::{cos, sin, Variable};
///
/// // The gradient is `cos(x)` as a graph node, not just as a number, which
/// // is what lets step 34 keep differentiating it (see the crate docs).
/// let x = Variable::from_scalar(0.7);
/// let y = sin(&x);
/// y.backward_with(false, true);
///
/// let first = x.grad().expect("dy/dx = cos(x)");
/// assert_eq!(first.data(), cos(&x).data());
/// assert!(first.creator().is_some());
/// ```
#[must_use]
pub fn sin(x: &Variable) -> Variable {
    apply1(Sin, &[x])
}

/// `y = cos(x)`, elementwise.
#[derive(Debug, Clone, Copy)]
pub struct Cos;

impl Op for Cos {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = one(xs, "Cos", "input");
        vec![x.mapv(f64::cos)]
    }

    fn backward(
        &self,
        inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let x = one(inputs, "Cos", "input");
        let gy = one(gys, "Cos", "output gradient");
        // gx = gy * -sin(x)
        vec![mul(gy, &neg(&sin(x)))]
    }
}

/// Cosine of a variable, elementwise — Python's `dezero.functions.cos`.
#[must_use]
pub fn cos(x: &Variable) -> Variable {
    apply1(Cos, &[x])
}

// ---------------------------------------------------------------------------
// Tanh  (step 35)
// ---------------------------------------------------------------------------

/// `y = tanh(x)`, elementwise.
#[derive(Debug, Clone, Copy)]
pub struct Tanh;

impl Op for Tanh {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = one(xs, "Tanh", "input");
        vec![x.mapv(f64::tanh)]
    }

    fn backward(
        &self,
        _inputs: &[Variable],
        outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let y = one(outputs, "Tanh", "output");
        let gy = one(gys, "Tanh", "output gradient");
        // gx = gy * (1 - y^2), phrased on the *output* like `functions.py`.
        // `y` is the same graph node the forward pass produced, so a second
        // backward pass walks back through this very op — that is where
        // step 35's deep derivative graph comes from.
        vec![mul(gy, &sub(&scalar(1.0), &mul(y, y)))]
    }
}

/// Hyperbolic tangent of a variable, elementwise — Python's
/// `dezero.functions.tanh`.
///
/// # Examples
///
/// ```
/// use dezero::{tanh, Variable};
///
/// // Step 35: y = tanh(x), then y' and y'' at x = 1.
/// let x = Variable::from_scalar(1.0);
/// let y = tanh(&x);
/// y.backward_with(false, true);
///
/// let gx = x.grad().expect("y'");
/// x.cleargrad();
/// gx.backward_with(false, true);
/// let gx2 = x.grad().expect("y''");
///
/// assert!((gx.data().expect("data").sum() - 0.419_974_341_614_026_1).abs() < 1e-12);
/// assert!((gx2.data().expect("data").sum() + 0.639_700_008_449_224_6).abs() < 1e-12);
/// ```
#[must_use]
pub fn tanh(x: &Variable) -> Variable {
    apply1(Tanh, &[x])
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

    // -- steps 27-35: sin / cos / tanh -------------------------------------

    /// The value of a 0-d variable's gradient.
    fn grad_scalar(v: &Variable) -> f64 {
        v.grad()
            .and_then(|g| g.data())
            .expect("gradient")
            .iter()
            .copied()
            .next()
            .expect("a 0-d array holds one element")
    }

    #[test]
    fn sin_cos_tanh_forward_match_the_standard_library() {
        let points = [-1.5_f64, -0.25, 0.0, 0.75, 2.0];
        let x = var(&points);
        for (op, expected) in [
            (sin(&x), points.map(f64::sin)),
            (cos(&x), points.map(f64::cos)),
            (tanh(&x), points.map(f64::tanh)),
        ] {
            let actual = op.data().expect("data");
            for (a, e) in actual.iter().zip(expected.iter()) {
                assert!((a - e).abs() < 1e-15, "{actual:?} vs {expected:?}");
            }
        }
    }

    #[test]
    fn sin_gradient_matches_numerical_diff() {
        gradient_check(sin, &var(&[-1.5, -0.25, 0.0, 0.75, 2.0]), EPS, RTOL, ATOL)
            .expect("sin gradient");
    }

    #[test]
    fn cos_gradient_matches_numerical_diff() {
        gradient_check(cos, &var(&[-1.5, -0.25, 0.0, 0.75, 2.0]), EPS, RTOL, ATOL)
            .expect("cos gradient");
    }

    #[test]
    fn tanh_gradient_matches_numerical_diff() {
        gradient_check(tanh, &var(&[-2.0, -0.5, 0.0, 0.5, 2.0]), EPS, RTOL, ATOL)
            .expect("tanh gradient");
    }

    #[test]
    fn sin_backward_is_cos_and_cos_backward_is_minus_sin() {
        let points = [-1.5_f64, 0.25, 1.0];

        let x = var(&points);
        sin(&x).backward();
        let gx = x.grad().and_then(|g| g.data()).expect("gradient");
        for (a, p) in gx.iter().zip(points.iter()) {
            assert!((a - p.cos()).abs() < 1e-15);
        }

        let x = var(&points);
        cos(&x).backward();
        let gx = x.grad().and_then(|g| g.data()).expect("gradient");
        for (a, p) in gx.iter().zip(points.iter()) {
            assert!((a + p.sin()).abs() < 1e-15);
        }
    }

    /// step 34: `sin -> cos -> -sin -> -cos -> sin`. Only a `backward` that
    /// goes through `apply` can produce the cycle; a raw-ndarray one would
    /// stop after the first derivative.
    #[test]
    fn step34_sin_derivatives_cycle_with_period_four() {
        let point = 0.7_f64;
        let expected = [point.cos(), -point.sin(), -point.cos(), point.sin()];

        let x = Variable::from_scalar(point);
        let y = sin(&x);
        y.backward_with(false, true);

        for (order, expected) in expected.iter().enumerate() {
            let actual = grad_scalar(&x);
            assert!(
                (actual - expected).abs() < 1e-12,
                "derivative {}: got {actual}, expected {expected}",
                order + 1
            );
            let gx = x.grad().expect("derivative");
            x.cleargrad();
            gx.backward_with(false, true);
        }
    }

    /// step 35, at the book's own point: `tanh(1)`, then `y'` and `y''`.
    #[test]
    fn step35_tanh_derivatives_match_the_book() {
        let x = Variable::from_scalar(1.0);
        let y = tanh(&x);
        assert!((y.data().expect("data").sum() - 0.761_594_155_955_764_9).abs() < 1e-15);

        y.backward_with(false, true);
        assert!((grad_scalar(&x) - 0.419_974_341_614_026_14).abs() < 1e-15);

        let gx = x.grad().expect("y'");
        x.cleargrad();
        gx.backward_with(false, true);
        assert!((grad_scalar(&x) + 0.639_700_008_449_224_6).abs() < 1e-15);
    }

    /// `Tanh::backward` reads the forward *output*, which the graph holds only
    /// through a `Weak`. A second backward pass therefore depends on something
    /// else keeping that output alive once the caller has dropped it — and the
    /// gradient graph does, because `y` is an input of its own `y * y` node.
    #[test]
    fn output_based_backward_survives_dropping_the_forward_output() {
        let x = Variable::from_scalar(1.0);
        let gx = {
            let y = tanh(&x);
            y.backward_with(false, true);
            x.grad().expect("y'")
        };

        x.cleargrad();
        gx.backward();
        assert!(
            (grad_scalar(&x) + 0.639_700_008_449_224_6).abs() < 1e-15,
            "y'' must not collapse to the zero substituted for a dead output"
        );
    }

    /// Cross-checks each second derivative against a finite difference of the
    /// *analytic* first derivative, evaluated in plain `f64`. Nothing in the
    /// comparison touches the graph, so an error in `backward` cannot cancel
    /// itself out.
    #[test]
    fn second_derivatives_match_a_finite_difference_of_the_analytic_first() {
        fn central_difference(f: impl Fn(f64) -> f64, x: f64, eps: f64) -> f64 {
            (f(x + eps) - f(x - eps)) / (2.0 * eps)
        }

        /// A name, the graph function, and its first derivative in plain `f64`.
        type Case = (&'static str, fn(&Variable) -> Variable, fn(f64) -> f64);

        let point = 0.7_f64;
        let cases: [Case; 3] = [
            ("sin", sin, f64::cos),
            ("cos", cos, |v| -v.sin()),
            ("tanh", tanh, |v| 1.0 - v.tanh() * v.tanh()),
        ];

        for (name, f, analytic_first) in cases {
            let x = Variable::from_scalar(point);
            let y = f(&x);
            y.backward_with(false, true);
            let gx = x.grad().expect("first derivative");
            x.cleargrad();
            gx.backward();

            let actual = grad_scalar(&x);
            let expected = central_difference(analytic_first, point, 1e-5);
            assert!(
                (actual - expected).abs() < 1e-6,
                "{name}'': autodiff gave {actual}, finite differences {expected}"
            );
        }
    }

    /// step 27's `my_sin`: the Maclaurin series for sine, summed until a term
    /// falls below `threshold`.
    fn taylor_sin(x: &Variable, threshold: f64) -> Variable {
        let mut total: Option<Variable> = None;
        let mut factorial = 1.0_f64; // (2i + 1)!

        for i in 0..30_u32 {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            let term = (sign / factorial) * crate::pow(x, f64::from(2 * i + 1));
            let magnitude = term
                .data()
                .expect("a term always holds data")
                .iter()
                .fold(0.0_f64, |acc, v| acc.max(v.abs()));

            total = Some(match total {
                None => term,
                Some(sum) => sum + term,
            });
            if magnitude < threshold {
                break;
            }
            factorial *= f64::from((2 * i + 2) * (2 * i + 3));
        }

        total.expect("the loop always contributes its first term")
    }

    #[test]
    fn step27_taylor_series_reproduces_sin_and_its_gradient() {
        let point = std::f64::consts::FRAC_PI_4;
        let x = Variable::from_scalar(point);
        let approximation = taylor_sin(&x, 1e-4);

        let value = approximation.data().expect("data").sum();
        assert!((value - point.sin()).abs() < 1e-4, "value was {value}");

        approximation.backward();
        let slope = grad_scalar(&x);
        assert!((slope - point.cos()).abs() < 1e-4, "slope was {slope}");
    }
}
