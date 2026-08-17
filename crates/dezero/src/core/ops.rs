//! Arithmetic: `+`, `-`, `*`, `/`, unary `-` and `**`.
//!
//! Port of the operator section of `vendor/dezero-python/dezero/core.py`
//! (steps 20–22). Each operation appears three times, by design:
//!
//! 1. a unit struct implementing [`Op`] — the mathematics;
//! 2. a free function ([`add`], [`mul`], ...) — the callable Python exposes;
//! 3. an `impl` of the matching `std::ops` trait — the sugar.
//!
//! Every `backward` here is written with [`Variable`] arithmetic rather than
//! `ndarray` arithmetic, so the backward pass builds graph nodes of its own and
//! can itself be differentiated (see `docs/ARCHITECTURE.md`).
//!
//! # Scalars
//!
//! Python relies on numpy broadcasting: `x * 2.0` wraps `2.0` in a
//! 0-dimensional array and lets numpy stretch it, and `Mul.backward` folds the
//! resulting gradient back with `sum_to`. `sum_to` arrives in step 40, so
//! until then a scalar operand is materialised at the other operand's shape.
//! The numbers are identical; only the memory use differs, and the
//! `TODO(step-40)` markers below say where the broadcasting version plugs in.
//!
//! # Mismatched shapes
//!
//! Two array operands of *different* shapes are rejected outright with a
//! panic, rather than being broadcast with a silently wrong backward pass.
//! Broadcasting support is step 40.
//!
//! Python's `__array_priority__ = 200` has no counterpart: it exists only to
//! beat `numpy.ndarray.__mul__` at dispatch, and the port never overloads
//! operators on `ArrayD`.

use ndarray::ArrayD;

use crate::core::function::{Op, apply1};
use crate::core::variable::Variable;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Destructures a slice expected to hold exactly one element.
///
/// `pub(crate)` so the unary ops in [`crate::functions`] share the wording of
/// the arity panics rather than re-spelling it once per op.
pub(crate) fn one<'a, T>(items: &'a [T], op: &str, role: &str) -> &'a T {
    let [item] = items else {
        panic!("dezero: {op} expects exactly 1 {role}, got {}", items.len());
    };
    item
}

/// Destructures a slice expected to hold exactly two elements.
fn two<'a, T>(items: &'a [T], op: &str, role: &str) -> (&'a T, &'a T) {
    let [first, second] = items else {
        panic!("dezero: {op} expects exactly 2 {role}, got {}", items.len());
    };
    (first, second)
}

/// Rejects operands that would need broadcasting.
///
/// # Panics
///
/// Panics if the shapes differ. `ndarray` would happily stretch the right
/// operand onto the left one's shape, but the matching backward pass needs
/// `sum_to` to fold the gradient back again — that is step 40. Failing loudly
/// beats returning a wrong gradient.
fn require_same_shape(x0: &ArrayD<f64>, x1: &ArrayD<f64>, op: &str) {
    assert!(
        x0.shape() == x1.shape(),
        "dezero: {op} got operands of shapes {:?} and {:?}; broadcasting between \
         differently shaped operands is not implemented yet (step 40)",
        x0.shape(),
        x1.shape()
    );
}

/// A detached variable filled with `value`, shaped like `like`.
///
/// # Panics
///
/// Panics if `like` holds no data, since there is no shape to copy.
pub(crate) fn constant_like(value: f64, like: &Variable) -> Variable {
    like.full_like(value).unwrap_or_else(|| {
        panic!("dezero: cannot build the constant {value} against a variable that holds no data")
    })
}

// ---------------------------------------------------------------------------
// Add
// ---------------------------------------------------------------------------

/// `y = x0 + x1`.
#[derive(Debug, Clone, Copy)]
pub struct Add;

impl Op for Add {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let (x0, x1) = two(xs, "Add", "inputs");
        require_same_shape(x0, x1, "Add");
        vec![*x0 + *x1]
    }

    fn backward(
        &self,
        _inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let gy = one(gys, "Add", "output gradients");
        // TODO(step-40): when the input shapes differ, Python folds each
        // gradient back with `dezero.functions.sum_to(gx, x.shape)`. Until
        // `sum_to` exists, `forward` refuses mismatched shapes outright.
        vec![gy.clone(), gy.clone()]
    }
}

/// Adds two variables elementwise — Python's `dezero.core.add`.
#[must_use]
pub fn add(x0: &Variable, x1: &Variable) -> Variable {
    apply1(Add, &[x0, x1])
}

// ---------------------------------------------------------------------------
// Sub
// ---------------------------------------------------------------------------

/// `y = x0 - x1`.
#[derive(Debug, Clone, Copy)]
pub struct Sub;

impl Op for Sub {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let (x0, x1) = two(xs, "Sub", "inputs");
        require_same_shape(x0, x1, "Sub");
        vec![*x0 - *x1]
    }

    fn backward(
        &self,
        _inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let gy = one(gys, "Sub", "output gradients");
        // TODO(step-40): `sum_to` each gradient when the input shapes differ.
        vec![gy.clone(), neg(gy)]
    }
}

/// Subtracts `x1` from `x0` elementwise — Python's `dezero.core.sub`.
#[must_use]
pub fn sub(x0: &Variable, x1: &Variable) -> Variable {
    apply1(Sub, &[x0, x1])
}

// ---------------------------------------------------------------------------
// Mul
// ---------------------------------------------------------------------------

/// `y = x0 * x1`, elementwise.
#[derive(Debug, Clone, Copy)]
pub struct Mul;

impl Op for Mul {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let (x0, x1) = two(xs, "Mul", "inputs");
        require_same_shape(x0, x1, "Mul");
        vec![*x0 * *x1]
    }

    fn backward(
        &self,
        inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let (x0, x1) = two(inputs, "Mul", "inputs");
        let gy = one(gys, "Mul", "output gradients");
        // TODO(step-40): `sum_to` each gradient when the input shapes differ.
        vec![mul(gy, x1), mul(gy, x0)]
    }
}

/// Multiplies two variables elementwise — Python's `dezero.core.mul`.
#[must_use]
pub fn mul(x0: &Variable, x1: &Variable) -> Variable {
    apply1(Mul, &[x0, x1])
}

// ---------------------------------------------------------------------------
// Div
// ---------------------------------------------------------------------------

/// `y = x0 / x1`, elementwise.
#[derive(Debug, Clone, Copy)]
pub struct Div;

impl Op for Div {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let (x0, x1) = two(xs, "Div", "inputs");
        require_same_shape(x0, x1, "Div");
        vec![*x0 / *x1]
    }

    fn backward(
        &self,
        inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let (x0, x1) = two(inputs, "Div", "inputs");
        let gy = one(gys, "Div", "output gradients");
        // gx0 = gy / x1 ; gx1 = gy * (-x0 / x1^2)
        let gx0 = div(gy, x1);
        let gx1 = mul(gy, &neg(&div(x0, &pow(x1, 2.0))));
        // TODO(step-40): `sum_to` each gradient when the input shapes differ.
        vec![gx0, gx1]
    }
}

/// Divides `x0` by `x1` elementwise — Python's `dezero.core.div`.
#[must_use]
pub fn div(x0: &Variable, x1: &Variable) -> Variable {
    apply1(Div, &[x0, x1])
}

// ---------------------------------------------------------------------------
// Neg
// ---------------------------------------------------------------------------

/// `y = -x`.
#[derive(Debug, Clone, Copy)]
pub struct Neg;

impl Op for Neg {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = one(xs, "Neg", "inputs");
        vec![-*x]
    }

    fn backward(
        &self,
        _inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let gy = one(gys, "Neg", "output gradients");
        vec![neg(gy)]
    }
}

/// Negates a variable — Python's `dezero.core.neg`.
#[must_use]
pub fn neg(x: &Variable) -> Variable {
    apply1(Neg, &[x])
}

// ---------------------------------------------------------------------------
// Pow
// ---------------------------------------------------------------------------

/// `y = x ** c` for a constant exponent `c`.
///
/// The exponent is part of the op, not an input, so no gradient flows to it —
/// matching Python, where `Pow.__init__` stores `c`.
#[derive(Debug, Clone, Copy)]
pub struct Pow {
    exponent: f64,
}

impl Pow {
    /// Creates the op for exponent `c`.
    #[must_use]
    pub fn new(exponent: f64) -> Self {
        Self { exponent }
    }
}

impl Op for Pow {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = one(xs, "Pow", "inputs");
        let c = self.exponent;
        vec![x.mapv(|v| v.powf(c))]
    }

    fn backward(
        &self,
        inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let x = one(inputs, "Pow", "inputs");
        let gy = one(gys, "Pow", "output gradients");
        let c = self.exponent;
        // gx = c * x^(c - 1) * gy
        let coefficient = constant_like(c, x);
        vec![mul(&mul(&coefficient, &pow(x, c - 1.0)), gy)]
    }
}

/// Raises `x` to the power `c` elementwise — Python's `dezero.core.pow`.
///
/// Rust has no `**` operator, so unlike the other five operations this one has
/// no sugar; call it by name.
#[must_use]
pub fn pow(x: &Variable, c: f64) -> Variable {
    apply1(Pow::new(c), &[x])
}

// ---------------------------------------------------------------------------
// Operator sugar
// ---------------------------------------------------------------------------

/// Generates the eight `std::ops` impls for one binary operation: every
/// combination of owned/borrowed [`Variable`] operands, plus `f64` on either
/// side.
///
/// `impl Trait<Variable> for f64` is allowed under the orphan rule because a
/// local type appears among the trait's parameters.
macro_rules! impl_binary_operator {
    ($trait_name:ident, $method:ident, $func:path) => {
        impl std::ops::$trait_name<&Variable> for &Variable {
            type Output = Variable;
            fn $method(self, rhs: &Variable) -> Variable {
                $func(self, rhs)
            }
        }

        impl std::ops::$trait_name<Variable> for &Variable {
            type Output = Variable;
            fn $method(self, rhs: Variable) -> Variable {
                $func(self, &rhs)
            }
        }

        impl std::ops::$trait_name<&Variable> for Variable {
            type Output = Variable;
            fn $method(self, rhs: &Variable) -> Variable {
                $func(&self, rhs)
            }
        }

        impl std::ops::$trait_name<Variable> for Variable {
            type Output = Variable;
            fn $method(self, rhs: Variable) -> Variable {
                $func(&self, &rhs)
            }
        }

        impl std::ops::$trait_name<f64> for &Variable {
            type Output = Variable;
            fn $method(self, rhs: f64) -> Variable {
                $func(self, &constant_like(rhs, self))
            }
        }

        impl std::ops::$trait_name<f64> for Variable {
            type Output = Variable;
            fn $method(self, rhs: f64) -> Variable {
                $func(&self, &constant_like(rhs, &self))
            }
        }

        impl std::ops::$trait_name<&Variable> for f64 {
            type Output = Variable;
            fn $method(self, rhs: &Variable) -> Variable {
                $func(&constant_like(self, rhs), rhs)
            }
        }

        impl std::ops::$trait_name<Variable> for f64 {
            type Output = Variable;
            fn $method(self, rhs: Variable) -> Variable {
                $func(&constant_like(self, &rhs), &rhs)
            }
        }
    };
}

impl_binary_operator!(Add, add, crate::core::ops::add);
impl_binary_operator!(Sub, sub, crate::core::ops::sub);
impl_binary_operator!(Mul, mul, crate::core::ops::mul);
impl_binary_operator!(Div, div, crate::core::ops::div);

impl std::ops::Neg for &Variable {
    type Output = Variable;

    fn neg(self) -> Variable {
        crate::core::ops::neg(self)
    }
}

impl std::ops::Neg for Variable {
    type Output = Variable;

    fn neg(self) -> Variable {
        crate::core::ops::neg(&self)
    }
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

    fn data(v: &Variable) -> ArrayD<f64> {
        v.data().expect("variable holds data")
    }

    fn grad(v: &Variable) -> ArrayD<f64> {
        data(&v.grad().expect("variable has a gradient"))
    }

    // -- forward values ----------------------------------------------------

    #[test]
    fn forward_values_match_elementwise_arithmetic() {
        let a = var(&[1.0, 2.0, 3.0]);
        let b = var(&[4.0, 5.0, 6.0]);

        assert_eq!(data(&add(&a, &b)), arr1(&[5.0, 7.0, 9.0]).into_dyn());
        assert_eq!(data(&sub(&a, &b)), arr1(&[-3.0, -3.0, -3.0]).into_dyn());
        assert_eq!(data(&mul(&a, &b)), arr1(&[4.0, 10.0, 18.0]).into_dyn());
        assert_eq!(data(&div(&b, &a)), arr1(&[4.0, 2.5, 2.0]).into_dyn());
        assert_eq!(data(&neg(&a)), arr1(&[-1.0, -2.0, -3.0]).into_dyn());
        assert_eq!(data(&pow(&a, 3.0)), arr1(&[1.0, 8.0, 27.0]).into_dyn());
    }

    #[test]
    fn operators_match_the_free_functions() {
        let a = var(&[1.0, 2.0, 3.0]);
        let b = var(&[4.0, 5.0, 6.0]);

        assert_eq!(data(&(&a + &b)), data(&add(&a, &b)));
        assert_eq!(data(&(a.clone() + b.clone())), data(&add(&a, &b)));
        assert_eq!(data(&(&a - &b)), data(&sub(&a, &b)));
        assert_eq!(data(&(&a * &b)), data(&mul(&a, &b)));
        assert_eq!(data(&(&a / &b)), data(&div(&a, &b)));
        assert_eq!(data(&(-&a)), data(&neg(&a)));
        assert_eq!(data(&(-a.clone())), data(&neg(&a)));
    }

    #[test]
    fn scalar_operands_work_on_both_sides() {
        let x = var(&[1.0, 2.0, 4.0]);

        assert_eq!(data(&(&x + 1.0)), arr1(&[2.0, 3.0, 5.0]).into_dyn());
        assert_eq!(data(&(1.0 + &x)), arr1(&[2.0, 3.0, 5.0]).into_dyn());
        assert_eq!(data(&(&x - 1.0)), arr1(&[0.0, 1.0, 3.0]).into_dyn());
        assert_eq!(data(&(1.0 - &x)), arr1(&[0.0, -1.0, -3.0]).into_dyn());
        assert_eq!(data(&(&x * 3.0)), arr1(&[3.0, 6.0, 12.0]).into_dyn());
        assert_eq!(data(&(3.0 * &x)), arr1(&[3.0, 6.0, 12.0]).into_dyn());
        assert_eq!(data(&(&x / 2.0)), arr1(&[0.5, 1.0, 2.0]).into_dyn());
        assert_eq!(data(&(4.0 / &x)), arr1(&[4.0, 2.0, 1.0]).into_dyn());

        // Owned-operand overloads.
        assert_eq!(data(&(x.clone() * 2.0)), arr1(&[2.0, 4.0, 8.0]).into_dyn());
        assert_eq!(data(&(2.0 * x.clone())), arr1(&[2.0, 4.0, 8.0]).into_dyn());
    }

    #[test]
    fn scalar_operands_are_differentiable() {
        // step-21: `3 * x + 1` must backpropagate like any other graph.
        let x = Variable::from_scalar(2.0);
        let y = 3.0 * &x + 1.0;
        y.backward();
        assert_eq!(data(&y), ndarray::arr0(7.0).into_dyn());
        assert_eq!(grad(&x), ndarray::arr0(3.0).into_dyn());
    }

    // -- gradient checks ---------------------------------------------------

    #[test]
    fn add_gradient_matches_numerical_diff() {
        let other = var(&[0.5, -1.5, 2.5]);
        gradient_check(|x| add(x, &other), &var(&[1.0, 2.0, 3.0]), EPS, RTOL, ATOL)
            .expect("add gradient");
    }

    #[test]
    fn sub_gradient_matches_numerical_diff() {
        let other = var(&[0.5, -1.5, 2.5]);
        gradient_check(|x| sub(x, &other), &var(&[1.0, 2.0, 3.0]), EPS, RTOL, ATOL)
            .expect("sub gradient (left operand)");
        gradient_check(|x| sub(&other, x), &var(&[1.0, 2.0, 3.0]), EPS, RTOL, ATOL)
            .expect("sub gradient (right operand)");
    }

    #[test]
    fn mul_gradient_matches_numerical_diff() {
        let other = var(&[0.5, -1.5, 2.5]);
        gradient_check(|x| mul(x, &other), &var(&[1.0, 2.0, 3.0]), EPS, RTOL, ATOL)
            .expect("mul gradient");
    }

    #[test]
    fn div_gradient_matches_numerical_diff() {
        let other = var(&[0.5, -1.5, 2.5]);
        gradient_check(|x| div(x, &other), &var(&[1.0, 2.0, 3.0]), EPS, RTOL, ATOL)
            .expect("div gradient (numerator)");
        gradient_check(|x| div(&other, x), &var(&[1.0, 2.0, 3.0]), EPS, RTOL, ATOL)
            .expect("div gradient (denominator)");
    }

    #[test]
    fn neg_gradient_matches_numerical_diff() {
        gradient_check(neg, &var(&[1.0, -2.0, 3.0]), EPS, RTOL, ATOL).expect("neg gradient");
    }

    #[test]
    fn pow_gradient_matches_numerical_diff() {
        gradient_check(|x| pow(x, 3.0), &var(&[1.0, 2.0, 3.0]), EPS, RTOL, ATOL)
            .expect("pow(3) gradient");
        gradient_check(|x| pow(x, 0.5), &var(&[0.25, 1.0, 4.0]), EPS, RTOL, ATOL)
            .expect("pow(0.5) gradient");
    }

    #[test]
    fn compound_expression_gradient_matches_numerical_diff() {
        let a = var(&[2.0, 3.0, 4.0]);
        let b = var(&[5.0, 6.0, 7.0]);
        gradient_check(
            |x| div(&add(&mul(x, &a), &pow(x, 2.0)), &b) - 1.0,
            &var(&[1.0, 2.0, 3.0]),
            EPS,
            RTOL,
            ATOL,
        )
        .expect("compound gradient");
    }

    #[test]
    fn two_dimensional_gradient_matches_numerical_diff() {
        let other = Variable::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn());
        let x = Variable::new(arr2(&[[0.5, 1.5], [2.5, 3.5]]).into_dyn());
        gradient_check(|x| mul(x, &other), &x, EPS, RTOL, ATOL).expect("2-d mul gradient");
    }

    // -- explicit analytic gradients ---------------------------------------

    #[test]
    fn mul_gradients_are_the_opposite_operands() {
        let a = var(&[2.0, 3.0]);
        let b = var(&[5.0, 7.0]);
        let y = mul(&a, &b);
        y.backward();
        assert_eq!(grad(&a), arr1(&[5.0, 7.0]).into_dyn());
        assert_eq!(grad(&b), arr1(&[2.0, 3.0]).into_dyn());
    }

    #[test]
    fn sub_gradient_is_signed() {
        let a = var(&[2.0, 3.0]);
        let b = var(&[5.0, 7.0]);
        let y = sub(&a, &b);
        y.backward();
        assert_eq!(grad(&a), arr1(&[1.0, 1.0]).into_dyn());
        assert_eq!(grad(&b), arr1(&[-1.0, -1.0]).into_dyn());
    }

    // -- out-of-scope shapes fail loudly -----------------------------------

    #[test]
    #[should_panic(expected = "broadcasting between differently shaped operands")]
    fn mismatched_shapes_are_rejected_until_step_40() {
        let a = var(&[1.0, 2.0, 3.0]);
        let b = Variable::from_scalar(2.0);
        let _ = add(&a, &b);
    }

    #[test]
    #[should_panic(expected = "expects exactly 2 inputs")]
    fn wrong_arity_is_rejected() {
        let a = var(&[1.0, 2.0, 3.0]);
        let _ = apply1(Add, &[&a]);
    }
}
