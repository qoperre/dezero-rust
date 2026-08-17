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
//! # Broadcasting (step 40)
//!
//! Operands of different shapes broadcast by numpy's rules, and each gradient
//! is folded back onto its own operand's shape with
//! [`sum_to`] — Python's
//! `if x0.shape != x1.shape: gx0 = dezero.functions.sum_to(gx0, x0.shape)`.
//!
//! `ndarray`'s own arithmetic operators are **not** a substitute: they stretch
//! only the *right* operand onto the left one's shape, so `row + matrix`
//! panics where numpy returns a matrix. Every binary forward here therefore
//! computes the numpy broadcast shape first (see [`broadcast_shape`]) and
//! stretches *both* operands onto it.
//!
//! # Scalars
//!
//! `x * 2.0` wraps `2.0` in a 0-dimensional variable and lets broadcasting
//! stretch it, exactly as Python's `as_array(2.0)` does — the scalar is never
//! materialised at the peer operand's shape.
//!
//! Python's `__array_priority__ = 200` has no counterpart: it exists only to
//! beat `numpy.ndarray.__mul__` at dispatch, and the port never overloads
//! operators on `ArrayD`.

use ndarray::{ArrayD, IxDyn, Zip};

use crate::core::function::{Op, apply1};
use crate::core::variable::Variable;
use crate::functions::reduce::sum_to;
use crate::utils::shape::broadcast_shape;

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
///
/// `pub(crate)` for the same reason [`one`] is: the binary ops in
/// [`crate::functions`] share this wording rather than re-spelling the arity
/// panic once per op.
pub(crate) fn two<'a, T>(items: &'a [T], op: &str, role: &str) -> (&'a T, &'a T) {
    let [first, second] = items else {
        panic!("dezero: {op} expects exactly 2 {role}, got {}", items.len());
    };
    (first, second)
}

/// Applies `f` elementwise, broadcasting the operands together first.
///
/// This is where numpy's symmetric broadcasting is reproduced: `ndarray`'s
/// `&a + &b` stretches only `b` onto `a`'s shape, which is enough for
/// `matrix + row` and panics on `row + matrix`. Resolving the target shape
/// explicitly and broadcasting both sides handles either order — and rejects
/// genuinely incompatible shapes with a message that names them.
///
/// # Panics
///
/// Panics if the two shapes do not broadcast together (numpy's `ValueError`).
fn zip_map(
    x0: &ArrayD<f64>,
    x1: &ArrayD<f64>,
    op: &str,
    f: impl Fn(f64, f64) -> f64,
) -> ArrayD<f64> {
    if x0.shape() == x1.shape() {
        return Zip::from(x0).and(x1).map_collect(|&a, &b| f(a, b));
    }

    let shape = broadcast_shape(x0.shape(), x1.shape()).unwrap_or_else(|| {
        panic!(
            "dezero: {op} got operands of shapes {:?} and {:?}, which do not broadcast together",
            x0.shape(),
            x1.shape()
        )
    });
    let dim = IxDyn(&shape);
    let (Some(a), Some(b)) = (x0.broadcast(dim.clone()), x1.broadcast(dim)) else {
        panic!(
            "dezero: internal invariant broken — {op} computed the broadcast shape {shape:?} \
             for operands of shapes {:?} and {:?}, but they do not broadcast to it",
            x0.shape(),
            x1.shape()
        );
    };
    Zip::from(a).and(b).map_collect(|&p, &q| f(p, q))
}

/// The shape of an operand, which must hold data.
///
/// # Panics
///
/// Panics if `v` holds no data. [`apply`](crate::apply) rejects empty inputs
/// before any op sees them, so this states an invariant.
fn shape_of(v: &Variable, op: &str) -> Vec<usize> {
    v.shape()
        .unwrap_or_else(|| panic!("dezero: {op} needs operands that hold data"))
}

/// Folds broadcast gradients back onto the shapes their operands actually had.
///
/// Python, in every one of `Add`/`Sub`/`Mul`/`Div`:
///
/// ```text
/// if x0.shape != x1.shape:  # for broadcast
///     gx0 = dezero.functions.sum_to(gx0, x0.shape)
///     gx1 = dezero.functions.sum_to(gx1, x1.shape)
/// ```
///
/// An operand that was stretched contributed to *several* output elements, so
/// its gradient is the sum over the copies — which is exactly what `sum_to`
/// computes. When the shapes already matched, `sum_to` would be the identity
/// anyway; the guard mirrors Python and keeps the graph free of no-op nodes.
fn fold_broadcast(
    op: &str,
    x0: &Variable,
    x1: &Variable,
    gx0: Variable,
    gx1: Variable,
) -> Vec<Variable> {
    let shape0 = shape_of(x0, op);
    let shape1 = shape_of(x1, op);
    if shape0 == shape1 {
        return vec![gx0, gx1];
    }
    vec![sum_to(&gx0, &shape0), sum_to(&gx1, &shape1)]
}

/// A detached 0-dimensional variable holding `value` — Python's
/// `as_array(2.0)`.
///
/// Scalars stay 0-d and reach the peer operand's shape through broadcasting,
/// so `x * 2.0` allocates one number rather than a second copy of `x`.
pub(crate) fn scalar(value: f64) -> Variable {
    Variable::from_scalar(value)
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
        vec![zip_map(x0, x1, "Add", |a, b| a + b)]
    }

    fn backward(
        &self,
        inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let (x0, x1) = two(inputs, "Add", "inputs");
        let gy = one(gys, "Add", "output gradients");
        fold_broadcast("Add", x0, x1, gy.clone(), gy.clone())
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
        vec![zip_map(x0, x1, "Sub", |a, b| a - b)]
    }

    fn backward(
        &self,
        inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let (x0, x1) = two(inputs, "Sub", "inputs");
        let gy = one(gys, "Sub", "output gradients");
        fold_broadcast("Sub", x0, x1, gy.clone(), neg(gy))
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
        vec![zip_map(x0, x1, "Mul", |a, b| a * b)]
    }

    fn backward(
        &self,
        inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let (x0, x1) = two(inputs, "Mul", "inputs");
        let gy = one(gys, "Mul", "output gradients");
        fold_broadcast("Mul", x0, x1, mul(gy, x1), mul(gy, x0))
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
        vec![zip_map(x0, x1, "Div", |a, b| a / b)]
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
        fold_broadcast("Div", x0, x1, gx0, gx1)
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
        // gx = c * x^(c - 1) * gy, with `c` a 0-d constant that broadcasts —
        // Python's plain float, wrapped by `as_array` at the operator boundary.
        vec![mul(&mul(&scalar(c), &pow(x, c - 1.0)), gy)]
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
                $func(self, &scalar(rhs))
            }
        }

        impl std::ops::$trait_name<f64> for Variable {
            type Output = Variable;
            fn $method(self, rhs: f64) -> Variable {
                $func(&self, &scalar(rhs))
            }
        }

        impl std::ops::$trait_name<&Variable> for f64 {
            type Output = Variable;
            fn $method(self, rhs: &Variable) -> Variable {
                $func(&scalar(self), rhs)
            }
        }

        impl std::ops::$trait_name<Variable> for f64 {
            type Output = Variable;
            fn $method(self, rhs: Variable) -> Variable {
                $func(&scalar(self), &rhs)
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

    // -- step 40: broadcasting ---------------------------------------------

    #[test]
    fn operands_of_different_shapes_broadcast() {
        let matrix = Variable::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn());
        let row = var(&[10.0, 20.0, 30.0]);
        let column = Variable::new(arr2(&[[100.0], [200.0]]).into_dyn());
        let scalar = Variable::from_scalar(2.0);

        assert_eq!(
            data(&add(&matrix, &row)),
            arr2(&[[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]]).into_dyn()
        );
        assert_eq!(
            data(&add(&matrix, &column)),
            arr2(&[[101.0, 102.0, 103.0], [204.0, 205.0, 206.0]]).into_dyn()
        );
        assert_eq!(
            data(&mul(&matrix, &scalar)),
            arr2(&[[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]).into_dyn()
        );
    }

    /// `ndarray`'s own `+` stretches only the *right* operand onto the left
    /// one's shape, so this is the case that fails without an explicit
    /// broadcast of both sides.
    #[test]
    fn the_smaller_operand_may_be_on_the_left() {
        let matrix = Variable::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn());
        let row = var(&[10.0, 20.0, 30.0]);

        assert_eq!(
            data(&add(&row, &matrix)),
            arr2(&[[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]]).into_dyn()
        );
        assert_eq!(
            data(&sub(&row, &matrix)),
            arr2(&[[9.0, 18.0, 27.0], [6.0, 15.0, 24.0]]).into_dyn()
        );
    }

    #[test]
    fn broadcast_gradients_are_summed_back_onto_each_operand() {
        // The row is added once per matrix row, so its gradient counts both.
        let matrix = Variable::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn());
        let row = var(&[10.0, 20.0, 30.0]);
        let y = add(&matrix, &row);
        y.backward();

        assert_eq!(
            grad(&matrix),
            ArrayD::from_elem(ndarray::IxDyn(&[2, 3]), 1.0)
        );
        assert_eq!(grad(&row), arr1(&[2.0, 2.0, 2.0]).into_dyn());

        // For `mul` the folded gradient is the *sum of the peers*, not a count.
        matrix.cleargrad();
        row.cleargrad();
        mul(&matrix, &row).backward();
        assert_eq!(
            grad(&matrix),
            arr2(&[[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]]).into_dyn()
        );
        assert_eq!(grad(&row), arr1(&[5.0, 7.0, 9.0]).into_dyn());
    }

    #[test]
    fn broadcast_gradients_match_numerical_diff() {
        let row = var(&[0.5, -1.5, 2.5]);
        let column = Variable::new(arr2(&[[2.0], [3.0]]).into_dyn());
        let matrix = Variable::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn());

        // Differentiating the *broadcast* operand is the interesting half: its
        // gradient is a sum over copies, which finite differences check for
        // free.
        gradient_check(|x| add(&matrix, x), &row, EPS, RTOL, ATOL).expect("add row");
        gradient_check(|x| mul(&matrix, x), &row, EPS, RTOL, ATOL).expect("mul row");
        gradient_check(|x| sub(&matrix, x), &row, EPS, RTOL, ATOL).expect("sub row");
        gradient_check(|x| div(&matrix, x), &row, EPS, RTOL, ATOL).expect("div row");
        gradient_check(|x| add(x, &matrix), &row, EPS, RTOL, ATOL).expect("row on the left");
        gradient_check(|x| mul(&matrix, x), &column, EPS, RTOL, ATOL).expect("mul column");
        gradient_check(
            |x| mul(&matrix, x),
            &Variable::from_scalar(3.0),
            EPS,
            RTOL,
            ATOL,
        )
        .expect("mul scalar");
    }

    /// Both operands stretched at once: `(2, 1, 4) * (3, 1)` broadcasts to
    /// `(2, 3, 4)`, so *each* gradient is a sum over copies at a different set
    /// of axes. Every shipped fixture has one operand already at the output
    /// shape, so this case is checked here instead; the numbers below come from
    /// `vendor/dezero-python` and the port reproduces them to the last bit.
    #[test]
    fn both_operands_may_be_broadcast_at_once() {
        let a = Variable::new(
            ArrayD::from_shape_vec(
                ndarray::IxDyn(&[2, 1, 4]),
                vec![
                    1.690_525_703_800_356,
                    -0.465_937_370_540_832_8,
                    0.032_820_163_678_584_4,
                    0.407_516_282_996_507_83,
                    -0.788_923_028_625_738_6,
                    0.002_065_572_905_948_13,
                    -0.000_890_385_857_931_362_8,
                    -1.754_724_306_345_420_8,
                ],
            )
            .expect("shape matches the element count"),
        );
        let b = Variable::new(
            ArrayD::from_shape_vec(
                ndarray::IxDyn(&[3, 1]),
                vec![
                    1.017_658_005_663_493_2,
                    0.600_498_515_919_549_4,
                    -0.625_428_973_966_759_7,
                ],
            )
            .expect("shape matches the element count"),
        );

        let y = mul(&a, &b);
        assert_eq!(y.shape(), Some(vec![2, 3, 4]));
        assert_eq!(
            data(&y)[[1, 2, 3]],
            1.097_455_422_512_150_7,
            "y[i, j, k] = a[i, 0, k] * b[j, 0]"
        );

        y.backward();
        // ga = sum over j of b[j] , at every (i, 0, k); gb = sum over i and k
        // of a, at every (j, 0).
        assert_eq!(
            grad(&a),
            ArrayD::from_elem(ndarray::IxDyn(&[2, 1, 4]), 0.992_727_547_616_282_8)
        );
        assert_eq!(
            grad(&b),
            ArrayD::from_elem(ndarray::IxDyn(&[3, 1]), -0.877_547_367_988_527_3)
        );
    }

    #[test]
    fn scalar_operands_stay_zero_dimensional() {
        // Step 21 used to materialise the scalar at the peer's shape; with
        // broadcasting it stays 0-d, exactly as Python's `as_array(2.0)` does.
        let x = var(&[1.0, 2.0, 4.0]);
        let y = &x * 2.0;
        let constant = y
            .creator()
            .expect("the multiplication is recorded")
            .inputs()
            .into_iter()
            .find(|input| input.id() != x.id())
            .expect("the scalar is the other operand");

        assert_eq!(constant.shape(), Some(vec![]));
        assert_eq!(data(&y), arr1(&[2.0, 4.0, 8.0]).into_dyn());

        y.backward();
        assert_eq!(grad(&x), arr1(&[2.0, 2.0, 2.0]).into_dyn());
        assert_eq!(
            grad(&constant),
            ndarray::arr0(7.0).into_dyn(),
            "the scalar's own gradient is summed back to 0-d"
        );
    }

    #[test]
    #[should_panic(expected = "which do not broadcast together")]
    fn incompatible_shapes_are_still_rejected() {
        let a = var(&[1.0, 2.0, 3.0]);
        let b = var(&[1.0, 2.0]);
        let _ = add(&a, &b);
    }

    #[test]
    #[should_panic(expected = "expects exactly 2 inputs")]
    fn wrong_arity_is_rejected() {
        let a = var(&[1.0, 2.0, 3.0]);
        let _ = apply1(Add, &[&a]);
    }
}
