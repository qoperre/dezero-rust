//! Matrix multiplication and the affine transform (steps 41–43).
//!
//! Port of the "matmul / linear" section of
//! `vendor/dezero-python/dezero/functions.py`.
//!
//! Both backwards are the same two lines, and they are the reason this file
//! exists as its own op rather than as a composition:
//!
//! ```text
//! gx = matmul(gy, W.T)
//! gW = matmul(x.T, gy)
//! ```
//!
//! Written in [`Variable`] arithmetic like every other backward here, so the
//! gradient of a matrix product is itself a differentiable matrix product.
//!
//! # Two `Linear`s
//!
//! Python has both `F.Linear` (a `Function` — the mathematics, here) and
//! `L.Linear` (a `Layer` — the mathematics *plus* the weights it owns, in
//! [`crate::layers::linear`]). The port keeps both names in the same two
//! places. Only the layer is re-exported at the crate root, because it is the
//! one user code names; reach for the function through
//! [`crate::functions::matmul::Linear`] or, far more likely, through the
//! [`linear`] free function.

use ndarray::{ArrayD, Ix2};

use crate::core::function::{Op, apply1};
use crate::core::ops::{one, two};
use crate::core::variable::Variable;
use crate::functions::reduce::sum_to;
use crate::functions::shape::transpose;

/// The shape of a variable that must hold data.
///
/// # Panics
///
/// Panics if `v` holds no data; [`apply`](crate::apply) has already rejected
/// that case for anything reaching an `Op`.
fn shape_of(v: &Variable, op: &str) -> Vec<usize> {
    v.shape()
        .unwrap_or_else(|| panic!("dezero: {op} needs variables that hold data"))
}

/// `a.dot(b)` for two 2-dimensional arrays — numpy's `x.dot(W)`.
///
/// `ndarray`'s `dot` is defined on `Array2`, not on the dynamic-dimension
/// `ArrayD` the graph carries, so the views are re-typed first. That re-typing
/// is also where the rank check happens.
///
/// # Panics
///
/// Panics if either operand is not 2-dimensional, or if the inner dimensions do
/// not agree (numpy's `ValueError: shapes ... not aligned`).
fn dot2(a: &ArrayD<f64>, b: &ArrayD<f64>, op: &str) -> ArrayD<f64> {
    let (Ok(left), Ok(right)) = (
        a.view().into_dimensionality::<Ix2>(),
        b.view().into_dimensionality::<Ix2>(),
    ) else {
        panic!(
            "dezero: {op} needs two 2-dimensional operands, got shapes {:?} and {:?}",
            a.shape(),
            b.shape()
        );
    };
    assert!(
        left.ncols() == right.nrows(),
        "dezero: {op} cannot multiply {:?} by {:?}: the inner dimensions differ",
        a.shape(),
        b.shape()
    );
    left.dot(&right).into_dyn()
}

// ---------------------------------------------------------------------------
// MatMul
// ---------------------------------------------------------------------------

/// `y = x.dot(W)`.
#[derive(Debug, Clone, Copy)]
pub struct MatMul;

impl Op for MatMul {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let (x, w) = two(xs, "MatMul", "inputs");
        vec![dot2(x, w, "MatMul")]
    }

    fn backward(
        &self,
        inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let (x, w) = two(inputs, "MatMul", "inputs");
        let gy = one(gys, "MatMul", "output gradient");
        vec![matmul(gy, &transpose(w)), matmul(&transpose(x), gy)]
    }
}

/// Multiplies two matrices — Python's `dezero.functions.matmul`.
///
/// Both operands must be 2-dimensional. numpy's `dot` generalises to other
/// ranks (vectors, stacked batches); DeZero's `MatMul.backward` is written for
/// the matrix case alone — `W.T` reverses *every* axis — so the port rejects
/// other ranks rather than computing a forward value whose gradient would be
/// silently wrong.
///
/// # Examples
///
/// ```
/// use dezero::{matmul, Variable};
/// use ndarray::arr2;
///
/// let x = Variable::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn());
/// let w = Variable::new(arr2(&[[1.0], [0.0], [-1.0]]).into_dyn());
///
/// let y = matmul(&x, &w);
/// assert_eq!(y.data(), Some(arr2(&[[-2.0], [-2.0]]).into_dyn()));
///
/// // dy/dx = gy . W^T, so every row of the gradient is W^T itself.
/// y.backward();
/// assert_eq!(
///     x.grad().and_then(|g| g.data()),
///     Some(arr2(&[[1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]).into_dyn())
/// );
/// ```
///
/// # Panics
///
/// Panics if either operand is not a 2-dimensional matrix, if their inner
/// dimensions differ, or if either holds no data.
#[must_use]
pub fn matmul(x: &Variable, w: &Variable) -> Variable {
    apply1(MatMul, &[x, w])
}

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

/// `y = x.dot(W) + b`, with an optional bias.
///
/// Python encodes "no bias" by passing `b=None`, which reaches `forward` as a
/// real `None` and `backward` as a `Variable` whose `data` is `None`. The port
/// encodes it in the **arity** instead: two inputs mean no bias, three mean a
/// bias. That is not a stylistic preference — [`apply`](crate::apply) refuses to
/// run an op on a variable that holds no data, so a `None` bias cannot be an
/// input at all.
#[derive(Debug, Clone, Copy)]
pub struct Linear;

/// Splits an argument list that is `(x, W)` or `(x, W, b)`.
///
/// # Panics
///
/// Panics on any other arity.
fn split<'a, T>(items: &'a [T], role: &str) -> (&'a T, &'a T, Option<&'a T>) {
    match items {
        [x, w] => (x, w, None),
        [x, w, b] => (x, w, Some(b)),
        _ => panic!(
            "dezero: Linear expects 2 or 3 {role} (x, W and an optional b), got {}",
            items.len()
        ),
    }
}

impl Op for Linear {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let (x, w, b) = split(xs, "inputs");
        let mut y = dot2(x, w, "Linear");
        if let Some(b) = b {
            // numpy's `y += b` broadcasts a `(out,)` bias across the batch. The
            // broadcast is spelled out so a bias of the wrong shape names itself
            // in the panic instead of surfacing as an `ndarray` internal error.
            let Some(view) = b.broadcast(y.raw_dim()) else {
                panic!(
                    "dezero: Linear cannot broadcast a bias of shape {:?} onto an output of \
                     shape {:?}",
                    b.shape(),
                    y.shape()
                );
            };
            y += &view;
        }
        vec![y]
    }

    fn backward(
        &self,
        inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let (x, w, b) = split(inputs, "inputs");
        let gy = one(gys, "Linear", "output gradient");

        let gx = matmul(gy, &transpose(w));
        let gw = matmul(&transpose(x), gy);
        match b {
            // The bias was broadcast across the batch, so its gradient is the
            // sum over the rows -- Python's `sum_to(gy, b.shape)`.
            Some(b) => vec![gx, gw, sum_to(gy, &shape_of(b, "Linear"))],
            None => vec![gx, gw],
        }
    }
}

/// The affine transform `y = x W + b` — Python's `dezero.functions.linear`.
///
/// `b` is optional, matching Python's `linear(x, W, b=None)`. Pass
/// `Some(&bias)` for the usual case and `None` for a bias-free layer.
///
/// This is one op rather than `matmul` followed by `+`, which matters for more
/// than tidiness: the fused version never materialises the intermediate
/// `x W`, which is the memory saving Python's `linear_simple` chases by
/// manually clearing `t.data` afterwards.
///
/// # Examples
///
/// ```
/// use dezero::{linear, Variable};
/// use ndarray::{arr1, arr2};
///
/// let x = Variable::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn());
/// let w = Variable::new(arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn());
/// let b = Variable::new(arr1(&[10.0, 20.0]).into_dyn());
///
/// let y = linear(&x, &w, Some(&b));
/// assert_eq!(y.data(), Some(arr2(&[[11.0, 22.0], [13.0, 24.0]]).into_dyn()));
///
/// // The bias is shared by every row, so its gradient counts them all.
/// y.backward();
/// assert_eq!(b.grad().and_then(|g| g.data()), Some(arr1(&[2.0, 2.0]).into_dyn()));
///
/// // Without a bias it is exactly `matmul`.
/// assert_eq!(linear(&x, &w, None).data(), dezero::matmul(&x, &w).data());
/// ```
///
/// # Panics
///
/// Panics if `x` or `w` is not a 2-dimensional matrix, if their inner
/// dimensions differ, if `b` does not broadcast onto the output, or if any
/// operand holds no data.
#[must_use]
pub fn linear(x: &Variable, w: &Variable, b: Option<&Variable>) -> Variable {
    match b {
        Some(b) => apply1(Linear, &[x, w, b]),
        None => apply1(Linear, &[x, w]),
    }
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

    fn x23() -> Variable {
        Variable::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn())
    }

    fn w32() -> Variable {
        Variable::new(arr2(&[[1.0, -1.0], [0.5, 2.0], [-2.0, 0.25]]).into_dyn())
    }

    fn b2() -> Variable {
        Variable::new(arr1(&[0.5, -1.5]).into_dyn())
    }

    fn data(v: &Variable) -> ArrayD<f64> {
        v.data().expect("variable holds data")
    }

    fn grad(v: &Variable) -> ArrayD<f64> {
        data(&v.grad().expect("variable has a gradient"))
    }

    // -- matmul forward ----------------------------------------------------

    #[test]
    fn matmul_forward_matches_the_hand_computed_product() {
        let y = matmul(&x23(), &w32());
        assert_eq!(y.shape(), Some(vec![2, 2]));
        assert_eq!(
            data(&y),
            arr2(&[[-4.0, 3.75], [-5.5, 7.5]]).into_dyn(),
            "rows of x dotted with columns of W"
        );
    }

    #[test]
    fn matmul_handles_the_degenerate_row_by_column_case() {
        let row = Variable::new(arr2(&[[1.0, 2.0, 3.0]]).into_dyn());
        let column = Variable::new(arr2(&[[4.0], [5.0], [6.0]]).into_dyn());
        assert_eq!(data(&matmul(&row, &column)), arr2(&[[32.0]]).into_dyn());
        assert_eq!(matmul(&column, &row).shape(), Some(vec![3, 3]));
    }

    // -- matmul backward ---------------------------------------------------

    #[test]
    fn matmul_backward_is_the_transposed_product() {
        let x = x23();
        let w = w32();
        matmul(&x, &w).backward();

        // With a seed of ones, gx = ones . W^T is the row sums of W repeated,
        // and gW = x^T . ones is the column sums of x repeated.
        assert_eq!(
            grad(&x),
            arr2(&[[0.0, 2.5, -1.75], [0.0, 2.5, -1.75]]).into_dyn()
        );
        assert_eq!(
            grad(&w),
            arr2(&[[5.0, 5.0], [7.0, 7.0], [9.0, 9.0]]).into_dyn()
        );
    }

    #[test]
    fn matmul_gradient_matches_numerical_diff() {
        gradient_check(|x| matmul(x, &w32()), &x23(), EPS, RTOL, ATOL).expect("d/dx");
        gradient_check(|w| matmul(&x23(), w), &w32(), EPS, RTOL, ATOL).expect("d/dW");

        // A non-uniform seed, so a transposed gradient could not pass by
        // symmetry.
        let weights = Variable::new(arr2(&[[1.0, -2.0], [0.5, 3.0]]).into_dyn());
        gradient_check(
            |x| sum_all(&(&matmul(x, &w32()) * &weights)),
            &x23(),
            EPS,
            RTOL,
            ATOL,
        )
        .expect("weighted d/dx");
    }

    #[test]
    #[should_panic(expected = "the inner dimensions differ")]
    fn matmul_rejects_mismatched_inner_dimensions() {
        let _ = matmul(&x23(), &x23());
    }

    #[test]
    #[should_panic(expected = "two 2-dimensional operands")]
    fn matmul_rejects_a_vector() {
        let v = Variable::new(arr1(&[1.0, 2.0, 3.0]).into_dyn());
        let _ = matmul(&v, &w32());
    }

    // -- linear ------------------------------------------------------------

    #[test]
    fn linear_adds_the_bias_to_every_row() {
        let y = linear(&x23(), &w32(), Some(&b2()));
        assert_eq!(
            data(&y),
            arr2(&[[-3.5, 2.25], [-5.0, 6.0]]).into_dyn(),
            "matmul, then b broadcast down the batch"
        );
    }

    #[test]
    fn linear_without_a_bias_is_matmul() {
        let y = linear(&x23(), &w32(), None);
        assert_eq!(data(&y), data(&matmul(&x23(), &w32())));

        // ... including in the backward pass.
        let (a, wa) = (x23(), w32());
        let (b, wb) = (x23(), w32());
        linear(&a, &wa, None).backward();
        matmul(&b, &wb).backward();
        assert_eq!(grad(&a), grad(&b));
        assert_eq!(grad(&wa), grad(&wb));
    }

    #[test]
    fn linear_bias_gradient_is_the_batch_sum() {
        let x = x23();
        let w = w32();
        let b = b2();
        linear(&x, &w, Some(&b)).backward();
        assert_eq!(
            grad(&b),
            arr1(&[2.0, 2.0]).into_dyn(),
            "one row of the seed per batch element"
        );
    }

    #[test]
    fn linear_gradients_match_numerical_diff() {
        gradient_check(|x| linear(x, &w32(), Some(&b2())), &x23(), EPS, RTOL, ATOL).expect("d/dx");
        gradient_check(|w| linear(&x23(), w, Some(&b2())), &w32(), EPS, RTOL, ATOL).expect("d/dW");
        gradient_check(|b| linear(&x23(), &w32(), Some(b)), &b2(), EPS, RTOL, ATOL).expect("d/db");
        gradient_check(|x| linear(x, &w32(), None), &x23(), EPS, RTOL, ATOL)
            .expect("d/dx, no bias");
    }

    /// A weighted objective, so a gradient that came out transposed or summed
    /// over the wrong axis cannot survive.
    #[test]
    fn linear_gradients_match_numerical_diff_under_a_non_uniform_seed() {
        let weights = Variable::new(arr2(&[[1.0, -2.0], [0.5, 3.0]]).into_dyn());
        let objective = |x: &Variable, w: &Variable, b: &Variable| {
            sum_all(&(&linear(x, w, Some(b)) * &weights))
        };
        gradient_check(|x| objective(x, &w32(), &b2()), &x23(), EPS, RTOL, ATOL).expect("d/dx");
        gradient_check(|w| objective(&x23(), w, &b2()), &w32(), EPS, RTOL, ATOL).expect("d/dW");
        gradient_check(|b| objective(&x23(), &w32(), b), &b2(), EPS, RTOL, ATOL).expect("d/db");
    }

    #[test]
    fn linear_accepts_a_row_shaped_bias() {
        // A `(1, out)` bias broadcasts just as a `(out,)` one does, and its
        // gradient comes back at its own shape.
        let b = Variable::new(arr2(&[[0.5, -1.5]]).into_dyn());
        let y = linear(&x23(), &w32(), Some(&b));
        assert_eq!(data(&y), arr2(&[[-3.5, 2.25], [-5.0, 6.0]]).into_dyn());
        y.backward();
        assert_eq!(grad(&b), arr2(&[[2.0, 2.0]]).into_dyn());
    }

    #[test]
    #[should_panic(expected = "cannot broadcast a bias of shape [3]")]
    fn linear_rejects_a_bias_that_does_not_broadcast() {
        let b = Variable::new(arr1(&[1.0, 2.0, 3.0]).into_dyn());
        let _ = linear(&x23(), &w32(), Some(&b));
    }

    #[test]
    #[should_panic(expected = "expects 2 or 3 inputs")]
    fn linear_rejects_a_wrong_arity() {
        let _ = apply1(Linear, &[&x23()]);
    }

    // -- higher-order ------------------------------------------------------

    /// Both backwards are `matmul` calls on `Variable`s, so the gradient of a
    /// matrix product can itself be differentiated (`docs/ARCHITECTURE.md`'s
    /// critical rule).
    #[test]
    fn matmul_gradients_stay_differentiable() {
        // y = sum(x . x), so dy/dx = ones . x^T + x^T . ones, and each entry of
        // d2y/dx2 counts how many times that entry is reused.
        let x = Variable::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn());
        let y = sum_all(&matmul(&x, &x));
        y.backward_with(false, true);

        let gx = x.grad().expect("first derivative");
        assert!(gx.creator().is_some(), "the gradient is a graph node");

        x.cleargrad();
        sum_all(&gx).backward();
        assert_eq!(
            grad(&x),
            ndarray::ArrayD::from_elem(ndarray::IxDyn(&[2, 2]), 4.0),
            "each entry appears in four terms of sum(x . x)"
        );
    }
}
