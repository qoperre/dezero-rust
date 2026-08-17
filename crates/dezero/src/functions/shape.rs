//! Tensor shape manipulation: [`reshape`] and [`transpose`] (step 37).
//!
//! Port of the "Tensor operations" section of
//! `vendor/dezero-python/dezero/functions.py`. Both functions move elements
//! without changing them, so both backwards are the *inverse move* applied to
//! the gradient — `Reshape` reshapes back to the input's shape, `Transpose`
//! transposes again.
//!
//! Written in [`Variable`] arithmetic like every other backward here, so a
//! reshaped or transposed gradient is itself reshapeable and differentiable.

use ndarray::{ArrayD, IxDyn};

use crate::core::function::{Op, apply1};
use crate::core::ops::one;
use crate::core::variable::Variable;

/// The shape of a variable that must hold data.
///
/// # Panics
///
/// Panics if `v` holds no data. Every variable reaching an `Op` has been
/// through [`apply`](crate::apply), which rejects empty inputs, so this states
/// an invariant rather than handling a case.
fn shape_of(v: &Variable, op: &str) -> Vec<usize> {
    v.shape()
        .unwrap_or_else(|| panic!("dezero: {op} needs a variable that holds data"))
}

/// Re-lays `x` into `shape`, reading and writing in C (row-major) order.
///
/// # Panics
///
/// Panics if the element counts differ, which is numpy's `ValueError`.
fn reshaped(x: &ArrayD<f64>, shape: &[usize]) -> ArrayD<f64> {
    // `to_shape` borrows when the layout allows and copies when it does not —
    // exactly what numpy's `reshape` does, and the reason a transposed array
    // can be reshaped at all.
    match x.to_shape(IxDyn(shape)) {
        Ok(reshaped) => reshaped.into_owned(),
        Err(error) => panic!(
            "dezero: cannot reshape an array of shape {:?} into {shape:?} ({error})",
            x.shape()
        ),
    }
}

// ---------------------------------------------------------------------------
// Reshape
// ---------------------------------------------------------------------------

/// `y = x.reshape(shape)`.
#[derive(Debug, Clone)]
pub struct Reshape {
    shape: Vec<usize>,
}

impl Reshape {
    /// Creates the op for a given target shape.
    #[must_use]
    pub fn new(shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
        }
    }
}

impl Op for Reshape {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = one(xs, "Reshape", "input");
        vec![reshaped(x, &self.shape)]
    }

    fn backward(
        &self,
        inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let x = one(inputs, "Reshape", "input");
        let gy = one(gys, "Reshape", "output gradient");
        // Python stores `x_shape` in forward; reading it back off the input is
        // the same thing, and is what `Mul.backward` does one file over.
        vec![reshape(gy, &shape_of(x, "Reshape"))]
    }
}

/// Reshapes a variable — Python's `dezero.functions.reshape`.
///
/// When `shape` already matches, the variable is returned as it is and **no
/// graph node is created**, exactly like Python's early `return as_variable(x)`.
///
/// Unlike numpy this takes concrete lengths only: there is no `-1` placeholder
/// to infer a dimension from the others. Callers that want one (Python's
/// `flatten`) can compute it from [`Variable::size`].
///
/// # Examples
///
/// ```
/// use dezero::{reshape, Variable};
/// use ndarray::arr2;
///
/// let x = Variable::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn());
/// let y = reshape(&x, &[6]);
/// assert_eq!(y.shape(), Some(vec![6]));
///
/// // The gradient is reshaped back, so it always matches the input.
/// y.backward();
/// assert_eq!(x.grad().and_then(|g| g.shape()), Some(vec![2, 3]));
/// ```
///
/// # Panics
///
/// Panics if `shape` does not have the same number of elements as `x`, or if
/// `x` holds no data.
#[must_use]
pub fn reshape(x: &Variable, shape: &[usize]) -> Variable {
    if x.shape().as_deref() == Some(shape) {
        return x.clone();
    }
    apply1(Reshape::new(shape), &[x])
}

// ---------------------------------------------------------------------------
// Transpose
// ---------------------------------------------------------------------------

/// `y = x.transpose()` — every axis reversed.
#[derive(Debug, Clone, Copy)]
pub struct Transpose;

impl Op for Transpose {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = one(xs, "Transpose", "input");
        // `.t()` only re-labels the strides; `to_owned` materialises the result
        // in C order so that everything downstream sees a standard-layout
        // array, as numpy's `.copy()` would.
        vec![x.t().to_owned()]
    }

    fn backward(
        &self,
        _inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let gy = one(gys, "Transpose", "output gradient");
        // Reversing the axes is an involution, so the inverse is itself.
        vec![transpose(gy)]
    }
}

/// Reverses a variable's axes — Python's `dezero.functions.transpose(x)` with
/// its default `axes=None`.
///
/// Generalised beyond the 2-D case: like numpy's `.T`, this reverses *all*
/// axes, so it is already correct at rank 3 and above. What is **not** ported
/// yet is Python's optional `axes` argument, an arbitrary permutation whose
/// backward needs the inverse permutation (`np.argsort`); no step up to 40 uses
/// it, and it arrives with the first function that does.
///
/// # Examples
///
/// ```
/// use dezero::{transpose, Variable};
/// use ndarray::arr2;
///
/// let x = Variable::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn());
/// let y = transpose(&x);
/// assert_eq!(y.data(), Some(arr2(&[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]).into_dyn()));
/// ```
///
/// # Panics
///
/// Panics if `x` holds no data.
#[must_use]
pub fn transpose(x: &Variable) -> Variable {
    apply1(Transpose, &[x])
}

/// The method forms the book adds in step 37, `Variable.reshape` and
/// `Variable.T`.
///
/// Inherent methods may be written in any module of the defining crate, so
/// these live beside the functions they delegate to rather than dragging a
/// dependency on [`crate::functions`] into `core::variable`.
impl Variable {
    /// Reshapes this variable — Python's `Variable.reshape`.
    ///
    /// See [`reshape`] for the details, including the no-node identity case.
    ///
    /// # Panics
    ///
    /// Panics if `shape` does not have the same number of elements.
    #[must_use]
    pub fn reshape(&self, shape: &[usize]) -> Self {
        reshape(self, shape)
    }

    /// Reverses this variable's axes — Python's `Variable.transpose()`.
    ///
    /// # Panics
    ///
    /// Panics if this variable holds no data.
    #[must_use]
    pub fn transpose(&self) -> Self {
        transpose(self)
    }

    /// Short alias for [`Variable::transpose`] — Python's `Variable.T`, and
    /// the spelling `ndarray` uses.
    ///
    /// # Panics
    ///
    /// Panics if this variable holds no data.
    #[must_use]
    pub fn t(&self) -> Self {
        transpose(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functions::reduce::sum_all;
    use crate::utils::gradient_check;
    use ndarray::{arr1, arr2, arr3};

    const EPS: f64 = 1e-4;
    const RTOL: f64 = 1e-4;
    const ATOL: f64 = 1e-5;

    fn matrix() -> Variable {
        Variable::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn())
    }

    #[test]
    fn reshape_forward_reads_in_c_order() {
        let y = reshape(&matrix(), &[3, 2]);
        assert_eq!(
            y.data(),
            Some(arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).into_dyn())
        );
    }

    #[test]
    fn reshape_backward_restores_the_input_shape() {
        let x = matrix();
        let y = reshape(&x, &[6]);
        y.backward();
        assert_eq!(
            x.grad().and_then(|g| g.data()),
            Some(arr2(&[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]).into_dyn())
        );
    }

    #[test]
    fn reshape_to_the_same_shape_builds_no_node() {
        let x = matrix();
        let y = reshape(&x, &[2, 3]);
        assert_eq!(y.id(), x.id(), "Python returns the very same variable");
        assert!(y.creator().is_none());
    }

    #[test]
    fn reshape_can_add_and_remove_unit_axes() {
        let x = matrix();
        assert_eq!(reshape(&x, &[1, 2, 3]).shape(), Some(vec![1, 2, 3]));
        assert_eq!(reshape(&x, &[6, 1]).shape(), Some(vec![6, 1]));
        assert_eq!(
            reshape(&Variable::new(arr2(&[[7.0]]).into_dyn()), &[1]).data(),
            Some(arr1(&[7.0]).into_dyn())
        );
    }

    #[test]
    fn reshape_gradient_matches_numerical_diff() {
        gradient_check(|x| reshape(x, &[3, 2]), &matrix(), EPS, RTOL, ATOL).expect("reshape");
    }

    #[test]
    #[should_panic(expected = "cannot reshape an array of shape [2, 3] into [4]")]
    fn reshape_rejects_a_different_element_count() {
        let _ = reshape(&matrix(), &[4]);
    }

    #[test]
    fn transpose_forward_swaps_the_axes() {
        let y = transpose(&matrix());
        assert_eq!(
            y.data(),
            Some(arr2(&[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]).into_dyn())
        );
    }

    #[test]
    fn transpose_backward_transposes_the_gradient() {
        // A non-symmetric weighting: a backward that forgot to transpose would
        // return the gradient with rows and columns swapped.
        let x = matrix();
        let weights = Variable::new(arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).into_dyn());
        sum_all(&(&transpose(&x) * &weights)).backward();
        assert_eq!(
            x.grad().and_then(|g| g.data()),
            Some(arr2(&[[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]).into_dyn())
        );
    }

    #[test]
    fn transpose_is_its_own_inverse() {
        let x = matrix();
        let y = transpose(&transpose(&x));
        assert_eq!(y.data(), x.data());
    }

    #[test]
    fn transpose_reverses_every_axis_at_rank_three() {
        // numpy's `.T` on a 3-d array reverses all three axes.
        let x =
            Variable::new(arr3(&[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]).into_dyn());
        let y = transpose(&x);
        assert_eq!(y.shape(), Some(vec![2, 2, 2]));
        let data = y.data().expect("data");
        assert_eq!(data[[0, 1, 1]], 7.0, "y[i, j, k] == x[k, j, i]");
        assert_eq!(data[[1, 0, 0]], 2.0);
    }

    #[test]
    fn transpose_of_a_vector_is_a_no_op() {
        let x = Variable::new(arr1(&[1.0, 2.0, 3.0]).into_dyn());
        assert_eq!(transpose(&x).data(), x.data());
    }

    #[test]
    fn transpose_gradient_matches_numerical_diff() {
        gradient_check(transpose, &matrix(), EPS, RTOL, ATOL).expect("transpose");
    }

    #[test]
    fn the_method_forms_delegate_to_the_functions() {
        let x = matrix();
        assert_eq!(x.reshape(&[6]).data(), reshape(&x, &[6]).data());
        assert_eq!(x.transpose().data(), transpose(&x).data());
        assert_eq!(x.t().data(), transpose(&x).data());
        assert_eq!(x.reshape(&[2, 3]).id(), x.id(), "the identity case too");
    }

    #[test]
    fn reshaping_a_transposed_variable_reads_the_transposed_order() {
        // The case that forces `reshaped` to copy: the transposed array is not
        // contiguous, and reading it in C order must follow the *logical*
        // layout, not the buffer.
        let y = reshape(&transpose(&matrix()), &[6]);
        assert_eq!(
            y.data(),
            Some(arr1(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).into_dyn())
        );
    }
}
