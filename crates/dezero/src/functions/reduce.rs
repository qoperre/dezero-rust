//! Reductions and broadcasting: [`sum`], [`sum_to`] and [`broadcast_to`]
//! (steps 38–40).
//!
//! Port of the "sum / sum_to / broadcast_to" section of
//! `vendor/dezero-python/dezero/functions.py`. These three are the hinge the
//! whole broadcasting story turns on:
//!
//! * `BroadcastTo::backward` is [`sum_to`],
//! * `SumTo::backward` is [`broadcast_to`],
//! * `Sum::backward` is a reshape followed by a [`broadcast_to`].
//!
//! They are *mutually inverse*, which is why a mistake in either one produces
//! a gradient of the right shape and the wrong values — silently. The shape
//! arithmetic itself lives in [`crate::utils::shape`], ported statement by
//! statement from `dezero/utils.py`.
//!
//! # The identity case
//!
//! Python's `broadcast_to(x, shape)` and `sum_to(x, shape)` `return
//! as_variable(x)` when the shape already matches, creating **no graph node**
//! at all. The port does the same, returning the very same [`Variable`]. That
//! is load-bearing: it means `y = broadcast_to(x, x.shape)` has `x`'s creator,
//! not a new one, and a backward pass started at `y` is a backward pass started
//! at `x`.

use std::borrow::Cow;

use ndarray::{ArrayD, Axis, IxDyn};

use crate::core::function::{Op, apply1};
use crate::core::ops::one;
use crate::core::variable::Variable;
use crate::functions::shape::reshape;
use crate::utils::shape::{normalize_axis, reshape_sum_backward, sum_to as sum_to_array};

/// Which axes a [`sum`] reduces — Python's `axis=None | int | tuple`.
///
/// # Examples
///
/// ```
/// use dezero::{sum, Axes, Variable};
/// use ndarray::arr2;
///
/// let x = Variable::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn());
///
/// assert_eq!(sum(&x, Axes::All, false).shape(), Some(vec![]));
/// assert_eq!(sum(&x, 0, false).shape(), Some(vec![3]));       // int
/// assert_eq!(sum(&x, [0, 1], true).shape(), Some(vec![1, 1])); // tuple
/// assert_eq!(sum(&x, -1, false).shape(), Some(vec![2]));       // from the end
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum Axes {
    /// Every axis — Python's `axis=None`, the default.
    #[default]
    All,
    /// Only the listed axes. Negative indices count back from the last axis,
    /// as in numpy.
    Only(Vec<isize>),
}

impl Axes {
    /// The listed axes, or `None` for [`Axes::All`] — the form
    /// [`reshape_sum_backward`] takes.
    #[must_use]
    pub fn as_slice(&self) -> Option<&[isize]> {
        match self {
            Self::All => None,
            Self::Only(axes) => Some(axes),
        }
    }

    /// Resolves to concrete axis indices for an array of `ndim` dimensions,
    /// ascending.
    ///
    /// # Panics
    ///
    /// Panics if an axis is out of bounds, or if one is listed twice (numpy's
    /// `ValueError: duplicate value in 'axis'`).
    #[must_use]
    pub fn resolve(&self, ndim: usize) -> Vec<usize> {
        match self {
            Self::All => (0..ndim).collect(),
            Self::Only(axes) => {
                let mut resolved: Vec<usize> =
                    axes.iter().map(|&a| normalize_axis(a, ndim)).collect();
                resolved.sort_unstable();
                let listed = resolved.len();
                resolved.dedup();
                assert!(
                    resolved.len() == listed,
                    "dezero: sum got a duplicate axis in {axes:?}"
                );
                resolved
            }
        }
    }
}

impl From<isize> for Axes {
    fn from(axis: isize) -> Self {
        Self::Only(vec![axis])
    }
}

impl From<Vec<isize>> for Axes {
    fn from(axes: Vec<isize>) -> Self {
        Self::Only(axes)
    }
}

impl From<&[isize]> for Axes {
    fn from(axes: &[isize]) -> Self {
        Self::Only(axes.to_vec())
    }
}

impl<const N: usize> From<[isize; N]> for Axes {
    fn from(axes: [isize; N]) -> Self {
        Self::Only(axes.to_vec())
    }
}

/// The shape of a variable that must hold data.
///
/// # Panics
///
/// Panics if `v` holds no data; [`apply`](crate::apply) has already rejected
/// that case for anything reaching an `Op`.
fn shape_of(v: &Variable, op: &str) -> Vec<usize> {
    v.shape()
        .unwrap_or_else(|| panic!("dezero: {op} needs a variable that holds data"))
}

// ---------------------------------------------------------------------------
// Sum
// ---------------------------------------------------------------------------

/// `y = x.sum(axis, keepdims)`.
#[derive(Debug, Clone)]
pub struct Sum {
    axis: Axes,
    keepdims: bool,
}

impl Sum {
    /// Creates the op.
    #[must_use]
    pub fn new(axis: impl Into<Axes>, keepdims: bool) -> Self {
        Self {
            axis: axis.into(),
            keepdims,
        }
    }
}

/// `x.sum(axis=axes, keepdims=keepdims)`.
///
/// `ndarray` reduces one axis at a time, so this loops. Descending order is
/// required: removing axis 2 leaves axes 0 and 1 where they were, while
/// removing axis 0 first would shift every later index.
fn sum_array(x: &ArrayD<f64>, axes: &Axes, keepdims: bool) -> ArrayD<f64> {
    let mut y = Cow::Borrowed(x);
    for &axis in axes.resolve(x.ndim()).iter().rev() {
        let reduced = y.sum_axis(Axis(axis));
        y = Cow::Owned(if keepdims {
            reduced.insert_axis(Axis(axis))
        } else {
            reduced
        });
    }
    y.into_owned()
}

impl Op for Sum {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = one(xs, "Sum", "input");
        vec![sum_array(x, &self.axis, self.keepdims)]
    }

    fn backward(
        &self,
        inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let x = one(inputs, "Sum", "input");
        let gy = one(gys, "Sum", "output gradient");
        let x_shape = shape_of(x, "Sum");

        // Python: `gy = utils.reshape_sum_backward(gy, x_shape, axis, keepdims)`
        // followed by `broadcast_to(gy, x_shape)`. The reshape puts back the
        // axes a `keepdims=False` sum dropped, so the broadcast spreads the
        // gradient over the axes that were actually summed.
        let aligned = reshape_sum_backward(
            &shape_of(gy, "Sum"),
            &x_shape,
            self.axis.as_slice(),
            self.keepdims,
        );
        vec![broadcast_to(&reshape(gy, &aligned), &x_shape)]
    }
}

/// Sums a variable over `axis` — Python's `dezero.functions.sum`.
///
/// `axis` accepts [`Axes::All`] (Python's `None`), a single `isize`, or an
/// array/`Vec` of them; see [`Axes`].
///
/// # Examples
///
/// ```
/// use dezero::{sum, Axes, Variable};
/// use ndarray::{arr1, arr2};
///
/// let x = Variable::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn());
///
/// assert_eq!(sum(&x, 0, false).data(), Some(arr1(&[5.0, 7.0, 9.0]).into_dyn()));
/// assert_eq!(sum(&x, 1, false).data(), Some(arr1(&[6.0, 15.0]).into_dyn()));
/// assert_eq!(sum(&x, Axes::All, false).data(), Some(ndarray::arr0(21.0).into_dyn()));
///
/// // Every element contributed once, so the gradient is all ones.
/// let y = sum(&x, 0, false);
/// y.backward();
/// assert_eq!(x.grad().and_then(|g| g.shape()), Some(vec![2, 3]));
/// ```
///
/// # Panics
///
/// Panics if an axis is out of bounds for `x`, if the same axis is listed
/// twice, or if `x` holds no data.
#[must_use]
pub fn sum(x: &Variable, axis: impl Into<Axes>, keepdims: bool) -> Variable {
    apply1(Sum::new(axis, keepdims), &[x])
}

/// Sums every element into a 0-d variable — Python's bare `F.sum(x)`.
///
/// Shorthand for `sum(x, Axes::All, false)`, which is far and away the most
/// common call: it is how a tensor becomes the scalar loss that
/// [`backward`](Variable::backward) is started from.
///
/// # Examples
///
/// ```
/// use dezero::{sum_all, Variable};
/// use ndarray::arr2;
///
/// let x = Variable::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn());
/// assert_eq!(sum_all(&x).data(), Some(ndarray::arr0(10.0).into_dyn()));
/// ```
///
/// # Panics
///
/// Panics if `x` holds no data.
#[must_use]
pub fn sum_all(x: &Variable) -> Variable {
    sum(x, Axes::All, false)
}

/// The method form the book adds in step 38, `Variable.sum`.
impl Variable {
    /// Sums this variable over `axis` — Python's `Variable.sum`.
    ///
    /// See [`sum`] for the details.
    ///
    /// # Examples
    ///
    /// ```
    /// use dezero::{Axes, Variable};
    /// use ndarray::arr2;
    ///
    /// let x = Variable::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn());
    /// assert_eq!(x.sum(Axes::All, false).data(), Some(ndarray::arr0(10.0).into_dyn()));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if an axis is out of bounds, if one is listed twice, or if this
    /// variable holds no data.
    #[must_use]
    pub fn sum(&self, axis: impl Into<Axes>, keepdims: bool) -> Self {
        sum(self, axis, keepdims)
    }
}

// ---------------------------------------------------------------------------
// SumTo
// ---------------------------------------------------------------------------

/// `y = utils.sum_to(x, shape)` — sums `x` down to `shape`.
#[derive(Debug, Clone)]
pub struct SumTo {
    shape: Vec<usize>,
}

impl SumTo {
    /// Creates the op for a given target shape.
    #[must_use]
    pub fn new(shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
        }
    }
}

impl Op for SumTo {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = one(xs, "SumTo", "input");
        vec![sum_to_array(x, &self.shape)]
    }

    fn backward(
        &self,
        inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let x = one(inputs, "SumTo", "input");
        let gy = one(gys, "SumTo", "output gradient");
        // The exact inverse of the forward: what was summed together gets the
        // same gradient back.
        vec![broadcast_to(gy, &shape_of(x, "SumTo"))]
    }
}

/// Sums a variable down to `shape` — Python's `dezero.functions.sum_to`.
///
/// The inverse of [`broadcast_to`]: axes that `shape` does not have are summed
/// away, and axes it has as 1 are summed but kept. When `shape` already
/// matches, the variable is returned as it is and **no graph node is created**.
///
/// # Examples
///
/// ```
/// use dezero::{sum_to, Variable};
/// use ndarray::arr2;
///
/// let x = Variable::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn());
/// assert_eq!(sum_to(&x, &[1, 3]).data(), Some(arr2(&[[5.0, 7.0, 9.0]]).into_dyn()));
/// assert_eq!(sum_to(&x, &[2, 1]).data(), Some(arr2(&[[6.0], [15.0]]).into_dyn()));
/// ```
///
/// # Panics
///
/// Panics if `x` cannot be summed down to `shape`, or if it holds no data.
#[must_use]
pub fn sum_to(x: &Variable, shape: &[usize]) -> Variable {
    if x.shape().as_deref() == Some(shape) {
        return x.clone();
    }
    apply1(SumTo::new(shape), &[x])
}

// ---------------------------------------------------------------------------
// BroadcastTo
// ---------------------------------------------------------------------------

/// `y = np.broadcast_to(x, shape)` — repeats `x` up to `shape`.
#[derive(Debug, Clone)]
pub struct BroadcastTo {
    shape: Vec<usize>,
}

impl BroadcastTo {
    /// Creates the op for a given target shape.
    #[must_use]
    pub fn new(shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
        }
    }
}

/// `np.broadcast_to(x, shape)`, materialised.
///
/// numpy returns a zero-stride read-only view; a [`Variable`] owns its data, so
/// the view is copied out. Nothing observable differs — only the memory.
///
/// # Panics
///
/// Panics if `x` does not broadcast to `shape`, which is numpy's `ValueError`.
fn broadcast_array(x: &ArrayD<f64>, shape: &[usize]) -> ArrayD<f64> {
    match x.broadcast(IxDyn(shape)) {
        Some(view) => view.to_owned(),
        None => panic!(
            "dezero: cannot broadcast an array of shape {:?} to {shape:?}",
            x.shape()
        ),
    }
}

impl Op for BroadcastTo {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = one(xs, "BroadcastTo", "input");
        vec![broadcast_array(x, &self.shape)]
    }

    fn backward(
        &self,
        inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let x = one(inputs, "BroadcastTo", "input");
        let gy = one(gys, "BroadcastTo", "output gradient");
        // Each copy of an element contributed to the output independently, so
        // their gradients add up: that sum is exactly `sum_to`.
        vec![sum_to(gy, &shape_of(x, "BroadcastTo"))]
    }
}

/// Broadcasts a variable up to `shape` — Python's
/// `dezero.functions.broadcast_to`.
///
/// The inverse of [`sum_to`]. numpy's rules apply: axes are matched from the
/// back, and each one must already be equal to the target or be 1. When
/// `shape` already matches, the variable is returned as it is and **no graph
/// node is created**.
///
/// # Examples
///
/// ```
/// use dezero::{broadcast_to, sum_to, Variable};
/// use ndarray::{arr1, arr2};
///
/// let x = Variable::new(arr1(&[1.0, 2.0, 3.0]).into_dyn());
/// let y = broadcast_to(&x, &[2, 3]);
/// assert_eq!(y.data(), Some(arr2(&[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]).into_dyn()));
///
/// // Two copies of each element, so each gradient arrives twice.
/// y.backward();
/// assert_eq!(x.grad().and_then(|g| g.data()), Some(arr1(&[2.0, 2.0, 2.0]).into_dyn()));
///
/// // ... which is what `sum_to` says too.
/// assert_eq!(sum_to(&y, &[3]).data(), Some(arr1(&[2.0, 4.0, 6.0]).into_dyn()));
/// ```
///
/// # Panics
///
/// Panics if `x` does not broadcast to `shape`, or if it holds no data.
#[must_use]
pub fn broadcast_to(x: &Variable, shape: &[usize]) -> Variable {
    if x.shape().as_deref() == Some(shape) {
        return x.clone();
    }
    apply1(BroadcastTo::new(shape), &[x])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::gradient_check;
    use ndarray::{arr0, arr1, arr2, arr3};

    const EPS: f64 = 1e-4;
    const RTOL: f64 = 1e-4;
    const ATOL: f64 = 1e-5;

    fn matrix() -> Variable {
        Variable::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn())
    }

    fn cube() -> Variable {
        Variable::new(
            arr3(&[
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            ])
            .into_dyn(),
        )
    }

    fn data(v: &Variable) -> ArrayD<f64> {
        v.data().expect("variable holds data")
    }

    fn grad(v: &Variable) -> ArrayD<f64> {
        data(&v.grad().expect("variable has a gradient"))
    }

    // -- Axes --------------------------------------------------------------

    #[test]
    fn axes_convert_from_every_python_spelling() {
        assert_eq!(Axes::from(1), Axes::Only(vec![1]));
        assert_eq!(Axes::from([0, 2]), Axes::Only(vec![0, 2]));
        assert_eq!(Axes::from(vec![0, 2]), Axes::Only(vec![0, 2]));
        assert_eq!(Axes::from([0, 2].as_slice()), Axes::Only(vec![0, 2]));
        assert_eq!(Axes::default(), Axes::All);
    }

    #[test]
    fn axes_resolve_negatives_and_sort() {
        assert_eq!(Axes::All.resolve(3), vec![0, 1, 2]);
        assert_eq!(Axes::Only(vec![-1, 0]).resolve(3), vec![0, 2]);
        assert_eq!(Axes::All.resolve(0), Vec::<usize>::new());
        assert_eq!(Axes::All.as_slice(), None);
        assert_eq!(Axes::Only(vec![1]).as_slice(), Some([1].as_slice()));
    }

    #[test]
    #[should_panic(expected = "duplicate axis")]
    fn a_repeated_axis_is_rejected() {
        let _ = Axes::Only(vec![0, -2]).resolve(2);
    }

    // -- sum forward -------------------------------------------------------

    #[test]
    fn sum_over_every_axis_and_keepdims_combination() {
        let x = matrix();
        assert_eq!(data(&sum(&x, Axes::All, false)), arr0(21.0).into_dyn());
        assert_eq!(
            data(&sum(&x, Axes::All, true)),
            arr2(&[[21.0]]).into_dyn(),
            "keepdims keeps every reduced axis as a 1"
        );
        assert_eq!(data(&sum(&x, 0, false)), arr1(&[5.0, 7.0, 9.0]).into_dyn());
        assert_eq!(data(&sum(&x, 1, false)), arr1(&[6.0, 15.0]).into_dyn());
        assert_eq!(data(&sum(&x, 0, true)), arr2(&[[5.0, 7.0, 9.0]]).into_dyn());
        assert_eq!(data(&sum(&x, 1, true)), arr2(&[[6.0], [15.0]]).into_dyn());
        assert_eq!(data(&sum(&x, -1, false)), arr1(&[6.0, 15.0]).into_dyn());
    }

    #[test]
    fn sum_over_several_axes_at_once() {
        let x = cube();
        assert_eq!(
            data(&sum(&x, [0, 2], false)),
            arr1(&[18.0, 26.0, 34.0]).into_dyn()
        );
        assert_eq!(
            data(&sum(&x, [0, 2], true)).shape(),
            &[1, 3, 1],
            "keepdims leaves a 1 at each reduced axis"
        );
    }

    #[test]
    fn sum_of_a_scalar_is_itself() {
        let x = Variable::from_scalar(7.0);
        assert_eq!(data(&sum(&x, Axes::All, false)), arr0(7.0).into_dyn());
        assert_eq!(data(&sum(&x, Axes::All, true)), arr0(7.0).into_dyn());
    }

    // -- sum backward ------------------------------------------------------

    #[test]
    fn sum_backward_spreads_ones_over_the_input() {
        for axis in [Axes::All, Axes::Only(vec![0]), Axes::Only(vec![1])] {
            for keepdims in [false, true] {
                let x = matrix();
                sum(&x, axis.clone(), keepdims).backward();
                assert_eq!(
                    grad(&x),
                    ArrayD::from_elem(IxDyn(&[2, 3]), 1.0),
                    "axis {axis:?}, keepdims {keepdims}"
                );
            }
        }
    }

    /// The case `reshape_sum_backward` exists for: with `keepdims=False` the
    /// gradient of an `axis=1` sum arrives shaped `(2,)`, which would broadcast
    /// across *columns* — the transpose of the right answer — if it were not
    /// reshaped to `(2, 1)` first.
    #[test]
    fn sum_backward_aligns_the_reduced_axis() {
        let x = matrix();
        let weights = Variable::new(arr1(&[10.0, 100.0]).into_dyn());
        sum_all(&(&sum(&x, 1, false) * &weights)).backward();
        assert_eq!(
            grad(&x),
            arr2(&[[10.0, 10.0, 10.0], [100.0, 100.0, 100.0]]).into_dyn()
        );
    }

    #[test]
    fn sum_gradient_matches_numerical_diff() {
        for keepdims in [false, true] {
            gradient_check(|x| sum(x, Axes::All, keepdims), &matrix(), EPS, RTOL, ATOL)
                .expect("sum(all)");
            gradient_check(|x| sum(x, 0, keepdims), &matrix(), EPS, RTOL, ATOL).expect("sum(0)");
            gradient_check(|x| sum(x, 1, keepdims), &matrix(), EPS, RTOL, ATOL).expect("sum(1)");
            gradient_check(|x| sum(x, 1, keepdims), &cube(), EPS, RTOL, ATOL).expect("sum(1) 3-d");
            gradient_check(|x| sum(x, [0, 2], keepdims), &cube(), EPS, RTOL, ATOL)
                .expect("sum(0, 2)");
        }
    }

    /// A weighted sum, so the gradient is not the uninformative all-ones array
    /// that hides an axis mix-up.
    #[test]
    fn weighted_sum_gradient_matches_numerical_diff() {
        let weights = Variable::new(arr1(&[1.0, -2.0, 0.5]).into_dyn());
        gradient_check(
            |x| sum_all(&(&sum(x, 0, false) * &weights)),
            &matrix(),
            EPS,
            RTOL,
            ATOL,
        )
        .expect("weighted sum");

        // The same idea at rank 3, over two axes at once: the surviving axis is
        // the middle one, which only lines up if both reduced axes are put back
        // in the right places.
        let middle = Variable::new(arr1(&[1.0, -2.0, 0.5]).into_dyn());
        gradient_check(
            |x| sum_all(&(&sum(x, [0, 2], false) * &middle)),
            &cube(),
            EPS,
            RTOL,
            ATOL,
        )
        .expect("weighted 3-d sum over two axes");
    }

    #[test]
    fn the_method_form_delegates_to_the_function() {
        let x = matrix();
        assert_eq!(x.sum(Axes::All, false).data(), sum_all(&x).data());
        assert_eq!(x.sum(1, true).data(), sum(&x, 1, true).data());
    }

    // -- broadcast_to ------------------------------------------------------

    #[test]
    fn broadcast_to_repeats_along_the_missing_axes() {
        let row = Variable::new(arr1(&[1.0, 2.0, 3.0]).into_dyn());
        assert_eq!(
            data(&broadcast_to(&row, &[2, 3])),
            arr2(&[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]).into_dyn()
        );

        let column = Variable::new(arr2(&[[1.0], [2.0]]).into_dyn());
        assert_eq!(
            data(&broadcast_to(&column, &[2, 3])),
            arr2(&[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]).into_dyn()
        );

        let scalar = Variable::from_scalar(5.0);
        assert_eq!(
            data(&broadcast_to(&scalar, &[2, 2])),
            arr2(&[[5.0, 5.0], [5.0, 5.0]]).into_dyn()
        );
    }

    #[test]
    fn broadcast_to_the_same_shape_builds_no_node() {
        let x = matrix();
        let y = broadcast_to(&x, &[2, 3]);
        assert_eq!(y.id(), x.id(), "Python returns the very same variable");
        assert!(y.creator().is_none());
    }

    #[test]
    fn broadcast_to_backward_counts_the_copies() {
        let x = Variable::new(arr1(&[1.0, 2.0, 3.0]).into_dyn());
        broadcast_to(&x, &[4, 3]).backward();
        assert_eq!(grad(&x), arr1(&[4.0, 4.0, 4.0]).into_dyn());
    }

    #[test]
    fn broadcast_to_gradient_matches_numerical_diff() {
        let row = Variable::new(arr1(&[1.0, 2.0, 3.0]).into_dyn());
        gradient_check(|x| broadcast_to(x, &[2, 3]), &row, EPS, RTOL, ATOL).expect("row");

        let column = Variable::new(arr2(&[[1.0], [2.0]]).into_dyn());
        gradient_check(|x| broadcast_to(x, &[2, 3]), &column, EPS, RTOL, ATOL).expect("column");

        let scalar = Variable::from_scalar(3.0);
        gradient_check(|x| broadcast_to(x, &[2, 3, 4]), &scalar, EPS, RTOL, ATOL).expect("scalar");
    }

    #[test]
    #[should_panic(expected = "cannot broadcast an array of shape [2, 3] to [3, 3]")]
    fn broadcast_to_rejects_an_incompatible_shape() {
        let _ = broadcast_to(&matrix(), &[3, 3]);
    }

    // -- sum_to ------------------------------------------------------------

    #[test]
    fn sum_to_reduces_to_the_requested_shape() {
        let x = matrix();
        assert_eq!(
            data(&sum_to(&x, &[1, 3])),
            arr2(&[[5.0, 7.0, 9.0]]).into_dyn()
        );
        assert_eq!(
            data(&sum_to(&x, &[2, 1])),
            arr2(&[[6.0], [15.0]]).into_dyn()
        );
        assert_eq!(data(&sum_to(&x, &[3])), arr1(&[5.0, 7.0, 9.0]).into_dyn());
        assert_eq!(data(&sum_to(&x, &[])), arr0(21.0).into_dyn());
    }

    #[test]
    fn sum_to_the_same_shape_builds_no_node() {
        let x = matrix();
        let y = sum_to(&x, &[2, 3]);
        assert_eq!(y.id(), x.id());
        assert!(y.creator().is_none());
    }

    #[test]
    fn sum_to_backward_broadcasts_the_gradient() {
        let x = matrix();
        let weights = Variable::new(arr2(&[[1.0, 10.0, 100.0]]).into_dyn());
        sum_all(&(&sum_to(&x, &[1, 3]) * &weights)).backward();
        assert_eq!(
            grad(&x),
            arr2(&[[1.0, 10.0, 100.0], [1.0, 10.0, 100.0]]).into_dyn()
        );
    }

    #[test]
    fn sum_to_gradient_matches_numerical_diff() {
        gradient_check(|x| sum_to(x, &[1, 3]), &matrix(), EPS, RTOL, ATOL).expect("to row");
        gradient_check(|x| sum_to(x, &[2, 1]), &matrix(), EPS, RTOL, ATOL).expect("to column");
        gradient_check(|x| sum_to(x, &[]), &matrix(), EPS, RTOL, ATOL).expect("to scalar");
        gradient_check(|x| sum_to(x, &[3, 2]), &cube(), EPS, RTOL, ATOL).expect("3-d to 2-d");
    }

    // -- the two are inverses ---------------------------------------------

    /// `sum_to(broadcast_to(x, big), x.shape)` scales `x` by the number of
    /// copies, and the gradient counts them the same way. Getting either
    /// direction wrong survives every shape check and only shows up here.
    #[test]
    fn broadcast_to_then_sum_to_round_trips() {
        for (small, big) in [
            (vec![3_usize], vec![4_usize, 3]),
            (vec![2, 1], vec![2, 3]),
            (vec![], vec![2, 3]),
            (vec![1, 3, 1], vec![2, 3, 4]),
        ] {
            let size: usize = small.iter().product();
            #[allow(
                clippy::cast_precision_loss,
                reason = "the shapes here hold at most a few dozen elements"
            )]
            let values: Vec<f64> = (0..size).map(|v| 1.0 + v as f64).collect();
            let x = Variable::new(
                ArrayD::from_shape_vec(IxDyn(&small), values).expect("shape matches"),
            );

            let wide = broadcast_to(&x, &big);
            let back = sum_to(&wide, &small);

            #[allow(
                clippy::cast_precision_loss,
                reason = "the shapes here hold at most a few dozen elements"
            )]
            let copies = (big.iter().product::<usize>() / size) as f64;

            assert_eq!(
                back.data(),
                Some(data(&x).mapv(|v| v * copies)),
                "{small:?} -> {big:?} -> {small:?}"
            );

            // d(sum_to(broadcast_to(x)))/dx: the seed of ones is broadcast up
            // and summed back down, which counts the copies once.
            back.backward();
            assert_eq!(
                grad(&x),
                ArrayD::from_elem(IxDyn(&small), copies),
                "gradient of the round trip, {small:?} -> {big:?}"
            );
        }
    }

    /// The same round trip in the other order: summing down and broadcasting
    /// back must land on the original shape with each entry repeated.
    #[test]
    fn sum_to_then_broadcast_to_round_trips() {
        let x = matrix();
        let y = broadcast_to(&sum_to(&x, &[1, 3]), &[2, 3]);
        assert_eq!(
            data(&y),
            arr2(&[[5.0, 7.0, 9.0], [5.0, 7.0, 9.0]]).into_dyn()
        );
        y.backward();
        assert_eq!(grad(&x), ArrayD::from_elem(IxDyn(&[2, 3]), 2.0));
    }

    /// `sum` is `sum_to` with the reduced axes dropped, so the two must agree
    /// on both the value and the gradient.
    #[test]
    fn sum_and_sum_to_agree_where_they_overlap() {
        let a = matrix();
        let b = matrix();
        let via_sum = sum(&a, 0, true);
        let via_sum_to = sum_to(&b, &[1, 3]);
        assert_eq!(via_sum.data(), via_sum_to.data());

        via_sum.backward();
        via_sum_to.backward();
        assert_eq!(grad(&a), grad(&b));
    }

    // -- higher-order ------------------------------------------------------

    /// Both backwards are built from `Variable` ops, so a gradient that went
    /// through them can be differentiated again (`docs/ARCHITECTURE.md`'s
    /// critical rule).
    #[test]
    fn broadcasting_gradients_stay_differentiable() {
        // y = sum(x^3) -> dy/dx = 3x^2 -> d2y/dx2 = 6x.
        let x = Variable::new(arr1(&[1.0, 2.0, 3.0]).into_dyn());
        let y = sum_all(&crate::pow(&x, 3.0));
        y.backward_with(false, true);

        let gx = x.grad().expect("first derivative");
        assert_eq!(data(&gx), arr1(&[3.0, 12.0, 27.0]).into_dyn());
        assert!(gx.creator().is_some(), "the gradient is a graph node");

        x.cleargrad();
        gx.backward();
        assert_eq!(grad(&x), arr1(&[6.0, 12.0, 18.0]).into_dyn());
    }
}
