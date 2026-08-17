//! Array-level helpers that live below the graph.
//!
//! Two groups, both needed by the neural-network layer of steps 41–48:
//!
//! * [`max_keepdims`], [`sum_keepdims`] and [`logsumexp`] — the reduction
//!   idioms numpy spells `x.max(axis=..., keepdims=True)`. `ndarray` reduces one
//!   axis at a time and always drops it, so "keep the axis as a 1" is written
//!   out here once instead of at every call site. [`logsumexp`] is a
//!   statement-by-statement port of `vendor/dezero-python/dezero/utils.py`.
//! * [`gather_rows`] and [`scatter_add_rows`] — numpy's *fancy indexing*,
//!   `log_p[np.arange(N), t]`, which `ndarray` does not have at all. They are
//!   exact adjoints of one another: `scatter_add_rows` distributes back
//!   precisely what `gather_rows` picked out, which is what makes
//!   [`softmax_cross_entropy`](crate::softmax_cross_entropy) differentiable.
//!
//! Everything here takes and returns plain `ArrayD`, so it belongs in an
//! `Op::forward` (or in an op's construction of a *constant*), never in the
//! `Variable` arithmetic of an `Op::backward`.

use ndarray::{ArrayD, Axis, Ix2, IxDyn};

/// Reduces `axis` with `max`, keeping it as a length-1 axis — numpy's
/// `x.max(axis=axis, keepdims=True)`.
///
/// # Examples
///
/// ```
/// use dezero::utils::array::max_keepdims;
/// use ndarray::arr2;
///
/// let x = arr2(&[[1.0, 5.0, 3.0], [9.0, 2.0, 4.0]]).into_dyn();
/// assert_eq!(max_keepdims(&x, 1), arr2(&[[5.0], [9.0]]).into_dyn());
/// ```
///
/// # Panics
///
/// Panics if `axis` is out of bounds for `x`, or if `x` is empty along `axis`
/// (numpy raises `ValueError: zero-size array to reduction operation`).
#[must_use]
pub fn max_keepdims(x: &ArrayD<f64>, axis: usize) -> ArrayD<f64> {
    assert!(
        axis < x.ndim(),
        "dezero: axis {axis} is out of bounds for an array of shape {:?}",
        x.shape()
    );
    assert!(
        x.shape()[axis] > 0,
        "dezero: cannot take the maximum over empty axis {axis} of shape {:?}",
        x.shape()
    );
    x.map_axis(Axis(axis), |lane| {
        lane.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    })
    .insert_axis(Axis(axis))
}

/// Reduces `axis` with a sum, keeping it as a length-1 axis — numpy's
/// `x.sum(axis=axis, keepdims=True)`.
///
/// # Examples
///
/// ```
/// use dezero::utils::array::sum_keepdims;
/// use ndarray::arr2;
///
/// let x = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn();
/// assert_eq!(sum_keepdims(&x, 1), arr2(&[[6.0], [15.0]]).into_dyn());
/// ```
///
/// # Panics
///
/// Panics if `axis` is out of bounds for `x`.
#[must_use]
pub fn sum_keepdims(x: &ArrayD<f64>, axis: usize) -> ArrayD<f64> {
    assert!(
        axis < x.ndim(),
        "dezero: axis {axis} is out of bounds for an array of shape {:?}",
        x.shape()
    );
    x.sum_axis(Axis(axis)).insert_axis(Axis(axis))
}

/// `log(sum(exp(x), axis))`, computed without overflowing.
///
/// Statement-for-statement port of `dezero.utils.logsumexp`: subtract the
/// per-lane maximum before exponentiating, then add it back. The result keeps
/// `axis` as a length-1 axis so it broadcasts straight back onto `x`.
///
/// The shift is what makes this usable on raw logits — `exp(1000.0)` is `inf`,
/// while `exp(1000.0 - 1000.0)` is 1.
///
/// # Examples
///
/// ```
/// use dezero::utils::array::logsumexp;
/// use ndarray::arr2;
///
/// // Huge logits: the naive formula would overflow to `inf`.
/// let x = arr2(&[[1000.0, 1000.0]]).into_dyn();
/// let y = logsumexp(&x, 1);
/// assert_eq!(y.shape(), &[1, 1]);
/// assert!((y[[0, 0]] - (1000.0 + 2.0_f64.ln())).abs() < 1e-9);
/// ```
///
/// # Panics
///
/// Panics if `axis` is out of bounds for `x`, or if `x` is empty along `axis`.
#[must_use]
pub fn logsumexp(x: &ArrayD<f64>, axis: usize) -> ArrayD<f64> {
    let mut m = max_keepdims(x, axis);
    let mut y = x.clone();
    y -= &m; // broadcasts `m` back over `axis`
    y.mapv_inplace(f64::exp);
    let mut s = sum_keepdims(&y, axis);
    s.mapv_inplace(f64::ln);
    m += &s;
    m
}

/// Picks one element out of every row — numpy's `x[np.arange(n), indices]`.
///
/// `x` is 2-dimensional and `indices` has one column index per row; the result
/// is 1-dimensional with `x.shape()[0]` elements. `ndarray` has no fancy
/// indexing, so this is the hand-written stand-in (see `docs/ARCHITECTURE.md`,
/// "Fancy indexing").
///
/// [`scatter_add_rows`] is its adjoint.
///
/// # Examples
///
/// ```
/// use dezero::utils::array::gather_rows;
/// use ndarray::{arr1, arr2};
///
/// let x = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn();
/// assert_eq!(gather_rows(&x, &[2, 0]), arr1(&[3.0, 4.0]).into_dyn());
/// ```
///
/// # Panics
///
/// Panics if `x` is not 2-dimensional, if `indices` does not have one entry per
/// row of `x`, or if an index is not a valid column of `x` (numpy's
/// `IndexError`).
#[must_use]
pub fn gather_rows(x: &ArrayD<f64>, indices: &[usize]) -> ArrayD<f64> {
    let Ok(x) = x.view().into_dimensionality::<Ix2>() else {
        panic!(
            "dezero: gather needs a 2-dimensional array, got shape {:?}",
            x.shape()
        );
    };
    let (rows, columns) = x.dim();
    assert!(
        indices.len() == rows,
        "dezero: gather got {} indices for {rows} rows",
        indices.len()
    );

    let picked: Vec<f64> = indices
        .iter()
        .enumerate()
        .map(|(row, &column)| {
            assert!(
                column < columns,
                "dezero: gather index {column} at row {row} is out of bounds for {columns} columns"
            );
            x[[row, column]]
        })
        .collect();

    ArrayD::from_shape_vec(IxDyn(&[rows]), picked)
        .expect("the buffer holds exactly one value per row")
}

/// Scatters one value back into every row — the adjoint of [`gather_rows`].
///
/// Returns a `(values.len(), columns)` array that is zero everywhere except at
/// `[row, indices[row]]`, where it holds `values[row]`. Repeated indices *add*,
/// which is what makes this the true adjoint rather than merely an assignment.
///
/// With `values` all ones this is numpy's `np.eye(columns)[t]`, the one-hot
/// encoding [`softmax_cross_entropy`](crate::softmax_cross_entropy) subtracts
/// in its backward pass.
///
/// # Examples
///
/// ```
/// use dezero::utils::array::{gather_rows, scatter_add_rows};
/// use ndarray::{arr1, arr2};
///
/// let g = arr1(&[7.0, 9.0]).into_dyn();
/// assert_eq!(
///     scatter_add_rows(&[2, 0], &g, 3),
///     arr2(&[[0.0, 0.0, 7.0], [9.0, 0.0, 0.0]]).into_dyn()
/// );
///
/// // Adjoint: <gather(x), g> == <x, scatter(g)> for every x.
/// let x = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn();
/// let left: f64 = (&gather_rows(&x, &[2, 0]) * &g).sum();
/// let right: f64 = (&x * &scatter_add_rows(&[2, 0], &g, 3)).sum();
/// assert!((left - right).abs() < 1e-12);
/// ```
///
/// # Panics
///
/// Panics if `values` is not 1-dimensional, if it does not have one entry per
/// index, or if an index is out of bounds for `columns`.
#[must_use]
pub fn scatter_add_rows(indices: &[usize], values: &ArrayD<f64>, columns: usize) -> ArrayD<f64> {
    assert!(
        values.ndim() == 1,
        "dezero: scatter needs 1-dimensional values, got shape {:?}",
        values.shape()
    );
    assert!(
        values.len() == indices.len(),
        "dezero: scatter got {} values for {} indices",
        values.len(),
        indices.len()
    );

    let mut out = ArrayD::zeros(IxDyn(&[indices.len(), columns]));
    for (row, (&column, &value)) in indices.iter().zip(values.iter()).enumerate() {
        assert!(
            column < columns,
            "dezero: scatter index {column} at row {row} is out of bounds for {columns} columns"
        );
        out[[row, column]] += value;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, arr3};

    fn matrix() -> ArrayD<f64> {
        arr2(&[[1.0, 5.0, 3.0], [9.0, 2.0, 4.0]]).into_dyn()
    }

    #[test]
    fn max_keepdims_reduces_the_named_axis_only() {
        assert_eq!(
            max_keepdims(&matrix(), 0),
            arr2(&[[9.0, 5.0, 4.0]]).into_dyn()
        );
        assert_eq!(max_keepdims(&matrix(), 1), arr2(&[[5.0], [9.0]]).into_dyn());
    }

    #[test]
    fn max_keepdims_works_at_rank_three() {
        let x = arr3(&[[[1.0, 2.0], [3.0, 4.0]], [[8.0, 7.0], [6.0, 5.0]]]).into_dyn();
        let y = max_keepdims(&x, 2);
        assert_eq!(y.shape(), &[2, 2, 1]);
        assert_eq!(y[[1, 0, 0]], 8.0);
    }

    #[test]
    fn max_keepdims_handles_all_negative_lanes() {
        // The fold seeds with -inf, so an all-negative lane must still come out
        // as its own largest element rather than 0.
        let x = arr2(&[[-3.0, -1.0, -2.0]]).into_dyn();
        assert_eq!(max_keepdims(&x, 1), arr2(&[[-1.0]]).into_dyn());
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn max_keepdims_rejects_a_bad_axis() {
        let _ = max_keepdims(&matrix(), 2);
    }

    #[test]
    fn sum_keepdims_matches_numpy() {
        let x = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn();
        assert_eq!(sum_keepdims(&x, 0), arr2(&[[5.0, 7.0, 9.0]]).into_dyn());
        assert_eq!(sum_keepdims(&x, 1), arr2(&[[6.0], [15.0]]).into_dyn());
    }

    #[test]
    fn logsumexp_matches_the_direct_formula_on_tame_input() {
        let x = arr2(&[[0.5, -1.0, 2.0], [1.0, 1.0, 1.0]]).into_dyn();
        let actual = logsumexp(&x, 1);
        assert_eq!(actual.shape(), &[2, 1]);

        for row in 0..2 {
            let direct: f64 = (0..3).map(|c| x[[row, c]].exp()).sum::<f64>().ln();
            assert!((actual[[row, 0]] - direct).abs() < 1e-12, "row {row}");
        }
    }

    /// The shift is the whole point: the direct formula overflows here.
    #[test]
    fn logsumexp_survives_logits_that_would_overflow() {
        let x = arr2(&[[800.0, 800.0, 800.0]]).into_dyn();
        let y = logsumexp(&x, 1);
        assert!(y[[0, 0]].is_finite());
        assert!((y[[0, 0]] - (800.0 + 3.0_f64.ln())).abs() < 1e-9);
        assert!(
            x.mapv(f64::exp).sum().is_infinite(),
            "the unshifted computation really does overflow"
        );
    }

    #[test]
    fn logsumexp_over_axis_zero() {
        let x = arr2(&[[0.0], [0.0]]).into_dyn();
        let y = logsumexp(&x, 0);
        assert_eq!(y.shape(), &[1, 1]);
        assert!((y[[0, 0]] - 2.0_f64.ln()).abs() < 1e-15);
    }

    #[test]
    fn gather_picks_one_element_per_row() {
        let x = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).into_dyn();
        assert_eq!(
            gather_rows(&x, &[0, 2, 1]),
            arr1(&[1.0, 6.0, 8.0]).into_dyn()
        );
    }

    #[test]
    #[should_panic(expected = "out of bounds for 3 columns")]
    fn gather_rejects_an_index_past_the_last_column() {
        let x = arr2(&[[1.0, 2.0, 3.0]]).into_dyn();
        let _ = gather_rows(&x, &[3]);
    }

    #[test]
    #[should_panic(expected = "1 indices for 2 rows")]
    fn gather_rejects_a_short_index_list() {
        let _ = gather_rows(&matrix(), &[0]);
    }

    #[test]
    #[should_panic(expected = "2-dimensional")]
    fn gather_rejects_a_non_matrix() {
        let _ = gather_rows(&arr1(&[1.0, 2.0]).into_dyn(), &[0]);
    }

    #[test]
    fn scatter_with_ones_is_a_one_hot_encoding() {
        let ones = ArrayD::from_elem(IxDyn(&[3]), 1.0);
        assert_eq!(
            scatter_add_rows(&[0, 2, 1], &ones, 3),
            arr2(&[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]).into_dyn()
        );
    }

    /// `scatter_add_rows` is the adjoint of `gather_rows`, i.e.
    /// `<gather(x), g> == <x, scatter(g)>` for every `x` and `g`. Getting the
    /// row/column order backwards passes every shape check and fails this.
    #[test]
    fn scatter_is_the_adjoint_of_gather() {
        let x = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]).into_dyn();
        let g = arr1(&[0.5, -2.0, 3.0]).into_dyn();
        for indices in [[0_usize, 1, 2], [2, 2, 0], [1, 0, 1]] {
            let left: f64 = (&gather_rows(&x, &indices) * &g).sum();
            let right: f64 = (&x * &scatter_add_rows(&indices, &g, 3)).sum();
            assert!((left - right).abs() < 1e-12, "indices {indices:?}");
        }
    }

    #[test]
    #[should_panic(expected = "2 values for 1 indices")]
    fn scatter_rejects_a_length_mismatch() {
        let _ = scatter_add_rows(&[0], &arr1(&[1.0, 2.0]).into_dyn(), 3);
    }

    #[test]
    #[should_panic(expected = "1-dimensional values")]
    fn scatter_rejects_non_vector_values() {
        let _ = scatter_add_rows(&[0], &arr2(&[[1.0]]).into_dyn(), 3);
    }
}
