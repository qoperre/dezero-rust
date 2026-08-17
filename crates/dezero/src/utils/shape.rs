//! Shape arithmetic: the numpy rules broadcasting is built out of.
//!
//! Port of the "Utility functions for numpy (numpy magic)" section of
//! `vendor/dezero-python/dezero/utils.py` — [`sum_to`] and
//! [`reshape_sum_backward`] — plus [`broadcast_shape`], which numpy provides as
//! `np.broadcast_shapes` and `ndarray` does not.
//!
//! Everything here is *array* level: shapes in, shapes out, no graph. The
//! differentiable `Variable`-level wrappers live in
//! [`crate::functions::reduce`].
//!
//! These functions are short and unusually load-bearing: a wrong axis here
//! produces a gradient of the right *shape* but the wrong *values*, which no
//! type can catch. They are ported statement by statement from the reference
//! and pinned by the widest fixture matrix in the suite.

use std::borrow::Cow;

use ndarray::{ArrayD, Axis};

/// Resolves a possibly negative axis index against a rank — numpy's
/// `normalize_axis_index`.
///
/// # Examples
///
/// ```
/// use dezero::utils::shape::normalize_axis;
///
/// assert_eq!(normalize_axis(0, 3), 0);
/// assert_eq!(normalize_axis(-1, 3), 2);
/// ```
///
/// # Panics
///
/// Panics if the axis is out of bounds for `ndim`, as numpy raises
/// `AxisError`.
#[must_use]
pub fn normalize_axis(axis: isize, ndim: usize) -> usize {
    let rank = isize::try_from(ndim).ok();
    let resolved = if axis < 0 {
        rank.and_then(|rank| axis.checked_add(rank))
    } else {
        Some(axis)
    };

    match resolved.and_then(|a| usize::try_from(a).ok()) {
        Some(a) if a < ndim => a,
        _ => panic!("dezero: axis {axis} is out of bounds for an array of {ndim} dimensions"),
    }
}

/// The shape two operands broadcast to, or `None` if they do not broadcast.
///
/// numpy's rule, which `ndarray` does not expose: line the shapes up at their
/// *trailing* axis, pad the shorter one with leading 1s, and take the larger of
/// each pair — every pair must be equal or contain a 1.
///
/// # Examples
///
/// ```
/// use dezero::utils::shape::broadcast_shape;
///
/// assert_eq!(broadcast_shape(&[2, 3], &[3]), Some(vec![2, 3]));
/// assert_eq!(broadcast_shape(&[3], &[2, 3]), Some(vec![2, 3]));
/// assert_eq!(broadcast_shape(&[2, 1], &[1, 3]), Some(vec![2, 3]));
/// assert_eq!(broadcast_shape(&[2, 2], &[2, 3]), None);
/// ```
#[must_use]
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut shape = Vec::with_capacity(rank);

    for i in 0..rank {
        // Count back from the end of each shape; a missing axis reads as 1.
        let da = a.len().checked_sub(rank - i).map_or(1, |k| a[k]);
        let db = b.len().checked_sub(rank - i).map_or(1, |k| b[k]);
        match (da, db) {
            (1, d) | (d, 1) => shape.push(d),
            (d, e) if d == e => shape.push(d),
            _ => return None,
        }
    }

    Some(shape)
}

/// Sums `x` down to `shape`, the inverse of broadcasting *to* `shape`.
///
/// Line-by-line port of `utils.sum_to`:
///
/// ```text
/// ndim = len(shape)
/// lead = x.ndim - ndim
/// lead_axis = tuple(range(lead))
/// axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
/// y = x.sum(lead_axis + axis, keepdims=True)
/// if lead > 0:
///     y = y.squeeze(lead_axis)
/// ```
///
/// The two-stage shape is the whole subtlety: axes that only *exist* in `x`
/// (the `lead` ones) are summed away entirely, while axes that exist in both
/// but are 1 in `shape` are summed with `keepdims` so they stay as 1.
///
/// `ndarray` has no multi-axis reduction, so the `keepdims=True` sum is spelled
/// as a descending loop of `sum_axis` + `insert_axis`. Descending order keeps
/// the not-yet-visited (smaller) axis indices valid.
///
/// # Examples
///
/// ```
/// use dezero::utils::shape::sum_to;
/// use ndarray::arr2;
///
/// let x = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn();
/// assert_eq!(sum_to(&x, &[1, 3]), arr2(&[[5.0, 7.0, 9.0]]).into_dyn());
/// assert_eq!(sum_to(&x, &[2, 1]), arr2(&[[6.0], [15.0]]).into_dyn());
/// ```
///
/// # Panics
///
/// Panics if `x` cannot be summed to `shape`: either `shape` has more
/// dimensions than `x`, or a non-1 entry of `shape` disagrees with the matching
/// axis of `x`. Python leaves both cases to numpy — the first raises, the
/// second silently returns an array of the wrong shape. Since a silently wrong
/// gradient is exactly the hazard this module exists to prevent, the port
/// checks instead.
#[must_use]
pub fn sum_to(x: &ArrayD<f64>, shape: &[usize]) -> ArrayD<f64> {
    let ndim = shape.len();
    assert!(
        x.ndim() >= ndim,
        "dezero: cannot sum an array of shape {:?} down to {shape:?}, which has more dimensions",
        x.shape()
    );
    let lead = x.ndim() - ndim;

    for (i, &sx) in shape.iter().enumerate() {
        assert!(
            sx == 1 || sx == x.shape()[i + lead],
            "dezero: cannot sum an array of shape {:?} down to {shape:?}; \
             axis {i} of the target is neither 1 nor {}",
            x.shape(),
            x.shape()[i + lead]
        );
    }

    // `lead_axis + axis`, already ascending and duplicate-free by construction.
    let mut axes: Vec<usize> = (0..lead).collect();
    axes.extend(
        shape
            .iter()
            .enumerate()
            .filter(|&(_, &sx)| sx == 1)
            .map(|(i, _)| i + lead),
    );

    // x.sum(axes, keepdims=True)
    let mut y = Cow::Borrowed(x);
    for &axis in axes.iter().rev() {
        y = Cow::Owned(y.sum_axis(Axis(axis)).insert_axis(Axis(axis)));
    }

    // y.squeeze(lead_axis): every leading axis is 1 after the keepdims sum.
    let mut y = y.into_owned();
    for _ in 0..lead {
        y = y.index_axis_move(Axis(0), 0);
    }
    y
}

/// The shape a `Sum`'s output gradient must be reshaped to before it is
/// broadcast back over the input.
///
/// Port of the shape computation in `utils.reshape_sum_backward`; the reshape
/// itself is done by the caller, on a `Variable`, so that it stays in the
/// graph.
///
/// A `sum` with `keepdims=False` *drops* the reduced axes, so its gradient
/// arrives with those axes missing and cannot be broadcast back over the input
/// — `sum(x(2, 3), axis=0)` gives a gradient of shape `(3,)`, which would
/// broadcast over rows rather than columns for `axis=1`. Re-inserting the
/// reduced axes as 1s restores the alignment. Nothing needs re-inserting when
/// the input was 0-d, when every axis was summed (`axis=None`), or when
/// `keepdims` already kept them.
///
/// `axis` is `None` for Python's `axis=None`; negative entries count from the
/// end, as in numpy.
///
/// # Examples
///
/// ```
/// use dezero::utils::shape::reshape_sum_backward;
///
/// // sum(x(2, 3), axis=1) -> gy(2,), which must become (2, 1).
/// assert_eq!(reshape_sum_backward(&[2], &[2, 3], Some(&[1]), false), vec![2, 1]);
/// // keepdims already did it.
/// assert_eq!(reshape_sum_backward(&[2, 1], &[2, 3], Some(&[1]), true), vec![2, 1]);
/// // axis=None reduces everything; the gradient broadcasts as it is.
/// assert_eq!(reshape_sum_backward(&[], &[2, 3], None, false), Vec::<usize>::new());
/// ```
///
/// # Panics
///
/// Panics if an entry of `axis` is out of bounds for `x_shape`.
#[must_use]
pub fn reshape_sum_backward(
    gy_shape: &[usize],
    x_shape: &[usize],
    axis: Option<&[isize]>,
    keepdims: bool,
) -> Vec<usize> {
    let ndim = x_shape.len();

    // Python: `if not (ndim == 0 or tupled_axis is None or keepdims)`.
    let Some(axis) = axis.filter(|_| ndim != 0 && !keepdims) else {
        return gy_shape.to_vec();
    };

    let mut actual_axis: Vec<usize> = axis.iter().map(|&a| normalize_axis(a, ndim)).collect();
    actual_axis.sort_unstable();

    let mut shape = gy_shape.to_vec();
    for a in actual_axis {
        shape.insert(a, 1);
    }
    shape
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn, arr0, arr1, arr2, arr3};

    /// `0, 1, 2, ...` in the given shape, so a wrong axis cannot pass by luck.
    fn ramp(shape: &[usize]) -> ArrayD<f64> {
        let n = shape.iter().product();
        #[allow(
            clippy::cast_precision_loss,
            reason = "the test shapes hold at most a few dozen elements"
        )]
        let values: Vec<f64> = (0..n).map(|v| v as f64).collect();
        ArrayD::from_shape_vec(IxDyn(shape), values).expect("shape matches the element count")
    }

    // -- normalize_axis ----------------------------------------------------

    #[test]
    fn negative_axes_count_from_the_end() {
        assert_eq!(normalize_axis(-1, 3), 2);
        assert_eq!(normalize_axis(-3, 3), 0);
        assert_eq!(normalize_axis(2, 3), 2);
    }

    #[test]
    #[should_panic(expected = "axis 3 is out of bounds")]
    fn axis_past_the_end_is_rejected() {
        let _ = normalize_axis(3, 3);
    }

    #[test]
    #[should_panic(expected = "axis -4 is out of bounds")]
    fn axis_before_the_start_is_rejected() {
        let _ = normalize_axis(-4, 3);
    }

    #[test]
    #[should_panic(expected = "out of bounds for an array of 0 dimensions")]
    fn a_scalar_has_no_axes() {
        let _ = normalize_axis(0, 0);
    }

    // -- broadcast_shape ---------------------------------------------------

    #[test]
    fn broadcasting_is_symmetric() {
        // The property `ndarray`'s own operators lack: which side is bigger
        // must not matter.
        for (a, b) in [
            (vec![2, 3], vec![3]),
            (vec![2, 3], vec![2, 1]),
            (vec![2, 3, 4], vec![3, 4]),
            (vec![1, 3, 1], vec![2, 1, 4]),
            (vec![2, 3], vec![]),
        ] {
            let forward = broadcast_shape(&a, &b);
            let reverse = broadcast_shape(&b, &a);
            assert_eq!(forward, reverse, "{a:?} vs {b:?}");
            assert!(forward.is_some(), "{a:?} and {b:?} should broadcast");
        }
    }

    #[test]
    fn broadcast_shape_takes_the_larger_of_each_axis() {
        assert_eq!(broadcast_shape(&[1, 3, 1], &[2, 1, 4]), Some(vec![2, 3, 4]));
        assert_eq!(broadcast_shape(&[], &[2, 3]), Some(vec![2, 3]));
        assert_eq!(broadcast_shape(&[], &[]), Some(vec![]));
    }

    #[test]
    fn incompatible_shapes_do_not_broadcast() {
        assert_eq!(broadcast_shape(&[2, 2], &[2, 3]), None);
        assert_eq!(broadcast_shape(&[3], &[2]), None);
        assert_eq!(broadcast_shape(&[2, 3, 4], &[3, 5]), None);
    }

    #[test]
    fn a_zero_length_axis_only_broadcasts_against_one() {
        // numpy: (0,) and (1,) broadcast to (0,); (0,) and (2,) do not.
        assert_eq!(broadcast_shape(&[0], &[1]), Some(vec![0]));
        assert_eq!(broadcast_shape(&[0], &[2]), None);
    }

    // -- sum_to ------------------------------------------------------------

    #[test]
    fn sum_to_reduces_rows_and_columns() {
        let x = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn();
        assert_eq!(sum_to(&x, &[1, 3]), arr2(&[[5.0, 7.0, 9.0]]).into_dyn());
        assert_eq!(sum_to(&x, &[2, 1]), arr2(&[[6.0], [15.0]]).into_dyn());
        assert_eq!(sum_to(&x, &[1, 1]), arr2(&[[21.0]]).into_dyn());
    }

    #[test]
    fn sum_to_drops_the_lead_axes_entirely() {
        // (2, 3, 4) -> (3, 4): axis 0 is summed away, not kept as a 1.
        let x = ramp(&[2, 3, 4]);
        let y = sum_to(&x, &[3, 4]);
        assert_eq!(y.shape(), &[3, 4]);
        for i in 0..3 {
            for j in 0..4 {
                assert_eq!(y[[i, j]], x[[0, i, j]] + x[[1, i, j]]);
            }
        }
    }

    #[test]
    fn sum_to_mixes_lead_axes_and_kept_ones() {
        // (2, 3, 4) -> (1, 4): axis 0 vanishes, axis 1 is summed but kept.
        let x = ramp(&[2, 3, 4]);
        let y = sum_to(&x, &[1, 4]);
        assert_eq!(y.shape(), &[1, 4]);
        assert!((y.sum() - x.sum()).abs() < 1e-12);
        for j in 0..4 {
            let expected: f64 = (0..2)
                .map(|i| (0..3).map(|k| x[[i, k, j]]).sum::<f64>())
                .sum();
            assert!((y[[0, j]] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn sum_to_the_same_shape_copies() {
        let x = arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn();
        assert_eq!(sum_to(&x, &[2, 2]), x);
        assert_eq!(sum_to(&arr0(7.0).into_dyn(), &[]), arr0(7.0).into_dyn());
    }

    #[test]
    fn sum_to_a_scalar_removes_every_axis() {
        let x = arr1(&[1.0, 2.0, 3.0]).into_dyn();
        assert_eq!(sum_to(&x, &[]), arr0(6.0).into_dyn());
        assert_eq!(sum_to(&x, &[1]), arr1(&[6.0]).into_dyn());
    }

    #[test]
    fn sum_to_inverts_broadcasting() {
        // Broadcasting an array up and summing it back down must scale it by
        // the number of copies -- the identity `SumTo`/`BroadcastTo` rest on.
        let x = arr1(&[1.0, 2.0, 3.0]).into_dyn();
        let wide = x
            .broadcast(IxDyn(&[4, 3]))
            .expect("(3,) broadcasts to (4, 3)")
            .to_owned();
        assert_eq!(sum_to(&wide, &[3]), arr1(&[4.0, 8.0, 12.0]).into_dyn());
    }

    #[test]
    fn sum_to_matches_a_hand_summed_three_dimensional_case() {
        let x = arr3(&[
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ])
        .into_dyn();
        // (3, 2, 2) -> (1, 2): axis 0 away, axis 1 summed and kept.
        assert_eq!(sum_to(&x, &[1, 2]), arr2(&[[36.0, 42.0]]).into_dyn());
    }

    #[test]
    #[should_panic(expected = "which has more dimensions")]
    fn sum_to_rejects_a_wider_target() {
        let _ = sum_to(&arr1(&[1.0, 2.0]).into_dyn(), &[1, 2]);
    }

    #[test]
    #[should_panic(expected = "is neither 1 nor")]
    fn sum_to_rejects_a_mismatched_axis() {
        let _ = sum_to(&arr2(&[[1.0, 2.0, 3.0]]).into_dyn(), &[1, 2]);
    }

    // -- reshape_sum_backward ----------------------------------------------

    #[test]
    fn reshape_sum_backward_reinserts_the_reduced_axes() {
        assert_eq!(
            reshape_sum_backward(&[3], &[2, 3], Some(&[0]), false),
            [1, 3]
        );
        assert_eq!(
            reshape_sum_backward(&[2], &[2, 3], Some(&[1]), false),
            [2, 1]
        );
        assert_eq!(
            reshape_sum_backward(&[2], &[2, 3, 4], Some(&[1, 2]), false),
            [2, 1, 1]
        );
    }

    #[test]
    fn reshape_sum_backward_sorts_before_inserting() {
        // Descending axes must still produce (1, 3, 1): inserting in the order
        // given would put the 1s in the wrong places.
        assert_eq!(
            reshape_sum_backward(&[3], &[2, 3, 4], Some(&[2, 0]), false),
            [1, 3, 1]
        );
    }

    #[test]
    fn reshape_sum_backward_accepts_negative_axes() {
        assert_eq!(
            reshape_sum_backward(&[2], &[2, 3], Some(&[-1]), false),
            [2, 1]
        );
    }

    #[test]
    fn reshape_sum_backward_passes_the_shape_through() {
        // keepdims, axis=None and 0-d inputs each leave the gradient alone.
        assert_eq!(
            reshape_sum_backward(&[2, 1], &[2, 3], Some(&[1]), true),
            [2, 1]
        );
        assert_eq!(
            reshape_sum_backward(&[], &[2, 3], None, false),
            Vec::<usize>::new()
        );
        assert_eq!(
            reshape_sum_backward(&[], &[], Some(&[0]), false),
            Vec::<usize>::new()
        );
    }
}
