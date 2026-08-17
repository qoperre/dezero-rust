//! The array-level plumbing convolution is built from (steps 56–57).
//!
//! Three groups, all below the graph:
//!
//! * [`Pair`] and [`pair`] — Python's `dezero.utils.pair`, the "an `int` means
//!   the same value twice" convention every kernel/stride/padding argument
//!   uses;
//! * [`get_conv_outsize`] and [`get_deconv_outsize`] — the output-size
//!   arithmetic of step 56, ported from `dezero/utils.py`;
//! * [`im2col_array`] and [`col2im_array`] — the patch-extraction pair from
//!   `dezero/functions_conv.py`, and the reason a convolution can be written as
//!   a single matrix product.
//!
//! # `im2col` and `col2im` are adjoints, by construction
//!
//! `im2col` is a *linear* map: every entry of its output is either zero (a
//! padded position) or a verbatim copy of one input pixel. Written as a matrix
//! it is a 0/1 selection matrix `S`, and `col2im` is exactly `Sᵀ` — it adds
//! each patch element back onto the pixel it was copied from, which is why
//! repeated visits *accumulate* rather than assign.
//!
//! Getting that pairing wrong is the classic convolution bug: the shapes still
//! agree, the forward pass still runs, and only the gradient is silently wrong.
//! So both directions here walk the *same* iterator, [`Patches::for_each`],
//! which yields `(row, column, pixel)` triples; `im2col` reads the pixel and
//! writes the patch entry, `col2im` reads the patch entry and adds to the
//! pixel. Neither can drift from the other without the shared visitor changing
//! underneath both, and `utils::conv`'s tests pin the adjoint identity
//! `<im2col(x), g> == <x, col2im(g)>` numerically as well.
//!
//! This mirrors [`gather_rows`](crate::utils::array::gather_rows) /
//! [`scatter_add_rows`](crate::utils::array::scatter_add_rows) one module over,
//! which stand in the same relation for the loss functions.
//!
//! # Divergence from the reference's padding
//!
//! Python pads by `(PH, PH + SH - 1)` — an extra `SH - 1` rows on the bottom
//! and `SW - 1` columns on the right — and `col2im_array` allocates the same
//! oversized buffer before cropping it away. That margin is never read and
//! never written: the largest padded row any patch touches is
//! `(KH - 1) + SH * (OH - 1) <= H + 2 * PH - 1`, which is inside the ordinary
//! padding already. It exists in Chainer's implementation to keep numpy's
//! `j:j_lim:SH` slicing from running short. This port indexes explicitly rather
//! than slicing, so the margin has nothing to do and is not allocated. The
//! numbers are identical.

use ndarray::{Array2, Array4, ArrayD, ArrayView2, ArrayView4, CowArray, Ix2, Ix4, IxDyn};

// ---------------------------------------------------------------------------
// pair
// ---------------------------------------------------------------------------

/// A `(height, width)` pair — the value Python's `dezero.utils.pair` returns.
///
/// DeZero lets every spatial argument be either a single `int` (meaning "the
/// same in both directions") or an explicit `(h, w)` tuple. Python decides
/// which at run time with `isinstance`; Rust decides at compile time with
/// `From`, so `conv2d(&x, &w, None, 1, 0)` and `conv2d(&x, &w, None, (2, 1),
/// 0)` both type-check and neither can be a third thing.
///
/// # Examples
///
/// ```
/// use dezero::utils::conv::{pair, Pair};
///
/// assert_eq!(pair(3), (3, 3));
/// assert_eq!(pair((3, 5)), (3, 5));
/// assert_eq!(pair([3, 5]), (3, 5));
/// assert_eq!(Pair::new(2, 4).as_tuple(), (2, 4));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Pair {
    /// The vertical component, applied along the `H` axis of an `[N, C, H, W]`
    /// batch.
    pub height: usize,
    /// The horizontal component, applied along the `W` axis.
    pub width: usize,
}

impl Pair {
    /// A pair with an explicit height and width.
    #[must_use]
    pub fn new(height: usize, width: usize) -> Self {
        Self { height, width }
    }

    /// The pair as a plain tuple, which is what the internals destructure.
    #[must_use]
    pub fn as_tuple(self) -> (usize, usize) {
        (self.height, self.width)
    }
}

impl From<usize> for Pair {
    /// Python's `pair(3) -> (3, 3)`.
    fn from(value: usize) -> Self {
        Self::new(value, value)
    }
}

impl From<(usize, usize)> for Pair {
    fn from(value: (usize, usize)) -> Self {
        Self::new(value.0, value.1)
    }
}

impl From<[usize; 2]> for Pair {
    fn from(value: [usize; 2]) -> Self {
        Self::new(value[0], value[1])
    }
}

impl From<Pair> for (usize, usize) {
    fn from(value: Pair) -> Self {
        value.as_tuple()
    }
}

/// Normalises a spatial argument to `(height, width)` — Python's
/// `dezero.utils.pair`.
///
/// # Examples
///
/// ```
/// use dezero::utils::conv::pair;
///
/// assert_eq!(pair(1), (1, 1));
/// assert_eq!(pair((2, 3)), (2, 3));
/// ```
#[must_use]
pub fn pair(value: impl Into<Pair>) -> (usize, usize) {
    value.into().as_tuple()
}

// ---------------------------------------------------------------------------
// output-size arithmetic (step 56)
// ---------------------------------------------------------------------------

/// The output length of a convolution along one axis — Python's
/// `get_conv_outsize`.
///
/// `(input_size + 2 * pad - kernel_size) / stride + 1`, rounded down: the
/// window starts at `-pad` and advances by `stride` for as long as it still
/// fits inside the padded input. A remainder means the last few input elements
/// are never covered, which is exactly what numpy's floor division encodes.
///
/// # Examples
///
/// ```
/// use dezero::utils::conv::get_conv_outsize;
///
/// assert_eq!(get_conv_outsize(7, 3, 1, 0), 5); // 3x3 over 7x7, no padding
/// assert_eq!(get_conv_outsize(7, 3, 2, 1), 4); // strided, "same"-ish padding
/// assert_eq!(get_conv_outsize(5, 3, 1, 1), 5); // pad 1 keeps a 3x3 the size
/// assert_eq!(get_conv_outsize(6, 2, 2, 0), 3); // halved by a 2x2 pooling
/// ```
///
/// # Panics
///
/// Panics if `stride` is 0, or if the kernel is wider than the padded input.
/// Python computes a negative or zero size there and fails further downstream
/// with a shape error that no longer names the cause; this is the same family
/// as `utils::sum_to`'s target check (`docs/DIVERGENCES.md`).
#[must_use]
pub fn get_conv_outsize(input_size: usize, kernel_size: usize, stride: usize, pad: usize) -> usize {
    assert!(
        stride > 0,
        "dezero: a convolution stride must be at least 1"
    );
    let padded = input_size + 2 * pad;
    assert!(
        padded >= kernel_size,
        "dezero: a kernel of size {kernel_size} does not fit in an input of size {input_size} \
         padded to {padded}"
    );
    (padded - kernel_size) / stride + 1
}

/// The output length of a transposed convolution along one axis — Python's
/// `get_deconv_outsize`.
///
/// `stride * (size - 1) + kernel_size - 2 * pad`: the inverse of
/// [`get_conv_outsize`] on the sizes it can invert, which is what makes
/// `deconv2d` able to reconstruct the shape a `conv2d` consumed.
///
/// # Examples
///
/// ```
/// use dezero::utils::conv::{get_conv_outsize, get_deconv_outsize};
///
/// assert_eq!(get_deconv_outsize(5, 3, 1, 0), 7);
///
/// // Round trip: deconv restores a size conv reduced, whenever conv lost
/// // nothing to its floor division.
/// for size in 1..20 {
///     let out = get_conv_outsize(size, 3, 2, 1);
///     if (size + 2 - 3) % 2 == 0 {
///         assert_eq!(get_deconv_outsize(out, 3, 2, 1), size);
///     }
/// }
/// ```
///
/// # Panics
///
/// Panics if `size` is 0, or if the padding removes more than the whole span
/// (which would be a negative size in Python).
#[must_use]
pub fn get_deconv_outsize(size: usize, kernel_size: usize, stride: usize, pad: usize) -> usize {
    assert!(
        size > 0,
        "dezero: a transposed convolution needs an input of at least 1 element per axis"
    );
    let span = stride * (size - 1) + kernel_size;
    assert!(
        span >= 2 * pad,
        "dezero: padding {pad} removes more than the whole {span}-element span of a transposed \
         convolution"
    );
    span - 2 * pad
}

// ---------------------------------------------------------------------------
// the patch geometry
// ---------------------------------------------------------------------------

/// Where every patch element of an `[N, C, H, W]` batch comes from.
///
/// One value object holding the full geometry, so [`im2col_array`] and
/// [`col2im_array`] cannot disagree about it — see the module documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Patches {
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    out_h: usize,
    out_w: usize,
}

impl Patches {
    /// Resolves the geometry for an image shape and a kernel/stride/padding.
    ///
    /// # Panics
    ///
    /// Panics for the reasons [`get_conv_outsize`] does.
    pub(crate) fn new(
        img_shape: (usize, usize, usize, usize),
        kernel: (usize, usize),
        stride: (usize, usize),
        pad: (usize, usize),
    ) -> Self {
        let (batch, channels, height, width) = img_shape;
        let (kernel_h, kernel_w) = kernel;
        let (stride_h, stride_w) = stride;
        let (pad_h, pad_w) = pad;
        Self {
            batch,
            channels,
            height,
            width,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h: get_conv_outsize(height, kernel_h, stride_h, pad_h),
            out_w: get_conv_outsize(width, kernel_w, stride_w, pad_w),
        }
    }

    /// The spatial size of the result, `(OH, OW)`.
    pub(crate) fn out_size(&self) -> (usize, usize) {
        (self.out_h, self.out_w)
    }

    /// Rows of the matrix form: `N * OH * OW`, one per output position.
    pub(crate) fn rows(&self) -> usize {
        self.batch * self.out_h * self.out_w
    }

    /// Columns of the matrix form: `C * KH * KW`, one per patch element.
    pub(crate) fn columns(&self) -> usize {
        self.channels * self.kernel_h * self.kernel_w
    }

    /// The matrix shape `(N * OH * OW, C * KH * KW)`.
    pub(crate) fn matrix_shape(&self) -> (usize, usize) {
        (self.rows(), self.columns())
    }

    /// The rank-6 shape `(N, C, KH, KW, OH, OW)` Python calls `to_matrix=False`.
    pub(crate) fn tensor_shape(&self) -> [usize; 6] {
        [
            self.batch,
            self.channels,
            self.kernel_h,
            self.kernel_w,
            self.out_h,
            self.out_w,
        ]
    }

    /// The image shape `(N, C, H, W)`.
    pub(crate) fn img_shape(&self) -> (usize, usize, usize, usize) {
        (self.batch, self.channels, self.height, self.width)
    }

    /// Visits every `(row, column, pixel)` the extraction relates.
    ///
    /// Positions that fall in the padding are simply *not visited*: in the
    /// forward direction they stay at the zero the buffer was created with, and
    /// in the adjoint direction they have no pixel to add to. That is the whole
    /// of the padding semantics, in one omission.
    ///
    /// The iteration order — output position outermost, kernel column innermost
    /// — walks both the matrix and the image close to sequentially.
    pub(crate) fn for_each(&self, mut visit: impl FnMut(usize, usize, [usize; 4])) {
        for n in 0..self.batch {
            for out_y in 0..self.out_h {
                for out_x in 0..self.out_w {
                    let row = (n * self.out_h + out_y) * self.out_w + out_x;
                    for c in 0..self.channels {
                        for j in 0..self.kernel_h {
                            let padded_y = j + self.stride_h * out_y;
                            if padded_y < self.pad_h || padded_y >= self.pad_h + self.height {
                                continue;
                            }
                            let y = padded_y - self.pad_h;
                            let column = (c * self.kernel_h + j) * self.kernel_w;
                            for i in 0..self.kernel_w {
                                let padded_x = i + self.stride_w * out_x;
                                if padded_x < self.pad_w || padded_x >= self.pad_w + self.width {
                                    continue;
                                }
                                visit(row, column + i, [n, c, y, padded_x - self.pad_w]);
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// rank-4 helpers
// ---------------------------------------------------------------------------

/// Destructures a shape that must be `[N, C, H, W]`.
///
/// # Panics
///
/// Panics if `shape` is not of rank 4.
pub(crate) fn nchw(shape: &[usize], op: &str) -> (usize, usize, usize, usize) {
    let [n, c, h, w] = shape[..] else {
        panic!("dezero: {op} needs a 4-dimensional (N, C, H, W) array, got shape {shape:?}");
    };
    (n, c, h, w)
}

/// Re-types a dynamic-rank array as a rank-4 view.
///
/// # Panics
///
/// Panics if `x` is not of rank 4.
pub(crate) fn view4<'a>(x: &'a ArrayD<f64>, op: &str) -> ArrayView4<'a, f64> {
    match x.view().into_dimensionality::<Ix4>() {
        Ok(view) => view,
        Err(_) => panic!(
            "dezero: {op} needs a 4-dimensional (N, C, H, W) array, got shape {:?}",
            x.shape()
        ),
    }
}

/// Re-types a dynamic-rank array as a rank-2 view.
///
/// # Panics
///
/// Panics if `x` is not of rank 2.
pub(crate) fn view2<'a>(x: &'a ArrayD<f64>, op: &str) -> ArrayView2<'a, f64> {
    match x.view().into_dimensionality::<Ix2>() {
        Ok(view) => view,
        Err(_) => panic!(
            "dezero: {op} needs a 2-dimensional matrix, got shape {:?}",
            x.shape()
        ),
    }
}

/// Flattens `[N, C, H, W]` into the `(N * H * W, C)` matrix whose rows are
/// pixels and whose columns are channels — numpy's
/// `x.transpose(0, 2, 3, 1).reshape(-1, C)`.
///
/// This is the layout every contraction in `functions::conv` reduces to: it is
/// the row order [`Patches::for_each`] uses, so a matrix in this form lines up
/// with an `im2col` result without any further shuffling.
///
/// # Panics
///
/// Panics if `x` is not of rank 4.
pub(crate) fn nchw_to_rows(x: &ArrayD<f64>, op: &str) -> Array2<f64> {
    let view = view4(x, op);
    let (n, c, h, w) = view.dim();
    let rows = view
        .permuted_axes([0, 2, 3, 1])
        .as_standard_layout()
        .into_owned();
    rows.into_shape_with_order((n * h * w, c))
        .expect("permuting to (N, H, W, C) preserves the element count")
}

/// The inverse of [`nchw_to_rows`]: `(N * H * W, C)` back to `[N, C, H, W]` —
/// numpy's `y.reshape(N, H, W, C).transpose(0, 3, 1, 2)`.
///
/// # Panics
///
/// Panics if `rows` does not have exactly `n * h * w` rows and `c` columns.
pub(crate) fn rows_to_nchw(
    rows: &Array2<f64>,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
) -> ArrayD<f64> {
    assert!(
        rows.dim() == (n * h * w, c),
        "dezero: cannot fold a {:?} matrix into an image of shape {:?}",
        rows.dim(),
        [n, c, h, w]
    );
    let reshaped = rows
        .view()
        .into_shape_with_order((n, h, w, c))
        .expect("the row and column counts were just checked");
    reshaped
        .permuted_axes([0, 3, 1, 2])
        .as_standard_layout()
        .into_owned()
        .into_dyn()
}

// ---------------------------------------------------------------------------
// im2col / col2im
// ---------------------------------------------------------------------------

/// Extracts every kernel-sized patch of `img` into the matrix rows they
/// convolve against — the `to_matrix=True` half of Python's `im2col_array`.
///
/// `img` is `[N, C, H, W]`; the result is `(N * OH * OW, C * KH * KW)`, one row
/// per output position and one column per weight of the filter.
///
/// # Panics
///
/// Panics if `img` is not of rank 4, or for the reasons
/// [`get_conv_outsize`] does.
#[must_use]
pub fn im2col_matrix(
    img: &ArrayD<f64>,
    kernel: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
) -> Array2<f64> {
    let view = view4(img, "im2col");
    let patches = Patches::new(view.dim(), kernel, stride, pad);
    let mut col = Array2::zeros(patches.matrix_shape());
    patches.for_each(|row, column, pixel| col[[row, column]] = view[pixel]);
    col
}

/// Adds every patch element back onto the pixel it came from — the adjoint of
/// [`im2col_matrix`], and the `to_matrix=True` half of Python's
/// `col2im_array`.
///
/// Overlapping patches *accumulate*. That is not a convenience: it is what
/// makes this the transpose of the extraction rather than merely its inverse on
/// the non-overlapping case, and therefore what makes a strided or padded
/// convolution differentiable.
///
/// # Panics
///
/// Panics if `img_shape` is not of rank 4, if `col`'s shape does not match the
/// geometry that shape implies, or for the reasons [`get_conv_outsize`] does.
#[must_use]
pub fn col2im_matrix(
    col: &ArrayView2<'_, f64>,
    img_shape: (usize, usize, usize, usize),
    kernel: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
) -> ArrayD<f64> {
    let patches = Patches::new(img_shape, kernel, stride, pad);
    assert!(
        col.dim() == patches.matrix_shape(),
        "dezero: col2im got a {:?} matrix but an image of shape {:?} with a {kernel:?} kernel \
         needs {:?}",
        col.dim(),
        [img_shape.0, img_shape.1, img_shape.2, img_shape.3],
        patches.matrix_shape()
    );

    let mut img = Array4::zeros(img_shape);
    patches.for_each(|row, column, pixel| img[pixel] += col[[row, column]]);
    img.into_dyn()
}

/// Extracts the patches of an `[N, C, H, W]` batch — Python's `im2col_array`.
///
/// With `to_matrix` the result is `(N * OH * OW, C * KH * KW)`; without it, the
/// rank-6 `(N, C, KH, KW, OH, OW)` the fused convolution contracts directly.
/// The two hold the same numbers in a different order.
///
/// # Examples
///
/// ```
/// use dezero::utils::conv::im2col_array;
/// use ndarray::Array;
///
/// // A single 1x1x3x3 image, 2x2 kernel, stride 1, no padding.
/// let img = Array::from_shape_vec((1, 1, 3, 3), (1..=9).map(f64::from).collect())
///     .expect("3x3")
///     .into_dyn();
///
/// let col = im2col_array(&img, (2, 2), (1, 1), (0, 0), true);
/// assert_eq!(col.shape(), &[4, 4]);
/// // The first output position sees the top-left 2x2 block.
/// assert_eq!(col.slice(ndarray::s![0, ..]).to_vec(), vec![1.0, 2.0, 4.0, 5.0]);
///
/// let tensor = im2col_array(&img, (2, 2), (1, 1), (0, 0), false);
/// assert_eq!(tensor.shape(), &[1, 1, 2, 2, 2, 2]);
/// ```
///
/// # Panics
///
/// Panics if `img` is not of rank 4, or for the reasons [`get_conv_outsize`]
/// does.
#[must_use]
pub fn im2col_array(
    img: &ArrayD<f64>,
    kernel: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
    to_matrix: bool,
) -> ArrayD<f64> {
    let col = im2col_matrix(img, kernel, stride, pad);
    if to_matrix {
        return col.into_dyn();
    }

    let patches = Patches::new(view4(img, "im2col").dim(), kernel, stride, pad);
    let (out_h, out_w) = patches.out_size();
    let (n, c, _, _) = patches.img_shape();
    let (kernel_h, kernel_w) = kernel;
    // (N*OH*OW, C*KH*KW) -> (N, OH, OW, C, KH, KW) -> (N, C, KH, KW, OH, OW).
    col.into_shape_with_order(IxDyn(&[n, out_h, out_w, c, kernel_h, kernel_w]))
        .expect("the matrix holds exactly N*OH*OW*C*KH*KW elements")
        .permuted_axes(IxDyn(&[0, 3, 4, 5, 1, 2]))
        .as_standard_layout()
        .into_owned()
}

/// Folds patches back into an `[N, C, H, W]` batch — Python's `col2im_array`,
/// and the adjoint of [`im2col_array`].
///
/// `to_matrix` says which layout `col` is in, exactly as it does for
/// [`im2col_array`].
///
/// # Examples
///
/// ```
/// use dezero::utils::conv::{col2im_array, im2col_array};
/// use ndarray::{Array, ArrayD};
///
/// let img: ArrayD<f64> = Array::from_shape_vec((1, 1, 3, 3), (1..=9).map(f64::from).collect())
///     .expect("3x3")
///     .into_dyn();
/// let col = im2col_array(&img, (2, 2), (1, 1), (0, 0), true);
///
/// // <im2col(x), g> == <x, col2im(g)>: the two really are adjoints.
/// let g = col.mapv(|v| v * 0.5 - 1.0);
/// let back = col2im_array(&g, &[1, 1, 3, 3], (2, 2), (1, 1), (0, 0), true);
/// let left: f64 = (&col * &g).sum();
/// let right: f64 = (&img * &back).sum();
/// assert!((left - right).abs() < 1e-12);
/// ```
///
/// # Panics
///
/// Panics if `img_shape` is not of rank 4, if `col`'s rank or shape disagrees
/// with `to_matrix` and the geometry, or for the reasons [`get_conv_outsize`]
/// does.
#[must_use]
pub fn col2im_array(
    col: &ArrayD<f64>,
    img_shape: &[usize],
    kernel: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
    to_matrix: bool,
) -> ArrayD<f64> {
    let shape = nchw(img_shape, "col2im");
    let patches = Patches::new(shape, kernel, stride, pad);

    let matrix: CowArray<'_, f64, Ix2> = if to_matrix {
        CowArray::from(view2(col, "col2im"))
    } else {
        let expected = patches.tensor_shape();
        assert!(
            col.shape() == expected,
            "dezero: col2im expected a {expected:?} tensor for an image of shape {img_shape:?}, \
             got shape {:?}",
            col.shape()
        );
        // (N, C, KH, KW, OH, OW) -> (N, OH, OW, C, KH, KW) -> the matrix.
        let rows = col
            .view()
            .permuted_axes(IxDyn(&[0, 4, 5, 1, 2, 3]))
            .as_standard_layout()
            .into_owned();
        CowArray::from(
            rows.into_shape_with_order(patches.matrix_shape())
                .expect("the tensor holds exactly N*OH*OW*C*KH*KW elements"),
        )
    };

    col2im_matrix(&matrix.view(), shape, kernel, stride, pad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    fn image(n: usize, c: usize, h: usize, w: usize) -> ArrayD<f64> {
        let count = n * c * h * w;
        #[allow(
            clippy::cast_precision_loss,
            reason = "the test images have a few hundred elements at most"
        )]
        let values: Vec<f64> = (0..count).map(|v| (v as f64) * 0.5 - 3.0).collect();
        Array::from_shape_vec((n, c, h, w), values)
            .expect("the buffer matches the shape")
            .into_dyn()
    }

    // -- pair --------------------------------------------------------------

    #[test]
    fn pair_accepts_every_spelling() {
        assert_eq!(pair(4), (4, 4));
        assert_eq!(pair((4, 7)), (4, 7));
        assert_eq!(pair([4, 7]), (4, 7));
        assert_eq!(pair(Pair::new(4, 7)), (4, 7));
        assert_eq!(<(usize, usize)>::from(Pair::new(1, 2)), (1, 2));
    }

    // -- output sizes (step 56) -------------------------------------------

    /// The book's step-56 worked examples.
    #[test]
    fn conv_outsize_matches_the_books_examples() {
        assert_eq!(get_conv_outsize(4, 3, 1, 1), 4);
        assert_eq!(get_conv_outsize(7, 5, 2, 1), 3);
        assert_eq!(get_conv_outsize(7, 3, 1, 0), 5);
        assert_eq!(get_conv_outsize(7, 3, 2, 1), 4);
    }

    /// The formula, over the whole grid it is ever asked about.
    #[test]
    fn conv_outsize_matches_the_formula_everywhere() {
        for input in 1..24_usize {
            for kernel in 1..8_usize {
                for stride in 1..5_usize {
                    for pad in 0..4_usize {
                        if input + 2 * pad < kernel {
                            continue;
                        }
                        let expected = (input + 2 * pad - kernel) / stride + 1;
                        assert_eq!(
                            get_conv_outsize(input, kernel, stride, pad),
                            expected,
                            "input {input}, kernel {kernel}, stride {stride}, pad {pad}"
                        );
                    }
                }
            }
        }
    }

    /// The size never exceeds what actually fits: window `out - 1` must start
    /// inside the padded input and end no later than its last element.
    #[test]
    fn conv_outsize_counts_only_windows_that_fit() {
        for input in 1..20_usize {
            for kernel in 1..6_usize {
                for stride in 1..4_usize {
                    for pad in 0..3_usize {
                        if input + 2 * pad < kernel {
                            continue;
                        }
                        let out = get_conv_outsize(input, kernel, stride, pad);
                        assert!(out >= 1);
                        assert!(stride * (out - 1) + kernel <= input + 2 * pad);
                        assert!(
                            stride * out + kernel > input + 2 * pad,
                            "one more would fit"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn deconv_outsize_inverts_conv_outsize_when_nothing_was_lost() {
        for input in 1..20_usize {
            for kernel in 1..6_usize {
                for stride in 1..4_usize {
                    for pad in 0..3_usize {
                        if input + 2 * pad < kernel || (input + 2 * pad - kernel) % stride != 0 {
                            continue;
                        }
                        let out = get_conv_outsize(input, kernel, stride, pad);
                        assert_eq!(
                            get_deconv_outsize(out, kernel, stride, pad),
                            input,
                            "input {input}, kernel {kernel}, stride {stride}, pad {pad}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn deconv_outsize_matches_the_formula() {
        assert_eq!(get_deconv_outsize(5, 3, 1, 0), 7);
        assert_eq!(get_deconv_outsize(4, 3, 2, 1), 7);
        assert_eq!(get_deconv_outsize(1, 3, 1, 0), 3);
    }

    #[test]
    #[should_panic(expected = "stride must be at least 1")]
    fn a_zero_stride_is_rejected() {
        let _ = get_conv_outsize(5, 3, 0, 0);
    }

    #[test]
    #[should_panic(expected = "does not fit in an input")]
    fn an_oversized_kernel_is_rejected() {
        let _ = get_conv_outsize(2, 5, 1, 0);
    }

    #[test]
    #[should_panic(expected = "at least 1 element per axis")]
    fn deconv_rejects_an_empty_axis() {
        let _ = get_deconv_outsize(0, 3, 1, 0);
    }

    // -- im2col ------------------------------------------------------------

    #[test]
    fn im2col_extracts_the_expected_patches() {
        let img = Array::from_shape_vec((1, 1, 3, 3), (1..=9).map(f64::from).collect())
            .expect("3x3")
            .into_dyn();
        let col = im2col_matrix(&img, (2, 2), (1, 1), (0, 0));
        assert_eq!(col.dim(), (4, 4));
        assert_eq!(col.row(0).to_vec(), vec![1.0, 2.0, 4.0, 5.0]);
        assert_eq!(col.row(1).to_vec(), vec![2.0, 3.0, 5.0, 6.0]);
        assert_eq!(col.row(2).to_vec(), vec![4.0, 5.0, 7.0, 8.0]);
        assert_eq!(col.row(3).to_vec(), vec![5.0, 6.0, 8.0, 9.0]);
    }

    #[test]
    fn im2col_zero_fills_the_padding() {
        let img = Array::from_shape_vec((1, 1, 2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .expect("2x2")
            .into_dyn();
        // 3x3 kernel, pad 1: exactly one output position, the whole padded 4x4
        // minus its border.
        let col = im2col_matrix(&img, (3, 3), (1, 1), (1, 1));
        assert_eq!(col.dim(), (4, 9));
        // Top-left output position: only the bottom-right quadrant is real.
        assert_eq!(
            col.row(0).to_vec(),
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0]
        );
    }

    #[test]
    fn im2col_shapes_follow_the_output_size() {
        let img = image(2, 3, 7, 7);
        for (kernel, stride, pad, out) in [
            ((3, 3), (1, 1), (0, 0), 5),
            ((3, 3), (2, 2), (1, 1), 4),
            ((5, 5), (1, 1), (2, 2), 7),
        ] {
            let col = im2col_array(&img, kernel, stride, pad, true);
            assert_eq!(col.shape(), &[2 * out * out, 3 * kernel.0 * kernel.1]);
            let tensor = im2col_array(&img, kernel, stride, pad, false);
            assert_eq!(tensor.shape(), &[2, 3, kernel.0, kernel.1, out, out]);
        }
    }

    /// `to_matrix` only reorders: both layouts hold the same multiset of
    /// numbers, and the documented permutation maps one onto the other.
    #[test]
    fn the_two_layouts_are_one_permutation_apart() {
        let img = image(2, 3, 5, 5);
        let (kernel, stride, pad) = ((3, 3), (1, 1), (1, 1));
        let matrix = im2col_array(&img, kernel, stride, pad, true);
        let tensor = im2col_array(&img, kernel, stride, pad, false);

        let regrouped = matrix
            .clone()
            .into_shape_with_order(IxDyn(&[2, 5, 5, 3, 3, 3]))
            .expect("reshape")
            .permuted_axes(IxDyn(&[0, 3, 4, 5, 1, 2]))
            .as_standard_layout()
            .into_owned();
        assert_eq!(regrouped, tensor);
    }

    // -- col2im ------------------------------------------------------------

    /// The identity that matters: `<im2col(x), g> == <x, col2im(g)>` for every
    /// `x` and `g`. A transposed index, a wrong padding offset or an assignment
    /// where an accumulation belongs all pass every shape check and fail here.
    #[test]
    fn col2im_is_the_adjoint_of_im2col() {
        for (shape, kernel, stride, pad) in [
            ((1, 1, 4, 4), (2, 2), (1, 1), (0, 0)),
            ((2, 3, 5, 5), (3, 3), (1, 1), (1, 1)),
            ((2, 3, 7, 7), (3, 3), (2, 2), (1, 1)),
            ((1, 2, 6, 6), (2, 2), (2, 2), (0, 0)),
            ((1, 2, 6, 4), (3, 2), (1, 2), (2, 1)),
            ((3, 1, 8, 8), (5, 5), (3, 3), (2, 2)),
        ] {
            let (n, c, h, w) = shape;
            let x = image(n, c, h, w);
            let col = im2col_array(&x, kernel, stride, pad, true);
            // An asymmetric seed: a constant one would hide a transposed index.
            let mut step = 0.0;
            let g = col.mapv(|_| {
                step += 0.25;
                (step % 3.0) - 1.5
            });

            let back = col2im_array(&g, &[n, c, h, w], kernel, stride, pad, true);
            assert_eq!(back.shape(), &[n, c, h, w]);

            let left: f64 = (&col * &g).sum();
            let right: f64 = (&x * &back).sum();
            assert!(
                (left - right).abs() <= 1e-9 * left.abs().max(1.0),
                "shape {shape:?}, kernel {kernel:?}, stride {stride:?}, pad {pad:?}: \
                 <im2col(x), g> = {left} but <x, col2im(g)> = {right}"
            );
        }
    }

    /// The adjoint identity holds in the rank-6 layout too, which is the one
    /// the fused convolution uses.
    #[test]
    fn the_adjoint_identity_holds_in_the_tensor_layout() {
        let x = image(2, 3, 5, 5);
        let (kernel, stride, pad) = ((3, 3), (2, 2), (1, 1));
        let col = im2col_array(&x, kernel, stride, pad, false);
        let g = col.mapv(|v| v.mul_add(0.3, 0.7));
        let back = col2im_array(&g, &[2, 3, 5, 5], kernel, stride, pad, false);

        let left: f64 = (&col * &g).sum();
        let right: f64 = (&x * &back).sum();
        assert!((left - right).abs() < 1e-9);
    }

    /// Overlapping patches must *add*, not overwrite: with a stride below the
    /// kernel size every interior pixel is visited several times.
    #[test]
    fn col2im_accumulates_overlapping_patches() {
        let ones = ArrayD::from_elem(IxDyn(&[1, 1, 3, 3]), 1.0);
        let col = im2col_array(&ones, (2, 2), (1, 1), (0, 0), true);
        let back = col2im_array(&col, &[1, 1, 3, 3], (2, 2), (1, 1), (0, 0), true);
        // Corners belong to 1 patch, edges to 2, the centre to 4.
        assert_eq!(back[[0, 0, 0, 0]], 1.0);
        assert_eq!(back[[0, 0, 0, 1]], 2.0);
        assert_eq!(back[[0, 0, 1, 1]], 4.0);
    }

    /// A non-overlapping extraction is invertible, which is the easy special
    /// case the adjoint reduces to.
    #[test]
    fn col2im_inverts_im2col_when_patches_do_not_overlap() {
        let x = image(2, 3, 6, 6);
        let col = im2col_array(&x, (2, 2), (2, 2), (0, 0), true);
        let back = col2im_array(&col, &[2, 3, 6, 6], (2, 2), (2, 2), (0, 0), true);
        assert_eq!(back, x);
    }

    /// Pixels the geometry never reaches — here the last row and column, which
    /// the stride steps over — must come back as zero rather than as garbage.
    #[test]
    fn col2im_leaves_uncovered_pixels_at_zero() {
        let x = image(1, 1, 5, 5);
        let col = im2col_array(&x, (2, 2), (2, 2), (0, 0), true);
        let back = col2im_array(&col, &[1, 1, 5, 5], (2, 2), (2, 2), (0, 0), true);
        for i in 0..5 {
            assert_eq!(back[[0, 0, 4, i]], 0.0, "row 4 is never covered");
            assert_eq!(back[[0, 0, i, 4]], 0.0, "column 4 is never covered");
        }
    }

    #[test]
    #[should_panic(expected = "4-dimensional (N, C, H, W)")]
    fn im2col_rejects_a_rank_three_input() {
        let x = ArrayD::zeros(IxDyn(&[1, 2, 3]));
        let _ = im2col_array(&x, (2, 2), (1, 1), (0, 0), true);
    }

    #[test]
    #[should_panic(expected = "col2im got a")]
    fn col2im_rejects_a_mis_shaped_matrix() {
        let col = ArrayD::zeros(IxDyn(&[3, 4]));
        let _ = col2im_array(&col, &[1, 1, 4, 4], (2, 2), (1, 1), (0, 0), true);
    }

    // -- row/column helpers ------------------------------------------------

    #[test]
    fn nchw_rows_round_trip() {
        let x = image(2, 3, 4, 5);
        let rows = nchw_to_rows(&x, "test");
        assert_eq!(rows.dim(), (2 * 4 * 5, 3));
        assert_eq!(rows_to_nchw(&rows, 2, 3, 4, 5), x);
    }

    #[test]
    fn nchw_to_rows_orders_pixels_the_way_im2col_does() {
        let x = image(1, 2, 2, 2);
        let rows = nchw_to_rows(&x, "test");
        // Row (n, y, x) holds that pixel's channels.
        assert_eq!(rows[[0, 0]], x[[0, 0, 0, 0]]);
        assert_eq!(rows[[0, 1]], x[[0, 1, 0, 0]]);
        assert_eq!(rows[[3, 0]], x[[0, 0, 1, 1]]);
    }
}
