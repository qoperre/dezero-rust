//! Convolution, transposed convolution and max pooling (step 57).
//!
//! Port of `vendor/dezero-python/dezero/functions_conv.py`. Everything here
//! operates on rank-4 `[N, C, H, W]` batches, the shape the book settles on in
//! step 55, and everything reduces to the [`im2col`]/[`col2im`] pair in
//! [`crate::utils::conv`] plus one matrix product.
//!
//! # Why one matrix product is the whole idea
//!
//! A convolution is a sum over `C * KH * KW` products, repeated once per output
//! position. [`im2col`] lays those `C * KH * KW` values out as a *row* — one row
//! per output position — so the convolution becomes
//!
//! ```text
//! y = im2col(x) . W.reshape(OC, -1).T        (N*OH*OW, C*KH*KW) . (C*KH*KW, OC)
//! ```
//!
//! which is a single `dot`. Every op in this module is that same contraction
//! read in a different direction:
//!
//! | op | contraction |
//! |----|-------------|
//! | [`Conv2d`] | `col . W` |
//! | [`Deconv2d`] | `x . W`, then scattered back by `col2im` |
//! | [`Conv2dGradW`] | `gy.T . col` |
//! | [`Pooling`] | no product at all — a max over each row segment |
//!
//! # The backward chain
//!
//! The three convolution ops differentiate into each other, and none of them
//! touches a raw `ArrayD` to do it (`docs/ARCHITECTURE.md`):
//!
//! ```text
//! Conv2d      -> gx = deconv2d(gy, W)   gW = Conv2dGradW(x, gy)   gb = sum(gy, (0,2,3))
//! Deconv2d    -> gx = conv2d(gy, W)     gW = Conv2dGradW(gy, x)   gb = sum(gy, (0,2,3))
//! Conv2dGradW -> gx = deconv2d(gy, ggW) ggy = conv2d(x, ggW)
//! ```
//!
//! The system is closed, so a convolution's gradient is differentiable again,
//! to any order.
//!
//! # Two divergences from the reference
//!
//! * Python's `Conv2DGradW.backward` reads `self.outputs` — a list of
//!   *weakrefs* — and convolves with the weakref object rather than with the
//!   incoming gradient. It is unreachable in the book (nothing
//!   double-backpropagates through a convolution) and would raise a `TypeError`
//!   if it ever ran. [`Conv2dGradW`] uses the incoming gradient, which is the
//!   actual adjoint.
//! * Python's `Pooling2DWithIndexes` has no `backward` at all, so a third-order
//!   derivative of a pooling raises `NotImplementedError`. It is a *fixed*
//!   gather, hence linear, hence its own adjoint is exactly
//!   [`Pooling2dGrad`] — so the port writes the one line and the chain closes
//!   here too.

use ndarray::{Array2, ArrayD, IxDyn};

use crate::core::function::{Op, apply1};
use crate::core::ops::{one, two};
use crate::core::variable::Variable;
use crate::functions::reduce::sum;
use crate::utils::conv::{
    Pair, col2im_array, col2im_matrix, get_conv_outsize, get_deconv_outsize, im2col_array,
    im2col_matrix, nchw, nchw_to_rows, pair, rows_to_nchw,
};

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

/// Splits an argument list that is `(x, W)` or `(x, W, b)`.
///
/// The bias is encoded in the arity for the reason
/// [`linear`](crate::linear)'s is: [`apply`](crate::apply) refuses to run an op
/// on a variable that holds no data, so Python's `Variable(None)` bias cannot
/// be an input at all (`docs/DIVERGENCES.md`, row 16).
///
/// # Panics
///
/// Panics on any other arity.
fn split<'a, T>(items: &'a [T], op: &str, role: &str) -> (&'a T, &'a T, Option<&'a T>) {
    match items {
        [x, w] => (x, w, None),
        [x, w, b] => (x, w, Some(b)),
        _ => panic!(
            "dezero: {op} expects 2 or 3 {role} (x, W and an optional b), got {}",
            items.len()
        ),
    }
}

/// The spatial output size of a convolution over `[N, C, H, W]`.
fn out_size(
    height: usize,
    width: usize,
    kernel: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
) -> (usize, usize) {
    (
        get_conv_outsize(height, kernel.0, stride.0, pad.0),
        get_conv_outsize(width, kernel.1, stride.1, pad.1),
    )
}

// ---------------------------------------------------------------------------
// Im2col / Col2im
// ---------------------------------------------------------------------------

/// `y = im2col(x)` — patch extraction as a differentiable node.
#[derive(Debug, Clone, Copy)]
pub struct Im2col {
    kernel: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
    to_matrix: bool,
}

impl Im2col {
    /// Creates the op.
    #[must_use]
    pub fn new(
        kernel: impl Into<Pair>,
        stride: impl Into<Pair>,
        pad: impl Into<Pair>,
        to_matrix: bool,
    ) -> Self {
        Self {
            kernel: pair(kernel),
            stride: pair(stride),
            pad: pair(pad),
            to_matrix,
        }
    }
}

impl Op for Im2col {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = one(xs, "Im2col", "input");
        vec![im2col_array(
            x,
            self.kernel,
            self.stride,
            self.pad,
            self.to_matrix,
        )]
    }

    fn backward(
        &self,
        inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let x = one(inputs, "Im2col", "input");
        let gy = one(gys, "Im2col", "output gradient");
        // The adjoint of an extraction is the scatter-add that undoes it.
        vec![col2im(
            gy,
            &shape_of(x, "Im2col"),
            self.kernel,
            self.stride,
            self.pad,
            self.to_matrix,
        )]
    }
}

/// Extracts every kernel-sized patch of an `[N, C, H, W]` batch — Python's
/// `dezero.functions.im2col`.
///
/// With `to_matrix` the result is `(N*OH*OW, C*KH*KW)`, the layout a
/// convolution multiplies directly; without it, the rank-6
/// `(N, C, KH, KW, OH, OW)`.
///
/// # Examples
///
/// ```
/// use dezero::{im2col, Variable};
/// use ndarray::Array;
///
/// let x = Variable::new(
///     Array::from_shape_vec((1, 1, 3, 3), (1..=9).map(f64::from).collect())
///         .expect("3x3")
///         .into_dyn(),
/// );
///
/// let col = im2col(&x, 2, 1, 0, true);
/// assert_eq!(col.shape(), Some(vec![4, 4]));
///
/// // The gradient counts how many patches each pixel appeared in.
/// col.backward();
/// let g = x.grad().and_then(|g| g.data()).expect("gradient");
/// assert_eq!(g[[0, 0, 0, 0]], 1.0, "a corner belongs to one patch");
/// assert_eq!(g[[0, 0, 1, 1]], 4.0, "the centre belongs to four");
/// ```
///
/// # Panics
///
/// Panics if `x` is not a rank-4 batch, if the kernel does not fit in the
/// padded input, or if a stride is 0.
#[must_use]
pub fn im2col(
    x: &Variable,
    kernel: impl Into<Pair>,
    stride: impl Into<Pair>,
    pad: impl Into<Pair>,
    to_matrix: bool,
) -> Variable {
    apply1(Im2col::new(kernel, stride, pad, to_matrix), &[x])
}

/// `y = col2im(x)` — the adjoint of [`Im2col`] as a differentiable node.
#[derive(Debug, Clone)]
pub struct Col2im {
    input_shape: Vec<usize>,
    kernel: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
    to_matrix: bool,
}

impl Col2im {
    /// Creates the op for a target image shape.
    #[must_use]
    pub fn new(
        input_shape: &[usize],
        kernel: impl Into<Pair>,
        stride: impl Into<Pair>,
        pad: impl Into<Pair>,
        to_matrix: bool,
    ) -> Self {
        Self {
            input_shape: input_shape.to_vec(),
            kernel: pair(kernel),
            stride: pair(stride),
            pad: pair(pad),
            to_matrix,
        }
    }
}

impl Op for Col2im {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = one(xs, "Col2im", "input");
        vec![col2im_array(
            x,
            &self.input_shape,
            self.kernel,
            self.stride,
            self.pad,
            self.to_matrix,
        )]
    }

    fn backward(
        &self,
        _inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let gy = one(gys, "Col2im", "output gradient");
        vec![im2col(
            gy,
            self.kernel,
            self.stride,
            self.pad,
            self.to_matrix,
        )]
    }
}

/// Folds patches back into an `[N, C, H, W]` batch — Python's
/// `dezero.functions.col2im`, and the adjoint of [`im2col`].
///
/// # Examples
///
/// ```
/// use dezero::{col2im, im2col, Variable};
/// use ndarray::Array;
///
/// let x = Variable::new(
///     Array::from_shape_vec((1, 1, 4, 4), (1..=16).map(f64::from).collect())
///         .expect("4x4")
///         .into_dyn(),
/// );
///
/// // Non-overlapping patches: col2im is the exact inverse of im2col.
/// let col = im2col(&x, 2, 2, 0, true);
/// let back = col2im(&col, &[1, 1, 4, 4], 2, 2, 0, true);
/// assert_eq!(back.data(), x.data());
/// ```
///
/// # Panics
///
/// Panics if `input_shape` is not of rank 4, if `x`'s shape disagrees with the
/// geometry, or for the reasons [`im2col`] does.
#[must_use]
pub fn col2im(
    x: &Variable,
    input_shape: &[usize],
    kernel: impl Into<Pair>,
    stride: impl Into<Pair>,
    pad: impl Into<Pair>,
    to_matrix: bool,
) -> Variable {
    apply1(
        Col2im::new(input_shape, kernel, stride, pad, to_matrix),
        &[x],
    )
}

// ---------------------------------------------------------------------------
// Conv2d
// ---------------------------------------------------------------------------

/// `y = conv2d(x, W, b)` — a 2-dimensional convolution.
///
/// Named for Python's `dezero.functions_conv.Conv2d`. The *layer* of the same
/// name — the one that owns `W` and `b` — is [`crate::Conv2d`]; only the layer
/// is re-exported at the crate root, exactly as for `Linear`.
#[derive(Debug, Clone, Copy)]
pub struct Conv2d {
    stride: (usize, usize),
    pad: (usize, usize),
}

impl Conv2d {
    /// Creates the op.
    #[must_use]
    pub fn new(stride: impl Into<Pair>, pad: impl Into<Pair>) -> Self {
        Self {
            stride: pair(stride),
            pad: pair(pad),
        }
    }
}

impl Op for Conv2d {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let (x, w, b) = split(xs, "Conv2d", "inputs");
        let (batch, channels, height, width) = nchw(x.shape(), "the Conv2d input");
        let (out_channels, weight_channels, kernel_h, kernel_w) =
            nchw(w.shape(), "the Conv2d weight");
        assert!(
            weight_channels == channels,
            "dezero: Conv2d got a {channels}-channel input and a {weight_channels}-channel weight"
        );

        let kernel = (kernel_h, kernel_w);
        let (out_h, out_w) = out_size(height, width, kernel, self.stride, self.pad);

        // (N*OH*OW, C*KH*KW) . (C*KH*KW, OC) -- the whole convolution.
        let col = im2col_matrix(x, kernel, self.stride, self.pad);
        let weights = w
            .to_shape((out_channels, channels * kernel_h * kernel_w))
            .expect("a rank-4 weight always flattens to (OC, C*KH*KW)");
        let mut y = col.dot(&weights.t());

        if let Some(b) = b {
            let Some(view) = b.broadcast(y.raw_dim()) else {
                panic!(
                    "dezero: Conv2d cannot broadcast a bias of shape {:?} onto {out_channels} \
                     output channels",
                    b.shape()
                );
            };
            y += &view;
        }

        vec![rows_to_nchw(&y, batch, out_channels, out_h, out_w)]
    }

    fn backward(
        &self,
        inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let (x, w, b) = split(inputs, "Conv2d", "inputs");
        let gy = one(gys, "Conv2d", "output gradient");

        let (_, _, height, width) = nchw(&shape_of(x, "Conv2d"), "the Conv2d input");
        let (_, _, kernel_h, kernel_w) = nchw(&shape_of(w, "Conv2d"), "the Conv2d weight");

        // The input gradient is a transposed convolution, pinned to the input's
        // own size: the forward pass may have dropped a partial window, and
        // `get_deconv_outsize` cannot know that.
        let gx = deconv2d_with_outsize(gy, w, None, self.stride, self.pad, (height, width));
        let gw = apply1(
            Conv2dGradW::new((kernel_h, kernel_w), self.stride, self.pad),
            &[x, gy],
        );

        match b {
            // The bias was added once per output position of every image.
            Some(_) => vec![gx, gw, sum(gy, [0_isize, 2, 3], false)],
            None => vec![gx, gw],
        }
    }
}

/// Convolves an `[N, C, H, W]` batch with an `[OC, C, KH, KW]` filter bank —
/// Python's `dezero.functions.conv2d`.
///
/// `b` is optional and, when present, is one value per output channel.
///
/// # Examples
///
/// ```
/// use dezero::{conv2d, Variable};
/// use ndarray::{ArrayD, IxDyn};
///
/// let x = Variable::new(ArrayD::from_elem(IxDyn(&[1, 3, 5, 5]), 1.0));
/// let w = Variable::new(ArrayD::from_elem(IxDyn(&[4, 3, 3, 3]), 0.5));
///
/// // 3x3 filters, stride 1, no padding: 5x5 becomes 3x3.
/// let y = conv2d(&x, &w, None, 1, 0);
/// assert_eq!(y.shape(), Some(vec![1, 4, 3, 3]));
/// // Every window sums 3*3*3 = 27 ones against a weight of 0.5.
/// assert_eq!(y.data().expect("data")[[0, 0, 0, 0]], 13.5);
///
/// // Padding 1 keeps the spatial size.
/// assert_eq!(conv2d(&x, &w, None, 1, 1).shape(), Some(vec![1, 4, 5, 5]));
/// ```
///
/// # Panics
///
/// Panics if `x` or `W` is not of rank 4, if their channel counts disagree, if
/// `b` does not broadcast onto the output channels, or if the kernel does not
/// fit in the padded input.
#[must_use]
pub fn conv2d(
    x: &Variable,
    w: &Variable,
    b: Option<&Variable>,
    stride: impl Into<Pair>,
    pad: impl Into<Pair>,
) -> Variable {
    let op = Conv2d::new(stride, pad);
    match b {
        Some(b) => apply1(op, &[x, w, b]),
        None => apply1(op, &[x, w]),
    }
}

// ---------------------------------------------------------------------------
// Deconv2d
// ---------------------------------------------------------------------------

/// `y = deconv2d(x, W, b)` — a transposed convolution.
///
/// Python's `dezero.functions_conv.Deconv2d`. The weight is `[C, OC, KH, KW]`:
/// the *input* channel count comes first, the mirror image of [`Conv2d`]'s
/// layout, because this op scatters where that one gathers.
#[derive(Debug, Clone, Copy)]
pub struct Deconv2d {
    stride: (usize, usize),
    pad: (usize, usize),
    /// The spatial size to produce. `None` infers it with
    /// [`get_deconv_outsize`]; [`Conv2d::backward`] supplies it explicitly,
    /// because a forward convolution whose stride dropped a partial window
    /// cannot be inverted by the formula alone.
    outsize: Option<(usize, usize)>,
}

impl Deconv2d {
    /// Creates the op, inferring the output size.
    #[must_use]
    pub fn new(stride: impl Into<Pair>, pad: impl Into<Pair>) -> Self {
        Self {
            stride: pair(stride),
            pad: pair(pad),
            outsize: None,
        }
    }

    /// Creates the op with an explicit output size.
    #[must_use]
    pub fn with_outsize(
        stride: impl Into<Pair>,
        pad: impl Into<Pair>,
        outsize: impl Into<Pair>,
    ) -> Self {
        Self {
            stride: pair(stride),
            pad: pair(pad),
            outsize: Some(pair(outsize)),
        }
    }
}

impl Op for Deconv2d {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let (x, w, b) = split(xs, "Deconv2d", "inputs");
        let (batch, channels, height, width) = nchw(x.shape(), "the Deconv2d input");
        let (weight_channels, out_channels, kernel_h, kernel_w) =
            nchw(w.shape(), "the Deconv2d weight");
        assert!(
            weight_channels == channels,
            "dezero: Deconv2d got a {channels}-channel input and a {weight_channels}-channel weight"
        );

        let kernel = (kernel_h, kernel_w);
        let (out_h, out_w) = self.outsize.unwrap_or_else(|| {
            (
                get_deconv_outsize(height, kernel_h, self.stride.0, self.pad.0),
                get_deconv_outsize(width, kernel_w, self.stride.1, self.pad.1),
            )
        });

        // Every input pixel fans out into a whole patch, and `col2im` adds the
        // overlapping patches back together.
        let rows = nchw_to_rows(x, "Deconv2d");
        let weights = w
            .to_shape((channels, out_channels * kernel_h * kernel_w))
            .expect("a rank-4 weight always flattens to (C, OC*KH*KW)");
        let col = rows.dot(&weights);
        let mut y = col2im_matrix(
            &col.view(),
            (batch, out_channels, out_h, out_w),
            kernel,
            self.stride,
            self.pad,
        );

        if let Some(b) = b {
            let Ok(reshaped) = b.to_shape(IxDyn(&[1, out_channels, 1, 1])) else {
                panic!(
                    "dezero: Deconv2d needs one bias per output channel, got shape {:?} for \
                     {out_channels} channels",
                    b.shape()
                );
            };
            let Some(view) = reshaped.broadcast(y.raw_dim()) else {
                panic!(
                    "dezero: Deconv2d cannot broadcast a bias of shape {:?} onto an output of \
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
        let (x, w, b) = split(inputs, "Deconv2d", "inputs");
        let gy = one(gys, "Deconv2d", "output gradient");
        let (_, _, kernel_h, kernel_w) = nchw(&shape_of(w, "Deconv2d"), "the Deconv2d weight");

        // Scattering and gathering swap places: the adjoint of a transposed
        // convolution is an ordinary one.
        let gx = conv2d(gy, w, None, self.stride, self.pad);
        let gw = apply1(
            Conv2dGradW::new((kernel_h, kernel_w), self.stride, self.pad),
            &[gy, x],
        );

        match b {
            Some(_) => vec![gx, gw, sum(gy, [0_isize, 2, 3], false)],
            None => vec![gx, gw],
        }
    }
}

/// A transposed convolution — Python's `dezero.functions.deconv2d`.
///
/// The output size is inferred with [`get_deconv_outsize`]; use
/// [`deconv2d_with_outsize`] to pin it.
///
/// # Examples
///
/// ```
/// use dezero::{deconv2d, Variable};
/// use ndarray::{ArrayD, IxDyn};
///
/// let x = Variable::new(ArrayD::from_elem(IxDyn(&[1, 3, 5, 5]), 1.0));
/// // The weight is (in, out, KH, KW): the mirror of conv2d's layout.
/// let w = Variable::new(ArrayD::from_elem(IxDyn(&[3, 2, 3, 3]), 0.5));
///
/// let y = deconv2d(&x, &w, None, 1, 0);
/// assert_eq!(y.shape(), Some(vec![1, 2, 7, 7]), "a 3x3 kernel grows 5x5 to 7x7");
/// ```
///
/// # Panics
///
/// Panics for the reasons [`conv2d`] does, and additionally if an input axis is
/// empty.
#[must_use]
pub fn deconv2d(
    x: &Variable,
    w: &Variable,
    b: Option<&Variable>,
    stride: impl Into<Pair>,
    pad: impl Into<Pair>,
) -> Variable {
    apply_deconv(Deconv2d::new(stride, pad), x, w, b)
}

/// A transposed convolution with an explicit output size — Python's
/// `deconv2d(..., outsize=(h, w))`.
///
/// This is what [`Conv2d::backward`] uses: a strided forward convolution can
/// leave a partial window unvisited, and only the original input's size records
/// that.
///
/// # Panics
///
/// Panics for the reasons [`deconv2d`] does.
#[must_use]
pub fn deconv2d_with_outsize(
    x: &Variable,
    w: &Variable,
    b: Option<&Variable>,
    stride: impl Into<Pair>,
    pad: impl Into<Pair>,
    outsize: impl Into<Pair>,
) -> Variable {
    apply_deconv(Deconv2d::with_outsize(stride, pad, outsize), x, w, b)
}

/// Shared tail of [`deconv2d`] and [`deconv2d_with_outsize`].
fn apply_deconv(op: Deconv2d, x: &Variable, w: &Variable, b: Option<&Variable>) -> Variable {
    match b {
        Some(b) => apply1(op, &[x, w, b]),
        None => apply1(op, &[x, w]),
    }
}

// ---------------------------------------------------------------------------
// Conv2dGradW
// ---------------------------------------------------------------------------

/// `gW = Conv2dGradW(x, gy)` — the weight gradient of a convolution.
///
/// Python's `Conv2DGradW`. It is bilinear in its two inputs, which is why it
/// appears in *both* [`Conv2d::backward`] and [`Deconv2d::backward`], with the
/// arguments the other way round.
#[derive(Debug, Clone, Copy)]
pub struct Conv2dGradW {
    kernel: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
}

impl Conv2dGradW {
    /// Creates the op for the kernel geometry of the convolution it
    /// differentiates.
    #[must_use]
    pub fn new(kernel: impl Into<Pair>, stride: impl Into<Pair>, pad: impl Into<Pair>) -> Self {
        Self {
            kernel: pair(kernel),
            stride: pair(stride),
            pad: pair(pad),
        }
    }
}

impl Op for Conv2dGradW {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let (x, gy) = two(xs, "Conv2dGradW", "inputs");
        let (batch, channels, _, _) = nchw(x.shape(), "the Conv2dGradW input");
        let (grad_batch, out_channels, out_h, out_w) =
            nchw(gy.shape(), "the Conv2dGradW output gradient");
        assert!(
            grad_batch == batch,
            "dezero: Conv2dGradW got {batch} images and {grad_batch} output gradients"
        );

        let (kernel_h, kernel_w) = self.kernel;
        let col = im2col_matrix(x, self.kernel, self.stride, self.pad);
        assert!(
            col.nrows() == batch * out_h * out_w,
            "dezero: Conv2dGradW got an output gradient of spatial size {out_h}x{out_w} for an \
             input that convolves to {} positions per image",
            col.nrows() / batch.max(1)
        );

        // (OC, N*OH*OW) . (N*OH*OW, C*KH*KW): every output position votes on
        // every weight.
        let grad_rows = nchw_to_rows(gy, "Conv2dGradW");
        let gw = grad_rows.t().dot(&col);
        vec![
            gw.into_shape_with_order(IxDyn(&[out_channels, channels, kernel_h, kernel_w]))
                .expect("the product has exactly OC*C*KH*KW elements"),
        ]
    }

    fn backward(
        &self,
        inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let (x, gy) = two(inputs, "Conv2dGradW", "inputs");
        // Python reads `self.outputs` here -- a weakref -- and so never uses the
        // gradient flowing in. See this module's header.
        let ggw = one(gys, "Conv2dGradW", "output gradient");
        let (_, _, height, width) = nchw(&shape_of(x, "Conv2dGradW"), "the Conv2dGradW input");

        vec![
            deconv2d_with_outsize(gy, ggw, None, self.stride, self.pad, (height, width)),
            conv2d(x, ggw, None, self.stride, self.pad),
        ]
    }
}

// ---------------------------------------------------------------------------
// Pooling
// ---------------------------------------------------------------------------

/// The window offsets [`Pooling`] selected, one per output element.
///
/// Stored in `(N, C, OH, OW)` order as an index into the `KH * KW` window, and
/// carried into the backward pass: max pooling routes each gradient to the one
/// input that won its window, so the winners *are* the derivative.
type Winners = Vec<usize>;

/// `y = pooling(x)` — max pooling.
///
/// Python's `dezero.functions_conv.Pooling`.
#[derive(Debug, Clone)]
pub struct Pooling {
    kernel: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
    /// Python's `self.indexes`, assigned inside `forward` and read by
    /// `backward`. `Op::forward` takes `&mut self` for exactly this.
    winners: Winners,
}

impl Pooling {
    /// Creates the op.
    ///
    /// # Panics
    ///
    /// Panics if either kernel extent is 0: an empty window has no maximum.
    #[must_use]
    pub fn new(kernel: impl Into<Pair>, stride: impl Into<Pair>, pad: impl Into<Pair>) -> Self {
        let kernel = pair(kernel);
        assert!(
            kernel.0 > 0 && kernel.1 > 0,
            "dezero: a pooling window must be at least 1x1, got {kernel:?}"
        );
        Self {
            kernel,
            stride: pair(stride),
            pad: pair(pad),
            winners: Winners::new(),
        }
    }
}

impl Op for Pooling {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = one(xs, "Pooling", "input");
        let (batch, channels, height, width) = nchw(x.shape(), "the Pooling input");
        let (out_h, out_w) = out_size(height, width, self.kernel, self.stride, self.pad);
        let window = self.kernel.0 * self.kernel.1;

        let col = im2col_matrix(x, self.kernel, self.stride, self.pad);
        let mut values = Vec::with_capacity(batch * channels * out_h * out_w);
        let mut winners = Winners::with_capacity(values.capacity());

        for n in 0..batch {
            for c in 0..channels {
                let base = c * window;
                for out_y in 0..out_h {
                    for out_x in 0..out_w {
                        let row = (n * out_h + out_y) * out_w + out_x;
                        let mut best = 0;
                        let mut best_value = col[[row, base]];
                        for k in 1..window {
                            let candidate = col[[row, base + k]];
                            // Strictly greater, so a tie keeps the first index —
                            // numpy's `argmax` rule.
                            if candidate > best_value {
                                best_value = candidate;
                                best = k;
                            }
                        }
                        values.push(best_value);
                        winners.push(best);
                    }
                }
            }
        }

        self.winners = winners;
        vec![
            ArrayD::from_shape_vec(IxDyn(&[batch, channels, out_h, out_w]), values)
                .expect("one value per output position"),
        ]
    }

    fn backward(
        &self,
        inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let x = one(inputs, "Pooling", "input");
        let gy = one(gys, "Pooling", "output gradient");
        vec![apply1(
            Pooling2dGrad::new(
                self.kernel,
                self.stride,
                self.pad,
                &shape_of(x, "Pooling"),
                self.winners.clone(),
            ),
            &[gy],
        )]
    }
}

/// `gx = Pooling2dGrad(gy)` — routes each gradient back to the input that won
/// its window.
///
/// Python's `Pooling2DGrad`. It depends on `gy` alone: the winners were fixed
/// when the forward pass ran, which is why max pooling's second derivative is
/// zero.
#[derive(Debug, Clone)]
pub struct Pooling2dGrad {
    kernel: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
    input_shape: Vec<usize>,
    winners: Winners,
}

impl Pooling2dGrad {
    /// Creates the op from the pooling it inverts.
    #[must_use]
    pub fn new(
        kernel: impl Into<Pair>,
        stride: impl Into<Pair>,
        pad: impl Into<Pair>,
        input_shape: &[usize],
        winners: Vec<usize>,
    ) -> Self {
        Self {
            kernel: pair(kernel),
            stride: pair(stride),
            pad: pair(pad),
            input_shape: input_shape.to_vec(),
            winners,
        }
    }
}

impl Op for Pooling2dGrad {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let gy = one(xs, "Pooling2dGrad", "output gradient");
        let (batch, channels, height, width) = nchw(&self.input_shape, "the Pooling input");
        let (out_h, out_w) = out_size(height, width, self.kernel, self.stride, self.pad);
        let window = self.kernel.0 * self.kernel.1;
        assert!(
            gy.shape() == [batch, channels, out_h, out_w],
            "dezero: Pooling2dGrad expected a {:?} gradient, got shape {:?}",
            [batch, channels, out_h, out_w],
            gy.shape()
        );

        let mut col = Array2::zeros((batch * out_h * out_w, channels * window));
        let mut winner = self.winners.iter();
        for n in 0..batch {
            for c in 0..channels {
                let base = c * window;
                for out_y in 0..out_h {
                    for out_x in 0..out_w {
                        let Some(&index) = winner.next() else {
                            panic!(
                                "dezero: Pooling2dGrad has {} recorded winners for {} output \
                                 positions",
                                self.winners.len(),
                                batch * channels * out_h * out_w
                            );
                        };
                        col[[(n * out_h + out_y) * out_w + out_x, base + index]] =
                            gy[[n, c, out_y, out_x]];
                    }
                }
            }
        }

        vec![col2im_matrix(
            &col.view(),
            (batch, channels, height, width),
            self.kernel,
            self.stride,
            self.pad,
        )]
    }

    fn backward(
        &self,
        _inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let ggx = one(gys, "Pooling2dGrad", "output gradient");
        vec![apply1(
            Pooling2dWithIndexes::new(self.kernel, self.stride, self.pad, self.winners.clone()),
            &[ggx],
        )]
    }
}

/// `y = Pooling2dWithIndexes(x)` — re-reads a pooling's winners out of a *new*
/// input.
///
/// Python's `Pooling2DWithIndexes`, and the adjoint of [`Pooling2dGrad`]. Both
/// are the same fixed gather/scatter pair, so each is the other's backward and
/// the chain closes — where Python's leaves `backward` unimplemented.
#[derive(Debug, Clone)]
pub struct Pooling2dWithIndexes {
    kernel: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
    winners: Winners,
}

impl Pooling2dWithIndexes {
    /// Creates the op from the pooling whose winners it re-reads.
    #[must_use]
    pub fn new(
        kernel: impl Into<Pair>,
        stride: impl Into<Pair>,
        pad: impl Into<Pair>,
        winners: Vec<usize>,
    ) -> Self {
        Self {
            kernel: pair(kernel),
            stride: pair(stride),
            pad: pair(pad),
            winners,
        }
    }
}

impl Op for Pooling2dWithIndexes {
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
        let x = one(xs, "Pooling2dWithIndexes", "input");
        let (batch, channels, height, width) = nchw(x.shape(), "the Pooling input");
        let (out_h, out_w) = out_size(height, width, self.kernel, self.stride, self.pad);
        let window = self.kernel.0 * self.kernel.1;

        let col = im2col_matrix(x, self.kernel, self.stride, self.pad);
        let mut values = Vec::with_capacity(batch * channels * out_h * out_w);
        let mut winner = self.winners.iter();
        for n in 0..batch {
            for c in 0..channels {
                let base = c * window;
                for out_y in 0..out_h {
                    for out_x in 0..out_w {
                        let Some(&index) = winner.next() else {
                            panic!(
                                "dezero: Pooling2dWithIndexes has {} recorded winners for {} \
                                 output positions",
                                self.winners.len(),
                                batch * channels * out_h * out_w
                            );
                        };
                        values.push(col[[(n * out_h + out_y) * out_w + out_x, base + index]]);
                    }
                }
            }
        }

        vec![
            ArrayD::from_shape_vec(IxDyn(&[batch, channels, out_h, out_w]), values)
                .expect("one value per output position"),
        ]
    }

    fn backward(
        &self,
        inputs: &[Variable],
        _outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable> {
        let x = one(inputs, "Pooling2dWithIndexes", "input");
        let gy = one(gys, "Pooling2dWithIndexes", "output gradient");
        vec![apply1(
            Pooling2dGrad::new(
                self.kernel,
                self.stride,
                self.pad,
                &shape_of(x, "Pooling2dWithIndexes"),
                self.winners.clone(),
            ),
            &[gy],
        )]
    }
}

/// Max pooling over an `[N, C, H, W]` batch — Python's
/// `dezero.functions.pooling`.
///
/// Each `KH x KW` window contributes its largest element, and the gradient
/// flows back to that element alone. Ties go to the first index, as numpy's
/// `argmax` does.
///
/// # Examples
///
/// ```
/// use dezero::{pooling, Variable};
/// use ndarray::Array;
///
/// let x = Variable::new(
///     Array::from_shape_vec((1, 1, 4, 4), (1..=16).map(f64::from).collect())
///         .expect("4x4")
///         .into_dyn(),
/// );
///
/// let y = pooling(&x, 2, 2, 0);
/// assert_eq!(y.shape(), Some(vec![1, 1, 2, 2]));
/// let data = y.data().expect("data");
/// assert_eq!(data[[0, 0, 0, 0]], 6.0, "max of 1, 2, 5, 6");
/// assert_eq!(data[[0, 0, 1, 1]], 16.0);
///
/// // Only the winners receive gradient.
/// y.backward();
/// let g = x.grad().and_then(|g| g.data()).expect("gradient");
/// assert_eq!(g[[0, 0, 1, 1]], 1.0, "the 6 won its window");
/// assert_eq!(g[[0, 0, 0, 0]], 0.0, "the 1 did not");
/// ```
///
/// # Panics
///
/// Panics if `x` is not of rank 4, if the window is empty, or if the kernel
/// does not fit in the padded input.
#[must_use]
pub fn pooling(
    x: &Variable,
    kernel: impl Into<Pair>,
    stride: impl Into<Pair>,
    pad: impl Into<Pair>,
) -> Variable {
    apply1(Pooling::new(kernel, stride, pad), &[x])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functions::reduce::sum_all;
    use crate::utils::gradient_check;
    use ndarray::Array;

    const EPS: f64 = 1e-4;
    const RTOL: f64 = 1e-4;
    const ATOL: f64 = 1e-5;

    /// A deterministic, non-symmetric batch: a constant one would hide a
    /// transposed index and a symmetric one would hide a mirrored kernel.
    fn batch(n: usize, c: usize, h: usize, w: usize) -> Variable {
        let count = n * c * h * w;
        #[allow(
            clippy::cast_precision_loss,
            reason = "the test batches have a few hundred elements at most"
        )]
        let values: Vec<f64> = (0..count)
            .map(|v| ((v as f64) * 0.37).sin() * 2.0 + (v as f64) * 0.01)
            .collect();
        Variable::new(
            Array::from_shape_vec((n, c, h, w), values)
                .expect("the buffer matches the shape")
                .into_dyn(),
        )
    }

    // -- im2col / col2im ---------------------------------------------------

    #[test]
    fn im2col_forward_matches_the_array_helper() {
        let x = batch(2, 3, 5, 5);
        let expected = im2col_array(&x.data().expect("data"), (3, 3), (2, 2), (1, 1), true);
        assert_eq!(im2col(&x, 3, 2, 1, true).data(), Some(expected));
    }

    #[test]
    fn im2col_backward_is_col2im() {
        let x = batch(1, 2, 4, 4);
        let col = im2col(&x, 2, 1, 0, true);
        let seed = col.full_like(1.0).expect("shape");
        sum_all(&(&col * &seed)).backward();

        let expected = col2im_array(
            &seed.data().expect("data"),
            &[1, 2, 4, 4],
            (2, 2),
            (1, 1),
            (0, 0),
            true,
        );
        assert_eq!(x.grad().and_then(|g| g.data()), Some(expected));
    }

    #[test]
    fn im2col_gradient_matches_numerical_diff() {
        gradient_check(
            |x| sum_all(&(&im2col(x, 2, 1, 1, true) * 1.5)),
            &batch(1, 2, 3, 3),
            EPS,
            RTOL,
            ATOL,
        )
        .expect("im2col");
    }

    #[test]
    fn col2im_gradient_matches_numerical_diff() {
        // A col-shaped input for a 1x2x3x3 image with a 2x2 kernel, stride 1.
        gradient_check(
            |col| sum_all(&(&col2im(col, &[1, 2, 3, 3], 2, 1, 0, true) * 0.75)),
            &Variable::new(
                Array::from_shape_fn((4, 8), |(r, c)| {
                    f64::from(u32::try_from(r * 8 + c).expect("small")) * 0.1
                })
                .into_dyn(),
            ),
            EPS,
            RTOL,
            ATOL,
        )
        .expect("col2im");
    }

    #[test]
    fn col2im_undoes_im2col_on_non_overlapping_patches() {
        let x = batch(2, 3, 6, 6);
        let col = im2col(&x, 2, 2, 0, true);
        let back = col2im(&col, &[2, 3, 6, 6], 2, 2, 0, true);
        assert_eq!(back.data(), x.data());
    }

    // -- conv2d ------------------------------------------------------------

    #[test]
    fn conv2d_forward_shapes_follow_the_output_size() {
        let x = batch(2, 3, 7, 7);
        let w = batch(4, 3, 3, 3);
        assert_eq!(conv2d(&x, &w, None, 1, 0).shape(), Some(vec![2, 4, 5, 5]));
        assert_eq!(conv2d(&x, &w, None, 1, 1).shape(), Some(vec![2, 4, 7, 7]));
        assert_eq!(conv2d(&x, &w, None, 2, 1).shape(), Some(vec![2, 4, 4, 4]));
    }

    #[test]
    fn conv2d_with_a_one_by_one_kernel_is_a_channel_mix() {
        let x = batch(1, 2, 3, 3);
        let w = Variable::new(
            Array::from_shape_vec((1, 2, 1, 1), vec![2.0, -1.0])
                .expect("1x2x1x1")
                .into_dyn(),
        );
        let y = conv2d(&x, &w, None, 1, 0);
        let data = y.data().expect("data");
        let x_data = x.data().expect("data");
        for i in 0..3 {
            for j in 0..3 {
                let expected = 2.0f64.mul_add(x_data[[0, 0, i, j]], -x_data[[0, 1, i, j]]);
                assert!((data[[0, 0, i, j]] - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn conv2d_bias_is_added_once_per_output_position() {
        let x = Variable::new(ArrayD::zeros(IxDyn(&[2, 3, 4, 4])));
        let w = Variable::new(ArrayD::zeros(IxDyn(&[2, 3, 3, 3])));
        let b = Variable::new(ndarray::arr1(&[1.0, -2.0]).into_dyn());
        let y = conv2d(&x, &w, Some(&b), 1, 0);
        let data = y.data().expect("data");
        // `index_axis` rather than `s![.., 0, .., ..]`: the `s!` macro expands
        // to code carrying `allow(unsafe_code)`, which the crate's
        // `forbid(unsafe_code)` rejects outright.
        assert!(
            data.index_axis(ndarray::Axis(1), 0)
                .iter()
                .all(|v| *v == 1.0)
        );
        assert!(
            data.index_axis(ndarray::Axis(1), 1)
                .iter()
                .all(|v| *v == -2.0)
        );

        // ... so its gradient counts them: 2 images x 2x2 positions.
        sum_all(&y).backward();
        assert_eq!(
            b.grad().and_then(|g| g.data()),
            Some(ndarray::arr1(&[8.0, 8.0]).into_dyn())
        );
    }

    #[test]
    fn conv2d_gradients_match_numerical_diff() {
        for (stride, pad) in [(1, 0), (1, 1), (2, 1)] {
            let w = batch(2, 2, 3, 3);
            let b = Variable::new(ndarray::arr1(&[0.5, -0.25]).into_dyn());
            let x = batch(1, 2, 5, 5);

            gradient_check(
                |x| sum_all(&conv2d(x, &w, Some(&b), stride, pad)),
                &x,
                EPS,
                RTOL,
                ATOL,
            )
            .unwrap_or_else(|e| panic!("conv2d gx, stride {stride} pad {pad}: {e}"));

            gradient_check(
                |w| sum_all(&conv2d(&x, w, Some(&b), stride, pad)),
                &w,
                EPS,
                RTOL,
                ATOL,
            )
            .unwrap_or_else(|e| panic!("conv2d gW, stride {stride} pad {pad}: {e}"));

            gradient_check(
                |b| sum_all(&conv2d(&x, &w, Some(b), stride, pad)),
                &b,
                EPS,
                RTOL,
                ATOL,
            )
            .unwrap_or_else(|e| panic!("conv2d gb, stride {stride} pad {pad}: {e}"));
        }
    }

    /// A weighted seed rather than a plain sum: with all-ones gradients a
    /// transposed `gW` would still add up to the right total.
    #[test]
    fn conv2d_gradients_match_numerical_diff_under_a_weighted_seed() {
        let w = batch(3, 2, 2, 2);
        let weights = batch(1, 3, 3, 3);
        gradient_check(
            |x| sum_all(&(&conv2d(x, &w, None, 1, 0) * &weights)),
            &batch(1, 2, 4, 4),
            EPS,
            RTOL,
            ATOL,
        )
        .expect("conv2d under a weighted seed");
    }

    #[test]
    #[should_panic(expected = "3-channel input and a 2-channel weight")]
    fn conv2d_rejects_a_channel_mismatch() {
        let _ = conv2d(&batch(1, 3, 4, 4), &batch(2, 2, 3, 3), None, 1, 0);
    }

    // -- deconv2d ----------------------------------------------------------

    #[test]
    fn deconv2d_is_the_adjoint_of_conv2d() {
        // <conv2d(x, W), g> == <x, deconv2d(g, W)> whenever the geometry is
        // exactly invertible.
        let x = batch(2, 3, 6, 6);
        let w = batch(4, 3, 3, 3);
        let y = conv2d(&x, &w, None, 1, 0);
        let g = batch(2, 4, 4, 4);
        let back = deconv2d_with_outsize(&g, &w, None, 1, 0, (6, 6));

        let left: f64 = (&y.data().expect("y") * &g.data().expect("g")).sum();
        let right: f64 = (&x.data().expect("x") * &back.data().expect("back")).sum();
        assert!((left - right).abs() < 1e-9, "{left} vs {right}");
    }

    #[test]
    fn deconv2d_infers_the_output_size() {
        let x = batch(1, 3, 5, 5);
        let w = batch(3, 2, 3, 3);
        assert_eq!(deconv2d(&x, &w, None, 1, 0).shape(), Some(vec![1, 2, 7, 7]));
        assert_eq!(deconv2d(&x, &w, None, 2, 1).shape(), Some(vec![1, 2, 9, 9]));
    }

    #[test]
    fn deconv2d_gradients_match_numerical_diff() {
        let w = batch(2, 3, 3, 3);
        let b = Variable::new(ndarray::arr1(&[0.5, -0.25, 1.0]).into_dyn());
        let x = batch(1, 2, 4, 4);

        gradient_check(
            |x| sum_all(&deconv2d(x, &w, Some(&b), 1, 1)),
            &x,
            EPS,
            RTOL,
            ATOL,
        )
        .expect("deconv2d gx");
        gradient_check(
            |w| sum_all(&deconv2d(&x, w, Some(&b), 1, 1)),
            &w,
            EPS,
            RTOL,
            ATOL,
        )
        .expect("deconv2d gW");
        gradient_check(
            |b| sum_all(&deconv2d(&x, &w, Some(b), 1, 1)),
            &b,
            EPS,
            RTOL,
            ATOL,
        )
        .expect("deconv2d gb");
    }

    // -- second order ------------------------------------------------------

    /// The backward chain closes: a convolution's gradient is itself
    /// differentiable, which is what the whole `Variable`-arithmetic rule buys.
    #[test]
    fn conv2d_supports_a_second_derivative() {
        let x = batch(1, 2, 4, 4);
        let w = batch(2, 2, 3, 3);
        let y = sum_all(&conv2d(&x, &w, None, 1, 0));
        y.backward_with(false, true);

        let gx = x.grad().expect("first derivative");
        x.cleargrad();
        w.cleargrad();
        sum_all(&gx).backward();

        // d/dW of sum(dy/dx) is a constant fold of ones, and it must exist at
        // all -- which it only does if `Conv2d::backward` built graph nodes.
        assert!(w.grad().is_some(), "the gradient flowed back through to W");
        assert_eq!(w.grad().and_then(|g| g.shape()), Some(vec![2, 2, 3, 3]));
    }

    // -- pooling -----------------------------------------------------------

    #[test]
    fn pooling_picks_the_window_maximum() {
        let x = Variable::new(
            Array::from_shape_vec((1, 1, 4, 4), (1..=16).map(f64::from).collect())
                .expect("4x4")
                .into_dyn(),
        );
        let y = pooling(&x, 2, 2, 0);
        assert_eq!(y.shape(), Some(vec![1, 1, 2, 2]));
        assert_eq!(
            y.data().expect("data").iter().copied().collect::<Vec<_>>(),
            vec![6.0, 8.0, 14.0, 16.0]
        );
    }

    #[test]
    fn pooling_routes_the_gradient_to_the_winner_only() {
        let x = Variable::new(
            Array::from_shape_vec((1, 1, 2, 2), vec![1.0, 4.0, 3.0, 2.0])
                .expect("2x2")
                .into_dyn(),
        );
        pooling(&x, 2, 1, 0).backward();
        assert_eq!(
            x.grad().and_then(|g| g.data()),
            Some(
                Array::from_shape_vec((1, 1, 2, 2), vec![0.0, 1.0, 0.0, 0.0])
                    .expect("2x2")
                    .into_dyn()
            )
        );
    }

    /// numpy's `argmax` keeps the *first* maximum; so must this, or a tie
    /// would send gradient to a different pixel than Python does.
    #[test]
    fn pooling_breaks_ties_towards_the_first_index() {
        let x = Variable::new(
            Array::from_shape_vec((1, 1, 1, 2), vec![5.0, 5.0])
                .expect("1x2")
                .into_dyn(),
        );
        pooling(&x, (1, 2), 1, 0).backward();
        assert_eq!(
            x.grad().and_then(|g| g.data()),
            Some(
                Array::from_shape_vec((1, 1, 1, 2), vec![1.0, 0.0])
                    .expect("1x2")
                    .into_dyn()
            )
        );
    }

    #[test]
    fn pooling_ignores_the_padding_when_real_values_are_larger() {
        // Padding contributes zeros; with an all-negative input the padded
        // positions would win, exactly as they do in Python.
        let x = Variable::new(ArrayD::from_elem(IxDyn(&[1, 1, 2, 2]), -1.0));
        let y = pooling(&x, 3, 1, 1);
        assert_eq!(y.shape(), Some(vec![1, 1, 2, 2]));
        assert!(
            y.data().expect("data").iter().all(|v| *v == 0.0),
            "the zero padding is the maximum of every window"
        );
    }

    #[test]
    fn pooling_gradients_match_numerical_diff() {
        for (kernel, stride, pad) in [(2, 2, 0), (3, 1, 1), (2, 1, 0)] {
            // Distinct values only: a tie makes the finite difference
            // one-sided, and the analytic gradient is genuinely discontinuous
            // there.
            gradient_check(
                |x| sum_all(&pooling(x, kernel, stride, pad)),
                &batch(1, 2, 5, 5),
                EPS,
                RTOL,
                ATOL,
            )
            .unwrap_or_else(|e| panic!("pooling k{kernel} s{stride} p{pad}: {e}"));
        }
    }

    #[test]
    fn pooling_gradient_survives_a_second_backward() {
        let x = batch(1, 2, 4, 4);
        sum_all(&pooling(&x, 2, 2, 0)).backward_with(false, true);
        let gx = x.grad().expect("first derivative");
        x.cleargrad();
        // Max pooling routes each gradient through the argmax index, and those
        // indices are constants -- so `gx` has no differentiable dependence on
        // `x` at all. The second backward must therefore run without crashing
        // and leave `x.grad` empty, rather than produce zeros.
        //
        // Verified against the reference: the same program in
        // `vendor/dezero-python` also leaves `x.grad` as `None` here.
        sum_all(&gx).backward();
        assert!(
            x.grad().is_none(),
            "no graph path runs from a pooling gradient back to its input"
        );
    }

    #[test]
    #[should_panic(expected = "pooling window must be at least 1x1")]
    fn an_empty_pooling_window_is_rejected() {
        let _ = pooling(&batch(1, 1, 4, 4), 0, 1, 0);
    }
}
