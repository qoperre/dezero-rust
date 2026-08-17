//! A Rust port of **DeZero**, the teaching autodiff framework from
//! *Deep Learning from Scratch 3*, on top of [`ndarray`].
//!
//! The reference implementation lives in `vendor/dezero-python/` and is the
//! authority on semantics; `docs/ARCHITECTURE.md` records the design decisions
//! that differ because Rust is not Python.
//!
//! # What is here
//!
//! The complete core engine — the semantics of the book's steps 1–18, but
//! built in its final shape rather than the book's incremental narrative:
//!
//! * [`Variable`] — a graph node: data, gradient, creator, generation.
//! * [`Op`] / [`Function`] / [`apply`] — an operation, one invocation of it,
//!   and the driver that connects the two.
//! * [`Variable::backward`] — generation-ordered reverse-mode differentiation.
//! * [`no_grad`] / [`test_mode`] — scoped configuration.
//! * arithmetic ([`add`], [`mul`], ...) with full operator sugar and numpy
//!   broadcasting, and the elementary functions [`square`], [`exp`], [`sin`],
//!   [`cos`], [`tanh`].
//! * tensor shape manipulation — [`reshape`], [`transpose`] — and the
//!   reductions [`sum`], [`sum_to`], [`broadcast_to`].
//!
//! On top of that, the neural-network layer of steps 41–48:
//!
//! * [`matmul`] and [`linear`], the activations [`sigmoid`], [`relu`],
//!   [`softmax`], and the losses [`mean_squared_error`],
//!   [`softmax_cross_entropy`];
//! * [`Parameter`], [`Layer`] and [`Model`] — weights, the objects that own
//!   them, and networks built by composing those;
//! * [`Optimizer`] with [`Sgd`] and [`MomentumSgd`], and the [`Hook`]s
//!   [`WeightDecay`], [`ClipGrad`] and [`FreezeParam`].
//!
//! And the data plumbing of steps 48–51:
//!
//! * [`Dataset`] and [`DataLoader`] — examples, and the mini-[`Batch`]es a
//!   training loop eats;
//! * [`Spiral`], the toy classification set, and [`Mnist`] with its
//!   [IDX](read_idx)/gzip reader.
//!
//! There is no GPU module: step 52 swaps `numpy` for `cupy` behind a
//! `get_array_module` shim, and this port has no CUDA backend to put behind
//! such a shim. It is documented in `docs/DIVERGENCES.md` rather than faked.
//!
//! # Example
//!
//! ```
//! use dezero::{square, Variable};
//!
//! let x = Variable::from_scalar(2.0);
//! let y = square(&x) * 3.0 + 1.0; // y = 3x^2 + 1
//! y.backward();
//!
//! assert_eq!(y.data(), Variable::from_scalar(13.0).data());
//! // dy/dx = 6x = 12
//! assert_eq!(x.grad().and_then(|g| g.data()), Variable::from_scalar(12.0).data());
//! ```
//!
//! # Gradients are variables
//!
//! `x.grad()` is a [`Variable`], not an array, so a gradient can be
//! differentiated again — that is what `create_graph` is for:
//!
//! ```
//! use dezero::{pow, Variable};
//!
//! let x = Variable::from_scalar(2.0);
//! let y = pow(&x, 3.0);
//! y.backward_with(false, true);
//!
//! let gx = x.grad().expect("dy/dx = 3x^2 = 12");
//! x.cleargrad();
//! gx.backward();
//! assert_eq!(x.grad().and_then(|g| g.data()), Variable::from_scalar(12.0).data()); // 6x
//! ```
//!
//! Repeat the `cleargrad` + `backward_with(.., true)` pair and the derivatives
//! keep coming — step 34's `sin` cycle, to any order:
//!
//! ```
//! use dezero::{sin, Variable};
//!
//! let x = Variable::from_scalar(1.0);
//! let y = sin(&x);
//! y.backward_with(false, true);
//!
//! let mut derivative = x.grad().expect("y'");
//! for _ in 0..3 {
//!     x.cleargrad();
//!     derivative.backward_with(false, true);
//!     derivative = x.grad().expect("the next derivative");
//! }
//! // sin -> cos -> -sin -> -cos -> sin: the fourth derivative is sin again.
//! assert!((derivative.data().expect("data").sum() - 1.0_f64.sin()).abs() < 1e-12);
//! ```
//!
//! # Broadcasting
//!
//! Differently shaped operands broadcast by numpy's rules, and each gradient is
//! folded back onto its own operand's shape with [`sum_to`]:
//!
//! ```
//! use dezero::Variable;
//! use ndarray::{arr1, arr2};
//!
//! let matrix = Variable::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn());
//! let row = Variable::new(arr1(&[10.0, 20.0, 30.0]).into_dyn());
//!
//! let y = &matrix + &row;
//! y.backward();
//!
//! // The row was used once per matrix row, so its gradient counts both.
//! assert_eq!(y.shape(), Some(vec![2, 3]));
//! assert_eq!(row.grad().and_then(|g| g.data()), Some(arr1(&[2.0, 2.0, 2.0]).into_dyn()));
//! ```
//!
//! # Training a network
//!
//! [`Layer`] and [`Optimizer`] close the loop: a model owns [`Parameter`]s, a
//! loss produces a gradient for each of them, and the optimizer moves them.
//!
//! ```
//! use dezero::{mean_squared_error, seed, Layer, Mlp, Optimizer, Sgd, Variable};
//! use ndarray::arr2;
//!
//! seed(0); // the weights are drawn at random; pin the stream to reproduce a run
//!
//! // y = 2x + 1, learned by a one-hidden-layer network.
//! let x = Variable::new(arr2(&[[0.0], [1.0], [2.0], [3.0]]).into_dyn());
//! let y = Variable::new(arr2(&[[1.0], [3.0], [5.0], [7.0]]).into_dyn());
//!
//! // Two Linear layers with a sigmoid between them, shaped by the first batch.
//! let model = Mlp::new(&[10, 1]);
//! let mut optimizer = Sgd::new(0.05);
//! optimizer.setup(&model);
//!
//! let loss_now = |model: &Mlp| {
//!     mean_squared_error(&model.forward(&x), &y).data().expect("loss").sum()
//! };
//! let before = loss_now(&model);
//!
//! for _ in 0..200 {
//!     let loss = mean_squared_error(&model.forward(&x), &y);
//!     model.cleargrads();  // backward accumulates, so clear first
//!     loss.backward();
//!     optimizer.update();
//! }
//!
//! assert!(loss_now(&model) < before / 10.0, "the fit improves by an order of magnitude");
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod core;
pub mod data;
pub mod functions;
pub mod layers;
pub mod models;
pub mod optim;
pub mod utils;

pub use crate::core::config::{ConfigGuard, enable_backprop, is_train, no_grad, test_mode};
pub use crate::core::function::{Function, FunctionInner, Op, apply, apply1};
pub use crate::core::ops::{Add, Div, Mul, Neg, Pow, Sub, add, div, mul, neg, pow, sub};
pub use crate::core::parameter::Parameter;
pub use crate::core::variable::{Variable, VariableInner};
pub use crate::data::idx::{IdxArray, IdxError, decode_idx, read_idx};
pub use crate::data::mnist::{Mnist, MnistError, cache_dir as mnist_cache_dir};
pub use crate::data::{Batch, DataLoader, Dataset, MapInput, MapLabel, Spiral};
pub use crate::functions::activation::{
    ReLU, Sigmoid, Softmax, relu, sigmoid, softmax, softmax_axis,
};
pub use crate::functions::basic_math::{Cos, Exp, Sin, Square, Tanh, cos, exp, sin, square, tanh};
pub use crate::functions::conv::{
    Col2im, Conv2d, Deconv2d, Im2col, Pooling, col2im, conv2d, deconv2d, deconv2d_with_outsize,
    im2col, pooling,
};
pub use crate::functions::loss::{
    MeanSquaredError, SoftmaxCrossEntropy, mean_squared_error, softmax_cross_entropy,
};
pub use crate::functions::matmul::{MatMul, linear, matmul};
pub use crate::functions::reduce::{
    Axes, BroadcastTo, Sum, SumTo, broadcast_to, sum, sum_all, sum_to,
};
pub use crate::functions::shape::{Reshape, Transpose, reshape, transpose};
pub use crate::layers::{Layer, Linear};
pub use crate::models::{Mlp, Model, Sequential};
pub use crate::optim::{
    ClipGrad, FreezeParam, Hook, Hooks, MomentumSgd, Optimizer, Sgd, WeightDecay,
};
pub use crate::utils::random::{Rng, rand, randn, seed};
pub use crate::utils::{GradientCheckError, GradientMismatch, gradient_check, numerical_diff};
