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
//! * arithmetic ([`add`], [`mul`], ...) with full operator sugar, and
//!   [`square`] / [`exp`].
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
//! # Not yet implemented
//!
//! Broadcasting between differently shaped operands (step 40) is rejected with
//! a panic rather than silently producing wrong gradients; see
//! [`core::ops`].

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod core;
pub mod functions;
pub mod utils;

pub use crate::core::config::{ConfigGuard, enable_backprop, is_train, no_grad, test_mode};
pub use crate::core::function::{Function, FunctionInner, Op, apply, apply1};
pub use crate::core::ops::{Add, Div, Mul, Neg, Pow, Sub, add, div, mul, neg, pow, sub};
pub use crate::core::variable::{Variable, VariableInner};
pub use crate::functions::basic_math::{Exp, Square, exp, square};
pub use crate::utils::{GradientCheckError, GradientMismatch, gradient_check, numerical_diff};
