//! The autodiff engine: variables, functions, the backward pass, the
//! configuration flags they consult, and the arithmetic built on top of them.
//!
//! Port of `vendor/dezero-python/dezero/core.py`, split into one module per
//! concern. The layout follows `docs/ARCHITECTURE.md`.

pub mod backward;
pub mod config;
pub mod function;
pub mod ops;
pub mod variable;
