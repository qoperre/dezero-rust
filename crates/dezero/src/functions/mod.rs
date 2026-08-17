//! Differentiable functions built on the core engine.
//!
//! Port of `vendor/dezero-python/dezero/functions.py`, one module per section:
//!
//! * [`basic_math`] — `sin`, `cos`, `tanh`, `exp`, `square` (steps 2–35);
//! * [`shape`] — `reshape`, `transpose` (step 37);
//! * [`reduce`] — `sum`, `sum_to`, `broadcast_to` (steps 38–40).
//!
//! `matmul`, the activations and the loss functions arrive with step 41
//! onwards.

pub mod basic_math;
pub mod reduce;
pub mod shape;
