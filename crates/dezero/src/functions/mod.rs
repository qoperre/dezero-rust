//! Differentiable functions built on the core engine.
//!
//! Port of `vendor/dezero-python/dezero/functions.py`, one module per section:
//!
//! * [`basic_math`] — `sin`, `cos`, `tanh`, `exp`, `square` (steps 2–35);
//! * [`shape`] — `reshape`, `transpose` (step 37);
//! * [`reduce`] — `sum`, `sum_to`, `broadcast_to` (steps 38–40);
//! * [`matmul`] — `matmul` and the affine `linear` (steps 41–43);
//! * [`activation`] — `sigmoid`, `relu`, `softmax` (steps 43, 47);
//! * [`loss`] — `mean_squared_error`, `softmax_cross_entropy` (steps 42, 47);
//! * [`conv`] — `im2col`, `col2im`, `conv2d`, `deconv2d`, `pooling` (step 57).
//!
//! The normalisation family arrives with a later step.

pub mod activation;
pub mod basic_math;
pub mod conv;
pub mod loss;
pub mod matmul;
pub mod reduce;
pub mod shape;
