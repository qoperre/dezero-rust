//! A minimal Rust port of the core `DeZero` types, following the
//! "Deep Learning from Scratch 3" book (steps 1-2).
//!
//! This crate currently implements only the `Variable` container and the
//! `square` function, as a first slice to prove out the Python/Rust
//! parity-testing harness end-to-end.

use ndarray::ArrayD;

/// A container for an n-dimensional array of `f64` values.
///
/// Mirrors the Python `Variable` class from `DeZero`.
#[derive(Debug, Clone, PartialEq)]
pub struct Variable {
    /// The underlying n-dimensional data.
    pub data: ArrayD<f64>,
}

impl Variable {
    /// Creates a new `Variable` wrapping the given data.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::arr1;
    /// use dezero::Variable;
    ///
    /// let v = Variable::new(arr1(&[1.0, 2.0, 3.0]).into_dyn());
    /// assert_eq!(v.data.len(), 3);
    /// ```
    #[must_use]
    pub fn new(data: ArrayD<f64>) -> Self {
        Self { data }
    }
}

/// Computes the elementwise square of `x`.
///
/// Mirrors the Python `Square` `Function` from `DeZero`: `y = x ** 2`.
///
/// # Examples
///
/// ```
/// use ndarray::arr1;
/// use dezero::{square, Variable};
///
/// let x = Variable::new(arr1(&[1.0, 2.0, 3.0]).into_dyn());
/// let y = square(&x);
/// assert_eq!(y.data, arr1(&[1.0, 4.0, 9.0]).into_dyn());
/// ```
#[must_use]
pub fn square(x: &Variable) -> Variable {
    Variable::new(x.data.mapv(|v| v * v))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr0, arr1};

    #[test]
    fn square_scalar() {
        let x = Variable::new(arr0(3.0).into_dyn());
        let y = square(&x);
        assert_eq!(y.data, arr0(9.0).into_dyn());
    }

    #[test]
    fn square_1d() {
        let x = Variable::new(arr1(&[1.0, -2.0, 3.0]).into_dyn());
        let y = square(&x);
        assert_eq!(y.data, arr1(&[1.0, 4.0, 9.0]).into_dyn());
    }
}
