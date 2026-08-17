//! Global configuration flags, mirroring Python's `dezero.core.Config`.
//!
//! Python uses class attributes on `Config` plus a `contextmanager`
//! (`using_config`) to scope changes. The Rust port uses `thread_local!`
//! storage plus an RAII [`ConfigGuard`] that restores the previous value when
//! it is dropped:
//!
//! ```
//! use dezero::{enable_backprop, no_grad};
//!
//! assert!(enable_backprop());
//! {
//!     let _guard = no_grad();
//!     assert!(!enable_backprop());
//! }
//! assert!(enable_backprop());
//! ```
//!
//! `thread_local!` rather than a global is deliberate: `cargo test` runs tests
//! on multiple threads, and a process-global flag would let one test's
//! `no_grad()` leak into another. Python has no such hazard, so this is a place
//! where the port is safer than the reference at no cost.

use std::cell::Cell;

thread_local! {
    /// Whether `apply` should record the computational graph (Python:
    /// `Config.enable_backprop`).
    static ENABLE_BACKPROP: Cell<bool> = const { Cell::new(true) };

    /// Whether the library is in training mode (Python: `Config.train`).
    static TRAIN: Cell<bool> = const { Cell::new(true) };
}

/// Identifies which configuration flag a [`ConfigGuard`] restores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flag {
    EnableBackprop,
    Train,
}

impl Flag {
    /// Reads the current value of this flag.
    ///
    /// Falls back to the flag's default if the thread-local store has already
    /// been destroyed (only possible while a thread is shutting down).
    fn get(self) -> bool {
        match self {
            Self::EnableBackprop => ENABLE_BACKPROP.try_with(Cell::get).unwrap_or(true),
            Self::Train => TRAIN.try_with(Cell::get).unwrap_or(true),
        }
    }

    /// Writes `value` to this flag, returning the value it replaced.
    fn replace(self, value: bool) -> bool {
        match self {
            Self::EnableBackprop => ENABLE_BACKPROP
                .try_with(|cell| cell.replace(value))
                .unwrap_or(true),
            Self::Train => TRAIN.try_with(|cell| cell.replace(value)).unwrap_or(true),
        }
    }
}

/// Returns `true` when `apply` should build the backward graph.
///
/// Python: `Config.enable_backprop`.
#[must_use]
pub fn enable_backprop() -> bool {
    Flag::EnableBackprop.get()
}

/// Returns `true` when the library is in training mode.
///
/// Python: `Config.train`. Consumed by dropout and batch-norm in later steps.
#[must_use]
pub fn is_train() -> bool {
    Flag::Train.get()
}

/// RAII guard that restores a configuration flag when dropped.
///
/// Python's `using_config` context manager; holding the guard is the
/// equivalent of being inside the `with` block. The guard must be bound to a
/// variable — `let _guard = no_grad();`. Writing `let _ = no_grad();` drops it
/// immediately and has no effect, which is why the type is `#[must_use]`.
#[must_use = "the flag is restored when the guard is dropped; binding it to `_` \
              drops it immediately and has no effect"]
#[derive(Debug)]
pub struct ConfigGuard {
    flag: Flag,
    previous: bool,
}

impl ConfigGuard {
    /// Sets `flag` to `value` and captures the previous value for restoration.
    fn new(flag: Flag, value: bool) -> Self {
        let previous = flag.replace(value);
        Self { flag, previous }
    }
}

impl Drop for ConfigGuard {
    fn drop(&mut self) {
        self.flag.replace(self.previous);
    }
}

/// Disables graph construction for the lifetime of the returned guard.
///
/// Python: `dezero.no_grad()`.
///
/// ```
/// use dezero::{no_grad, square, Variable};
///
/// let x = Variable::from_scalar(3.0);
/// let y = {
///     let _guard = no_grad();
///     square(&x)
/// };
/// assert!(y.creator().is_none());
/// ```
pub fn no_grad() -> ConfigGuard {
    ConfigGuard::new(Flag::EnableBackprop, false)
}

/// Switches to inference mode for the lifetime of the returned guard.
///
/// Python: `dezero.test_mode()`.
pub fn test_mode() -> ConfigGuard {
    ConfigGuard::new(Flag::Train, false)
}

/// Forces `enable_backprop` to `value` for the lifetime of the returned guard.
///
/// This is the `with using_config('enable_backprop', create_graph)` wrapper
/// that `Variable::backward` puts around each `Op::backward` call; it is not
/// part of the public surface because `no_grad()` covers every user-facing
/// case.
pub(crate) fn using_enable_backprop(value: bool) -> ConfigGuard {
    ConfigGuard::new(Flag::EnableBackprop, value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_training_with_backprop() {
        assert!(enable_backprop());
        assert!(is_train());
    }

    #[test]
    fn no_grad_scopes_and_restores() {
        assert!(enable_backprop());
        {
            let _guard = no_grad();
            assert!(!enable_backprop());
        }
        assert!(enable_backprop());
    }

    #[test]
    fn test_mode_scopes_and_restores() {
        assert!(is_train());
        {
            let _guard = test_mode();
            assert!(!is_train());
            // The other flag is untouched.
            assert!(enable_backprop());
        }
        assert!(is_train());
    }

    #[test]
    fn guards_nest_and_restore_in_reverse_order() {
        let outer = no_grad();
        assert!(!enable_backprop());
        {
            let _inner = using_enable_backprop(true);
            assert!(enable_backprop());
        }
        assert!(!enable_backprop());
        drop(outer);
        assert!(enable_backprop());
    }
}
