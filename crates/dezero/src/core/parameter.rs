//! [`Parameter`]: a [`Variable`] a [`Layer`](crate::Layer) owns and an
//! [`Optimizer`](crate::Optimizer) is allowed to move (step 44).
//!
//! Port of `class Parameter(Variable): pass` in
//! `vendor/dezero-python/dezero/core.py`. Python's entire definition is an empty
//! subclass, because the only thing it has to support is
//! `isinstance(value, Parameter)` inside `Layer.__setattr__` — the type *is* the
//! information.
//!
//! # Why a newtype and not a trait
//!
//! Rust has no inheritance, so there is no "empty subclass". The two candidates
//! are a marker trait implemented for `Variable` (which would make *every*
//! variable a parameter — useless, since the whole point is telling the two
//! apart) and a newtype. `docs/ARCHITECTURE.md` picks the newtype:
//!
//! ```text
//! pub struct Parameter(Variable);   // + Deref<Target = Variable>
//! ```
//!
//! [`Deref`] means a `Parameter` reads exactly like the `Variable` it is —
//! `p.data()`, `p.grad()`, `p.cleargrad()`, `p.shape()` all work, and `&p`
//! coerces to `&Variable` wherever a function wants one, so
//! `linear(x, &self.w, ...)` needs no ceremony. What `Deref` does *not* give is
//! the reverse: a plain `Variable` never silently becomes a `Parameter`, which
//! is precisely the distinction Python spends an `isinstance` check on.
//!
//! There is deliberately **no `DerefMut`**. Nothing here needs it: a
//! [`Variable`] is an `Rc` over interior-mutable cells, so
//! [`set_data`](Variable::set_data) and [`cleargrad`](Variable::cleargrad)
//! already work through a shared reference — which is what lets a layer update
//! its own weights from `&self`.

use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use ndarray::ArrayD;

use crate::core::variable::Variable;

/// A [`Variable`] that a layer owns and an optimizer may update.
///
/// Cloning clones the underlying `Rc`: a clone *is* the same parameter, so a
/// layer and an optimizer can each hold one and see each other's writes. That
/// is what makes [`Optimizer::setup`](crate::Optimizer::setup) able to snapshot
/// a parameter list without copying any weights.
///
/// # Examples
///
/// ```
/// use dezero::Parameter;
/// use ndarray::arr2;
///
/// let w = Parameter::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn());
///
/// // Everything a Variable can do, a Parameter can do.
/// assert_eq!(w.shape(), Some(vec![2, 2]));
/// assert!(w.grad().is_none());
///
/// // Clones share identity, so an update through one is visible through both.
/// let alias = w.clone();
/// assert_eq!(w.id(), alias.id());
/// ```
#[derive(Clone, PartialEq, Eq)]
pub struct Parameter(Variable);

impl Parameter {
    /// Creates a parameter holding `data`.
    #[must_use]
    pub fn new(data: ArrayD<f64>) -> Self {
        Self(Variable::new(data))
    }

    /// Creates a parameter with no data yet — Python's `Parameter(None)`.
    ///
    /// This is the lazily-shaped case: [`Linear`](crate::Linear) constructs its
    /// `W` this way when no `in_size` was given, and fills it in on the first
    /// forward pass, once the input reveals the shape. The parameter's
    /// *identity* is fixed from construction, so an optimizer that registered it
    /// while it was still empty picks up the weights when they arrive.
    #[must_use]
    pub fn empty() -> Self {
        Self(Variable::empty())
    }

    /// Creates a named parameter — Python's `Parameter(data, name='W')`.
    ///
    /// The name is carried on the variable and is what a hierarchical
    /// save/load format will key on when that step lands.
    #[must_use]
    pub fn named(data: Option<ArrayD<f64>>, name: &str) -> Self {
        let parameter = match data {
            Some(data) => Self::new(data),
            None => Self::empty(),
        };
        parameter.set_name(Some(name.to_owned()));
        parameter
    }

    /// Promotes an existing variable to a parameter, sharing its identity.
    ///
    /// The variable and the parameter are the same node afterwards; this only
    /// changes how the type system sees it.
    #[must_use]
    pub fn wrap(variable: Variable) -> Self {
        Self(variable)
    }

    /// Borrows this parameter as the variable it is.
    ///
    /// Usually unnecessary — [`Deref`] coerces `&Parameter` to `&Variable`
    /// automatically at call sites. Reach for this when inference needs the
    /// help, for instance inside a generic collection.
    #[must_use]
    pub fn as_variable(&self) -> &Variable {
        &self.0
    }

    /// Consumes the parameter and returns the underlying variable.
    #[must_use]
    pub fn into_variable(self) -> Variable {
        self.0
    }
}

impl Deref for Parameter {
    type Target = Variable;

    fn deref(&self) -> &Variable {
        &self.0
    }
}

impl From<Variable> for Parameter {
    fn from(variable: Variable) -> Self {
        Self::wrap(variable)
    }
}

impl From<Parameter> for Variable {
    fn from(parameter: Parameter) -> Self {
        parameter.into_variable()
    }
}

impl Hash for Parameter {
    /// Pointer identity, like [`Variable`]'s — a `Parameter` is usable as the
    /// key of an optimizer's per-parameter state map.
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl fmt::Display for Parameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl fmt::Debug for Parameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Parameter").field(&self.0).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{square, sum_all};
    use ndarray::{arr1, arr2};
    use std::collections::HashSet;

    #[test]
    fn a_parameter_is_the_variable_it_wraps() {
        let p = Parameter::new(arr1(&[1.0, 2.0, 3.0]).into_dyn());
        assert_eq!(p.shape(), Some(vec![3]));
        assert_eq!(p.ndim(), Some(1));
        assert_eq!(p.size(), Some(3));
        assert!(p.creator().is_none());
    }

    #[test]
    fn empty_supports_the_lazily_shaped_case() {
        let w = Parameter::empty();
        assert!(w.data().is_none());
        assert!(w.shape().is_none());

        // The identity survives being filled in later, which is what lets an
        // optimizer register a parameter before its shape is known.
        let id = w.id();
        w.set_data(arr2(&[[1.0, 2.0]]).into_dyn());
        assert_eq!(w.id(), id);
        assert_eq!(w.shape(), Some(vec![1, 2]));
    }

    #[test]
    fn named_carries_the_name_onto_the_variable() {
        let w = Parameter::named(None, "W");
        assert_eq!(w.name().as_deref(), Some("W"));
        assert!(w.data().is_none());

        let b = Parameter::named(Some(arr1(&[0.0, 0.0]).into_dyn()), "b");
        assert_eq!(b.name().as_deref(), Some("b"));
        assert_eq!(b.shape(), Some(vec![2]));
    }

    #[test]
    fn clones_share_identity_and_state() {
        let p = Parameter::new(arr1(&[1.0]).into_dyn());
        let alias = p.clone();
        assert_eq!(p, alias);
        assert_eq!(p.id(), alias.id());

        alias.set_data(arr1(&[9.0]).into_dyn());
        assert_eq!(p.data(), Some(arr1(&[9.0]).into_dyn()));
    }

    #[test]
    fn distinct_parameters_with_equal_data_are_not_equal() {
        let a = Parameter::new(arr1(&[1.0]).into_dyn());
        let b = Parameter::new(arr1(&[1.0]).into_dyn());
        assert_ne!(a, b);
        assert_eq!(a.data(), b.data());
    }

    #[test]
    #[allow(
        clippy::mutable_key_type,
        reason = "that Parameter is a usable HashSet key despite its RefCells is \
                  exactly what this test pins down: Hash and Eq only ever look \
                  at the Rc pointer, never at the data behind it"
    )]
    fn parameters_work_as_hash_map_keys() {
        let a = Parameter::new(arr1(&[1.0]).into_dyn());
        let b = Parameter::new(arr1(&[1.0]).into_dyn());

        let mut seen: HashSet<Parameter> = HashSet::new();
        assert!(seen.insert(a.clone()));
        assert!(!seen.insert(a), "a clone is the same key");
        assert!(seen.insert(b));
        assert_eq!(seen.len(), 2);
    }

    /// The point of `Deref`: a parameter goes straight into the graph, with a
    /// gradient landing on it like any other leaf.
    #[test]
    fn a_parameter_participates_in_the_graph() {
        let w = Parameter::new(arr1(&[2.0, 3.0]).into_dyn());
        let y = sum_all(&square(&w));
        y.backward();
        assert_eq!(
            w.grad().and_then(|g| g.data()),
            Some(arr1(&[4.0, 6.0]).into_dyn())
        );

        w.cleargrad();
        assert!(w.grad().is_none());
    }

    #[test]
    fn conversions_round_trip_without_copying() {
        let v = Variable::new(arr1(&[1.0, 2.0]).into_dyn());
        let id = v.id();

        let p = Parameter::from(v.clone());
        assert_eq!(p.id(), id);
        assert_eq!(p.as_variable(), &v);

        let back: Variable = p.into();
        assert_eq!(back.id(), id);
    }

    #[test]
    fn rendering_matches_the_underlying_variable() {
        let p = Parameter::new(ndarray::arr0(2.0).into_dyn());
        assert_eq!(p.to_string(), "variable(2)");
        assert!(format!("{p:?}").starts_with("Parameter(Variable {"));
    }
}
