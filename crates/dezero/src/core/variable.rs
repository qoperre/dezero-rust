//! The [`Variable`] node type: a reference-counted box around an `ndarray`
//! value plus its place in the computational graph.
//!
//! Port of `Variable` in `vendor/dezero-python/dezero/core.py`.
//!
//! # Shape of the port
//!
//! ```text
//! Variable = Rc<VariableInner>        // cheap to clone; clones share identity
//! ```
//!
//! Python's attribute assignment (`x.grad = ...`) happens through shared
//! references, so every mutable field lives behind a `RefCell`/`Cell`. To keep
//! those borrows from straddling user code — which would turn an innocuous
//! call sequence into a runtime `already borrowed` panic — **no public
//! accessor ever hands out a live `Ref`**. Each one clones the value out of the
//! cell and returns an owned snapshot.

use std::cell::{Cell, Ref, RefCell};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::{Rc, Weak};

use ndarray::{ArrayD, IxDyn, arr0};

use crate::core::function::Function;

/// The heap payload of a [`Variable`].
///
/// Public only because a [`Weak`] handle to it appears in [`Function`]'s
/// output list — the weak edge that breaks the
/// `Variable -> Function -> Variable` cycle. All fields are private; go
/// through [`Variable`]'s methods.
#[derive(Debug)]
pub struct VariableInner {
    data: RefCell<Option<ArrayD<f64>>>,
    grad: RefCell<Option<Variable>>,
    creator: RefCell<Option<Function>>,
    generation: Cell<u32>,
    name: RefCell<Option<String>>,
}

/// A node in the computational graph.
///
/// Cloning a `Variable` clones the `Rc`, not the data: the clone *is* the same
/// node, exactly like passing a Python object around. Equality and hashing use
/// pointer identity (Python's `id()`), never the numeric contents.
#[derive(Clone)]
pub struct Variable(Rc<VariableInner>);

impl Variable {
    /// Creates a variable holding `data`.
    ///
    /// # Examples
    ///
    /// ```
    /// use dezero::Variable;
    /// use ndarray::arr1;
    ///
    /// let x = Variable::new(arr1(&[1.0, 2.0, 3.0]).into_dyn());
    /// assert_eq!(x.shape(), Some(vec![3]));
    /// ```
    #[must_use]
    pub fn new(data: ArrayD<f64>) -> Self {
        Self::from_option(Some(data))
    }

    /// Creates a variable with no data yet — Python's `Variable(None)`.
    ///
    /// Used by lazily-shaped parameters (`Linear.W` before the first forward
    /// pass, step 44+), which is why [`data`](Self::data) is an `Option` from
    /// the very first commit rather than being retrofitted later.
    #[must_use]
    pub fn empty() -> Self {
        Self::from_option(None)
    }

    /// Creates a variable holding a 0-dimensional array — Python's
    /// `Variable(np.array(2.0))`.
    #[must_use]
    pub fn from_scalar(value: f64) -> Self {
        Self::new(arr0(value).into_dyn())
    }

    /// Creates a detached variable of `self`'s shape, filled with `value`.
    ///
    /// Returns `None` when `self` holds no data. Detached: the result has no
    /// creator, so backpropagation stops there.
    #[must_use]
    pub fn full_like(&self, value: f64) -> Option<Self> {
        let shape = self.shape()?;
        Some(Self::new(ArrayD::from_elem(IxDyn(&shape), value)))
    }

    /// Shared construction path for [`new`](Self::new) and
    /// [`empty`](Self::empty).
    fn from_option(data: Option<ArrayD<f64>>) -> Self {
        Self(Rc::new(VariableInner {
            data: RefCell::new(data),
            grad: RefCell::new(None),
            creator: RefCell::new(None),
            generation: Cell::new(0),
            name: RefCell::new(None),
        }))
    }

    /// Rebuilds the newtype around an inner pointer obtained from a
    /// [`Weak`] upgrade.
    pub(crate) fn from_inner(inner: Rc<VariableInner>) -> Self {
        Self(inner)
    }

    /// Returns a weak handle to this node, for `Function`'s output list.
    pub(crate) fn downgrade(&self) -> Weak<VariableInner> {
        Rc::downgrade(&self.0)
    }

    /// Borrows the data in place, without cloning it.
    ///
    /// `pub(crate)` on purpose: the borrow is live, so it may only be held
    /// across code this crate controls (`apply`'s forward call). Public
    /// accessors always return owned snapshots instead.
    pub(crate) fn borrow_data(&self) -> Ref<'_, Option<ArrayD<f64>>> {
        self.0.data.borrow()
    }

    /// Pointer identity — the port's equivalent of Python's `id()`.
    ///
    /// Stable for the lifetime of the node and used for graph node ids and
    /// per-parameter optimizer state.
    #[must_use]
    pub fn id(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }

    /// Returns an owned copy of the data, or `None` if unset.
    ///
    /// This clones the array. Callers that only need metadata should use
    /// [`shape`](Self::shape), [`ndim`](Self::ndim) or [`size`](Self::size),
    /// which do not.
    #[must_use]
    pub fn data(&self) -> Option<ArrayD<f64>> {
        self.0.data.borrow().clone()
    }

    /// Replaces the data.
    pub fn set_data(&self, data: ArrayD<f64>) {
        *self.0.data.borrow_mut() = Some(data);
    }

    /// Returns a handle to the gradient, or `None` if it has not been computed.
    ///
    /// The gradient is itself a [`Variable`] (that is what makes
    /// double-backpropagation possible), and the returned handle shares
    /// identity with the stored one — mutating it mutates the gradient.
    #[must_use]
    pub fn grad(&self) -> Option<Self> {
        self.0.grad.borrow().clone()
    }

    /// Sets (or with `None`, clears) the gradient.
    pub fn set_grad(&self, grad: Option<Self>) {
        *self.0.grad.borrow_mut() = grad;
    }

    /// Clears the gradient — Python's `cleargrad()`.
    ///
    /// Call between iterations of a training loop, or before reusing a
    /// variable in a second graph, since [`backward`](Self::backward)
    /// *accumulates* into `grad`.
    pub fn cleargrad(&self) {
        self.set_grad(None);
    }

    /// Returns the function that produced this variable, if any.
    ///
    /// Leaf variables (user inputs, parameters, anything created under
    /// [`no_grad`](crate::no_grad)) have no creator.
    #[must_use]
    pub fn creator(&self) -> Option<Function> {
        self.0.creator.borrow().clone()
    }

    /// Records `func` as this variable's creator and places the node one
    /// generation after it.
    ///
    /// Python: `set_creator`. The generation is what orders the backward pass
    /// (step 16), so it must be set together with the creator.
    pub fn set_creator(&self, func: &Function) {
        self.0.generation.set(func.generation().saturating_add(1));
        *self.0.creator.borrow_mut() = Some(func.clone());
    }

    /// Returns the node's topological generation; 0 for leaves.
    #[must_use]
    pub fn generation(&self) -> u32 {
        self.0.generation.get()
    }

    /// Severs the link to this variable's creator — Python's `unchain()`.
    ///
    /// Everything upstream that nothing else holds is dropped immediately.
    pub fn unchain(&self) {
        *self.0.creator.borrow_mut() = None;
    }

    /// Severs every link upstream of this variable — Python's
    /// `unchain_backward()`.
    ///
    /// Truncated backpropagation through time (steps 59–60) uses this to free
    /// the history of a recurrent graph while keeping the current activations.
    pub fn unchain_backward(&self) {
        let Some(creator) = self.creator() else {
            return;
        };
        let mut stack = vec![creator];
        while let Some(func) = stack.pop() {
            for input in func.inputs() {
                if let Some(upstream) = input.creator() {
                    stack.push(upstream);
                    input.unchain();
                }
            }
        }
    }

    /// Returns the variable's name, used when rendering graphs.
    #[must_use]
    pub fn name(&self) -> Option<String> {
        self.0.name.borrow().clone()
    }

    /// Sets (or with `None`, clears) the variable's name.
    pub fn set_name(&self, name: Option<String>) {
        *self.0.name.borrow_mut() = name;
    }

    /// Returns the data's shape, or `None` if there is no data.
    #[must_use]
    pub fn shape(&self) -> Option<Vec<usize>> {
        self.0.data.borrow().as_ref().map(|d| d.shape().to_vec())
    }

    /// Returns the number of dimensions, or `None` if there is no data.
    #[must_use]
    pub fn ndim(&self) -> Option<usize> {
        self.0.data.borrow().as_ref().map(ArrayD::ndim)
    }

    /// Returns the total number of elements, or `None` if there is no data.
    #[must_use]
    pub fn size(&self) -> Option<usize> {
        self.0.data.borrow().as_ref().map(ArrayD::len)
    }

    /// Returns the length of the leading axis — Python's `__len__`.
    ///
    /// `None` when there is no data or the data is 0-dimensional (numpy raises
    /// `TypeError: len() of unsized object` in that case).
    #[must_use]
    #[allow(
        clippy::len_without_is_empty,
        reason = "mirrors Python's __len__ (leading-axis length); \"empty\" is \
                  not a meaningful predicate on a Variable, and a 0-d variable \
                  has no length at all"
    )]
    pub fn len(&self) -> Option<usize> {
        self.0
            .data
            .borrow()
            .as_ref()
            .and_then(|d| d.shape().first().copied())
    }
}

impl PartialEq for Variable {
    /// Pointer identity, never numeric equality — Python's `is`.
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Variable {}

impl Hash for Variable {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}

impl fmt::Display for Variable {
    /// Python's `Variable.__repr__`: `variable(...)` with continuation lines
    /// indented to line up under the opening parenthesis.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.data() {
            None => write!(f, "variable(None)"),
            Some(data) => {
                let body = data.to_string().replace('\n', "\n         ");
                write!(f, "variable({body})")
            }
        }
    }
}

impl fmt::Debug for Variable {
    /// Deliberately shallow: it reports whether a creator/gradient exists
    /// rather than recursing into them, so debug-printing a node near the end
    /// of a deep graph cannot dump the whole graph.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Variable")
            .field("id", &self.id())
            .field("name", &self.name())
            .field("shape", &self.shape())
            .field("generation", &self.generation())
            .field("has_grad", &self.grad().is_some())
            .field("has_creator", &self.creator().is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn new_stores_data_and_metadata() {
        let x = Variable::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn());
        assert_eq!(x.shape(), Some(vec![2, 3]));
        assert_eq!(x.ndim(), Some(2));
        assert_eq!(x.size(), Some(6));
        assert_eq!(x.len(), Some(2));
        assert!(x.grad().is_none());
        assert!(x.creator().is_none());
        assert_eq!(x.generation(), 0);
    }

    #[test]
    fn empty_variable_has_no_data() {
        let x = Variable::empty();
        assert!(x.data().is_none());
        assert!(x.shape().is_none());
        assert!(x.ndim().is_none());
        assert!(x.size().is_none());
        assert!(x.len().is_none());
        assert_eq!(x.to_string(), "variable(None)");
    }

    #[test]
    fn scalar_has_no_len() {
        let x = Variable::from_scalar(2.0);
        assert_eq!(x.ndim(), Some(0));
        assert_eq!(x.size(), Some(1));
        assert_eq!(x.len(), None);
    }

    #[test]
    fn full_like_matches_shape_and_is_detached() {
        let x = Variable::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn());
        let ones = x.full_like(1.0).unwrap();
        assert_eq!(ones.shape(), x.shape());
        assert!(ones.data().unwrap().iter().all(|v| *v == 1.0));
        assert!(ones.creator().is_none());
        assert!(Variable::empty().full_like(1.0).is_none());
    }

    #[test]
    fn clone_shares_identity_and_state() {
        let x = Variable::from_scalar(1.0);
        let alias = x.clone();
        assert_eq!(x, alias);
        assert_eq!(x.id(), alias.id());

        alias.set_data(arr0(7.0).into_dyn());
        assert_eq!(x.data(), Some(arr0(7.0).into_dyn()));
    }

    #[test]
    fn distinct_variables_with_equal_data_are_not_equal() {
        let a = Variable::from_scalar(1.0);
        let b = Variable::from_scalar(1.0);
        assert_ne!(a, b);
        assert_eq!(a.data(), b.data());
    }

    #[test]
    fn accessors_return_snapshots_not_live_borrows() {
        let x = Variable::new(arr1(&[1.0, 2.0]).into_dyn());
        // Holding the snapshot while mutating must not panic; that is the
        // whole point of never returning a `Ref`.
        let snapshot = x.data();
        x.set_data(arr1(&[9.0, 9.0]).into_dyn());
        assert_eq!(snapshot, Some(arr1(&[1.0, 2.0]).into_dyn()));
        assert_eq!(x.data(), Some(arr1(&[9.0, 9.0]).into_dyn()));
    }

    #[test]
    fn name_round_trips() {
        let x = Variable::from_scalar(1.0);
        assert!(x.name().is_none());
        x.set_name(Some("x".to_owned()));
        assert_eq!(x.name().as_deref(), Some("x"));
        x.set_name(None);
        assert!(x.name().is_none());
    }

    #[test]
    fn cleargrad_resets_gradient() {
        let x = Variable::from_scalar(1.0);
        x.set_grad(Some(Variable::from_scalar(5.0)));
        assert!(x.grad().is_some());
        x.cleargrad();
        assert!(x.grad().is_none());
    }

    #[test]
    fn unchain_detaches_one_node() {
        let x = Variable::from_scalar(2.0);
        let y = crate::square(&x);
        assert!(y.creator().is_some());

        y.unchain();
        assert!(y.creator().is_none());

        y.backward();
        assert!(x.grad().is_none(), "nothing upstream is reachable any more");
    }

    #[test]
    fn unchain_backward_severs_the_history_but_keeps_the_last_step() {
        // Python unchains the *inputs* of each visited function, so the node it
        // is called on keeps its own creator.
        let x = Variable::from_scalar(2.0);
        let a = crate::square(&x);
        let b = crate::square(&a);
        let y = crate::square(&b);

        y.unchain_backward();

        assert!(y.creator().is_some());
        assert!(b.creator().is_none());
        assert!(a.creator().is_none());

        y.backward();
        assert!(b.grad().is_some(), "one step of backprop still runs");
        assert!(a.grad().is_none());
        assert!(x.grad().is_none());
    }

    #[test]
    fn unchain_backward_on_a_leaf_is_a_no_op() {
        let x = Variable::from_scalar(2.0);
        x.unchain_backward();
        assert!(x.creator().is_none());
    }

    #[test]
    fn unchain_backward_frees_the_upstream_graph() {
        let x = Variable::from_scalar(2.0);
        let (y, producer_of_a) = {
            let a = crate::square(&x);
            // Watch `a` through the function that produced it: the link is
            // weak, so it reports exactly when `a` is really gone.
            let producer = a.creator().expect("creator");
            let b = crate::square(&a);
            let y = crate::square(&b);
            (y, producer)
        };
        assert!(
            producer_of_a.outputs()[0].is_some(),
            "the graph itself still holds a, even though the handle is gone"
        );

        y.unchain_backward();
        assert!(
            producer_of_a.outputs()[0].is_none(),
            "unchaining must actually release the history, not merely hide it"
        );
    }

    #[test]
    fn display_matches_python_repr_layout() {
        let x = Variable::from_scalar(2.0);
        assert_eq!(x.to_string(), "variable(2)");

        let m = Variable::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn());
        let rendered = m.to_string();
        assert!(rendered.starts_with("variable("));
        assert!(rendered.ends_with(')'));
        // Continuation lines are indented under the opening parenthesis.
        assert!(rendered.contains("\n         "));
    }
}
