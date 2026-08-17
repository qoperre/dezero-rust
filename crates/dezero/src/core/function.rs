//! The [`Op`] trait, the [`Function`] graph node, and the [`apply`] driver.
//!
//! Port of `Function` in `vendor/dezero-python/dezero/core.py`. Python splits
//! the roles across a single class: `Function.__call__` drives the machinery
//! while subclasses supply `forward`/`backward`. Rust splits them in two:
//!
//! * [`Op`] — the mathematics, implemented once per operation.
//! * [`Function`] — one *invocation* of an op, i.e. the graph node holding the
//!   inputs, the weak output links and the generation number.
//!
//! [`apply`] is `Function.__call__`.

use std::cell::RefCell;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::{Rc, Weak};

use ndarray::ArrayD;

use crate::core::config;
use crate::core::variable::{Variable, VariableInner};

/// A differentiable operation.
///
/// # The one rule that matters
///
/// > `backward` operates on [`Variable`]s, never on raw `ArrayD`.
///
/// `forward` receives plain arrays because it is the boundary where the graph
/// is entered; `backward` must build *new graph nodes* by calling the same
/// free functions user code calls (`mul`, `add`, ...). Doing raw `ndarray`
/// arithmetic there would produce numerically correct first derivatives and
/// make higher-order derivatives (steps 33–35) impossible without rewriting
/// every op. See `docs/ARCHITECTURE.md`.
///
/// # Example
///
/// ```
/// use dezero::{apply1, Op, Variable};
/// use ndarray::ArrayD;
///
/// #[derive(Debug)]
/// struct Double;
///
/// impl Op for Double {
///     fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>> {
///         vec![xs[0] * 2.0]
///     }
///
///     fn backward(
///         &self,
///         _inputs: &[Variable],
///         _outputs: &[Variable],
///         gys: &[Variable],
///     ) -> Vec<Variable> {
///         vec![&gys[0] * 2.0]
///     }
/// }
///
/// let x = Variable::from_scalar(3.0);
/// let y = apply1(Double, &[&x]);
/// y.backward();
/// assert_eq!(y.data(), Variable::from_scalar(6.0).data());
/// assert_eq!(x.grad().and_then(|g| g.data()), Variable::from_scalar(2.0).data());
/// ```
pub trait Op: fmt::Debug {
    /// Computes the outputs from the input arrays.
    ///
    /// Takes `&mut self` so an op may cache state computed here for use in
    /// `backward` (Python does this by assigning to `self` inside `forward`).
    fn forward(&mut self, xs: &[&ArrayD<f64>]) -> Vec<ArrayD<f64>>;

    /// Computes input gradients from output gradients.
    ///
    /// `inputs` and `outputs` are the variables of *this* invocation, matching
    /// Python's `self.inputs` / `self.outputs`; `gys` has one entry per output.
    /// The returned vector must have one entry per input.
    ///
    /// An output the user has already dropped cannot be resurrected (Python
    /// raises `AttributeError` on the dead weakref); it is passed here as a
    /// detached zero-filled variable of the right shape instead.
    fn backward(
        &self,
        inputs: &[Variable],
        outputs: &[Variable],
        gys: &[Variable],
    ) -> Vec<Variable>;
}

/// The heap payload of a [`Function`].
///
/// Public only so that [`Function`]'s documentation can refer to its fields;
/// all of them are private.
pub struct FunctionInner {
    op: Box<dyn Op>,
    /// Strong: an op's inputs must outlive it, so the backward pass can read
    /// them.
    inputs: Vec<Variable>,
    /// **Weak** — this is the edge that breaks the
    /// `Variable -> creator -> inputs -> Variable` reference cycle, mirroring
    /// Python's `weakref.ref(output)`.
    outputs: RefCell<Vec<Weak<VariableInner>>>,
    /// Shapes of the outputs, recorded at construction time.
    ///
    /// Deviation from `docs/ARCHITECTURE.md`'s field list, deliberately
    /// additive: once an output has been dropped, its weak link cannot tell us
    /// what shape its zero gradient should be. Python simply crashes in that
    /// situation; recording `usize` shapes costs nothing and lets the backward
    /// pass substitute a correctly shaped zero.
    output_shapes: Vec<Vec<usize>>,
    generation: u32,
}

/// One invocation of an [`Op`]: a node in the computational graph.
///
/// Cloning clones the `Rc`; equality and hashing are pointer identity, so a
/// `Function` can be used as a `HashSet` key the way Python uses `id()`.
#[derive(Clone)]
pub struct Function(Rc<FunctionInner>);

impl Function {
    /// Builds the graph node. Called only by [`apply`], after `forward` has
    /// run and the output variables exist.
    fn new(op: Box<dyn Op>, inputs: Vec<Variable>, outputs: &[Variable], generation: u32) -> Self {
        let output_shapes = outputs
            .iter()
            .map(|y| y.shape().unwrap_or_default())
            .collect();
        let weak_outputs = outputs.iter().map(Variable::downgrade).collect();
        Self(Rc::new(FunctionInner {
            op,
            inputs,
            outputs: RefCell::new(weak_outputs),
            output_shapes,
            generation,
        }))
    }

    /// Pointer identity — Python's `id()`.
    #[must_use]
    pub fn id(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }

    /// The generation this node sits in: `max(generation of inputs)`.
    ///
    /// Backpropagation processes higher generations first, which guarantees a
    /// node's output gradients are fully accumulated before it runs (step 16).
    #[must_use]
    pub fn generation(&self) -> u32 {
        self.0.generation
    }

    /// Returns handles to the input variables (strong references).
    #[must_use]
    pub fn inputs(&self) -> Vec<Variable> {
        self.0.inputs.clone()
    }

    /// Returns the number of outputs this invocation produced.
    #[must_use]
    pub fn output_count(&self) -> usize {
        self.0.output_shapes.len()
    }

    /// The operation's type name — Python's `f.__class__.__name__`.
    ///
    /// Every [`Op`] is [`Debug`](std::fmt::Debug), and a derived `Debug` opens
    /// with the type's own name, so the leading identifier is exactly the class
    /// name Python would print. That avoids widening the `Op` trait with a
    /// `name` method every implementor would have to write out by hand.
    ///
    /// Used to label nodes in the DOT graph of
    /// [`utils::dot`](crate::utils::dot).
    #[must_use]
    pub fn op_name(&self) -> String {
        let rendered = format!("{:?}", self.0.op);
        rendered
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .next()
            .unwrap_or_default()
            .to_owned()
    }

    /// Returns handles to the output variables, or `None` for any output the
    /// user has already dropped.
    ///
    /// The `None` case is exactly what the weak edge buys: a forward graph
    /// dies as soon as nothing outside it holds its outputs.
    #[must_use]
    pub fn outputs(&self) -> Vec<Option<Variable>> {
        self.0
            .outputs
            .borrow()
            .iter()
            .map(|weak| weak.upgrade().map(Variable::from_inner))
            .collect()
    }

    /// The recorded shape of output `index`, used to synthesise a zero
    /// gradient for a dropped output.
    pub(crate) fn output_shape(&self, index: usize) -> Option<&[usize]> {
        self.0.output_shapes.get(index).map(Vec::as_slice)
    }

    /// Runs the op's backward, supplying this invocation's inputs.
    pub(crate) fn backward(&self, outputs: &[Variable], gys: &[Variable]) -> Vec<Variable> {
        self.0.op.backward(&self.0.inputs, outputs, gys)
    }
}

impl PartialEq for Function {
    /// Pointer identity, matching Python's use of `Function` objects in a
    /// `seen_set`.
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Function {}

impl Hash for Function {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}

impl fmt::Debug for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Function")
            .field("op", &self.0.op)
            .field("generation", &self.0.generation)
            .field("inputs", &self.0.inputs.len())
            .field("outputs", &self.0.output_shapes.len())
            .finish()
    }
}

/// Runs `op` on `inputs`, returning its outputs — Python's
/// `Function.__call__`.
///
/// When [`enable_backprop`](crate::enable_backprop) is set (the default), the
/// invocation is recorded: a [`Function`] node is created, each output's
/// creator is set to it, and it holds strong references to the inputs and weak
/// references to the outputs. Under [`no_grad`](crate::no_grad) no node is
/// created at all, so nothing but the returned arrays survives the call.
///
/// # Panics
///
/// * If any input has no data (a [`Variable::empty`] that was never filled
///   in). The [`Op`] interface takes plain arrays, so there is no `Result`
///   channel to report this on, and every arithmetic path bottoms out in
///   operator overloads whose signatures cannot carry one either.
/// * If an `Op::forward` implementation mutates one of its own input
///   variables' data while it runs (the inputs are borrowed for the duration
///   of the call).
pub fn apply<O: Op + 'static>(op: O, inputs: &[&Variable]) -> Vec<Variable> {
    let mut op = op;

    let outputs = {
        // The borrows are held only across `forward`, which is why they may be
        // live borrows at all: no user code runs between the borrow and the
        // release except the op itself.
        let borrows: Vec<_> = inputs.iter().map(|v| v.borrow_data()).collect();
        let mut xs: Vec<&ArrayD<f64>> = Vec::with_capacity(borrows.len());
        for (index, borrow) in borrows.iter().enumerate() {
            let Some(array) = borrow.as_ref() else {
                panic!(
                    "dezero: input {index} of {op:?} holds no data; \
                     it was created with `Variable::empty()` and never filled in"
                );
            };
            xs.push(array);
        }
        let ys = op.forward(&xs);
        ys.into_iter().map(Variable::new).collect::<Vec<_>>()
    };

    if config::enable_backprop() {
        let generation = inputs.iter().map(|v| v.generation()).max().unwrap_or(0);
        let func = Function::new(
            Box::new(op),
            inputs.iter().copied().cloned().collect(),
            &outputs,
            generation,
        );
        for output in &outputs {
            output.set_creator(&func);
        }
    }

    outputs
}

/// [`apply`] for the common single-output case.
///
/// # Panics
///
/// Panics for the reasons [`apply`] does, and additionally if `op` produced a
/// number of outputs other than one.
pub fn apply1<O: Op + 'static>(op: O, inputs: &[&Variable]) -> Variable {
    let outputs = apply(op, inputs);
    let count = outputs.len();
    let Ok([output]) = <[Variable; 1]>::try_from(outputs) else {
        panic!("dezero: apply1 requires an op with exactly one output, but it produced {count}");
    };
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{no_grad, square};
    use std::collections::HashSet;

    #[test]
    fn apply_records_creator_inputs_and_generation() {
        let x = Variable::from_scalar(2.0);
        let y = square(&x);

        let func = y.creator().expect("square records its creator");
        assert_eq!(func.generation(), 0);
        assert_eq!(y.generation(), 1);
        assert_eq!(func.inputs().len(), 1);
        assert_eq!(func.inputs()[0], x);
        assert_eq!(func.outputs(), vec![Some(y.clone())]);
    }

    #[test]
    fn generation_is_the_max_over_inputs() {
        let x = Variable::from_scalar(2.0);
        let a = square(&x); // generation 1
        let b = square(&a); // generation 2
        let y = crate::add(&x, &b);

        let func = y.creator().expect("add records its creator");
        assert_eq!(func.generation(), 2);
        assert_eq!(y.generation(), 3);
    }

    #[test]
    fn no_grad_skips_graph_construction() {
        let x = Variable::from_scalar(2.0);
        let y = {
            let _guard = no_grad();
            square(&x)
        };
        assert!(y.creator().is_none());
        assert_eq!(y.generation(), 0);
        assert_eq!(y.data(), Variable::from_scalar(4.0).data());
    }

    #[test]
    fn dropping_an_output_breaks_the_cycle() {
        let x = Variable::from_scalar(2.0);
        let func = {
            let y = square(&x);
            let func = y.creator().expect("square records its creator");
            assert!(func.outputs()[0].is_some(), "output is alive while held");
            func
        };
        // `y` is gone and only the weak link remains, so the node is freed
        // even though the function that produced it is still reachable.
        assert!(
            func.outputs()[0].is_none(),
            "dropping the output must free it: the function holds only a Weak"
        );
        // The input, held strongly, is still there.
        assert_eq!(func.inputs()[0], x);
    }

    #[test]
    #[allow(
        clippy::mutable_key_type,
        reason = "that Function is a usable HashSet key despite its RefCell is \
                  exactly what this test pins down: Hash and Eq only ever look \
                  at the Rc pointer"
    )]
    fn function_identity_is_pointer_based() {
        let x = Variable::from_scalar(2.0);
        let a = square(&x);
        let b = square(&x);
        let fa = a.creator().expect("creator");
        let fb = b.creator().expect("creator");

        assert_eq!(fa, fa.clone());
        assert_ne!(fa, fb);

        let mut seen: HashSet<Function> = HashSet::new();
        assert!(seen.insert(fa.clone()));
        assert!(!seen.insert(fa));
        assert!(seen.insert(fb));
        assert_eq!(seen.len(), 2);
    }

    #[test]
    #[should_panic(expected = "holds no data")]
    fn applying_to_an_empty_variable_panics_with_context() {
        let x = Variable::empty();
        let _ = square(&x);
    }
}
