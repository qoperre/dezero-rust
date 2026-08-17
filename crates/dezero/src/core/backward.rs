//! The backward pass — port of `Variable.backward` in
//! `vendor/dezero-python/dezero/core.py`.
//!
//! Python keeps a list of pending functions and re-sorts it by `generation`
//! after every push; the port uses a [`BinaryHeap`] keyed on the same number,
//! which pops the highest generation first for the same reason and without the
//! repeated sort. Ties between equal generations are broken arbitrarily; the
//! algorithm does not depend on their order.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use ndarray::{ArrayD, IxDyn};

use crate::core::config;
use crate::core::function::Function;
use crate::core::ops;
use crate::core::variable::Variable;

/// A function waiting to be differentiated, ordered by generation.
///
/// `Ord` intentionally ignores identity: the heap only needs the generation,
/// and duplicate pushes are already prevented by the `seen` set.
#[derive(Debug, Clone)]
struct Pending {
    generation: u32,
    func: Function,
}

impl PartialEq for Pending {
    fn eq(&self, other: &Self) -> bool {
        self.generation == other.generation
    }
}

impl Eq for Pending {}

impl Ord for Pending {
    fn cmp(&self, other: &Self) -> Ordering {
        self.generation.cmp(&other.generation)
    }
}

impl PartialOrd for Pending {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Python's nested `add_func`: enqueue a function the first time it is seen.
#[allow(
    clippy::mutable_key_type,
    reason = "Function hashes and compares by Rc pointer identity (Python's \
              id()); the RefCell it contains is never observed by Hash or Eq"
)]
fn enqueue(heap: &mut BinaryHeap<Pending>, seen: &mut HashSet<Function>, func: Function) {
    if seen.insert(func.clone()) {
        heap.push(Pending {
            generation: func.generation(),
            func,
        });
    }
}

/// A detached zero variable shaped like `shape`.
fn zeros(shape: &[usize]) -> Variable {
    Variable::new(ArrayD::zeros(IxDyn(shape)))
}

/// A detached zero variable shaped like `like`.
///
/// # Panics
///
/// Panics if `like` holds no data. Every variable reachable from the backward
/// pass came out of an [`Op::forward`](crate::Op::forward), so this cannot
/// happen; the branch exists to state the invariant rather than to hide it.
fn zeros_like(like: &Variable) -> Variable {
    like.full_like(0.0).unwrap_or_else(|| {
        panic!("dezero: internal invariant broken — a graph node has no data during backward")
    })
}

impl Variable {
    /// Backpropagates from this variable, filling in `grad` on every upstream
    /// leaf.
    ///
    /// Equivalent to `backward_with(false, false)`, i.e. Python's
    /// `y.backward()`: intermediate gradients are discarded and the backward
    /// pass itself is not recorded.
    pub fn backward(&self) {
        self.backward_with(false, false);
    }

    /// Backpropagates with explicit control over gradient retention and
    /// higher-order differentiation.
    ///
    /// * `retain_grad` — keep the gradients of intermediate variables instead
    ///   of dropping them as soon as they have been consumed. Note that *this*
    ///   variable is an output of its own creator, so with `retain_grad =
    ///   false` its `grad` is cleared too by the time the call returns; that
    ///   matches Python (step 18).
    /// * `create_graph` — record the backward computation itself, so the
    ///   resulting gradients can be differentiated again (steps 33–35,
    ///   Newton's method). This works only because every
    ///   [`Op::backward`](crate::Op::backward) is written in terms of
    ///   [`Variable`] arithmetic.
    ///
    /// If `grad` is unset it is seeded with ones, exactly like Python. Calling
    /// this on a variable with no creator is a no-op beyond that seeding
    /// (Python raises `AttributeError` in that case; the port simply has
    /// nothing to do).
    ///
    /// Gradients *accumulate*: calling this twice without
    /// [`cleargrad`](Variable::cleargrad) sums the two passes.
    ///
    /// # Examples
    ///
    /// ```
    /// use dezero::{pow, Variable};
    ///
    /// // y = x^4, differentiated twice: dy/dx = 4x^3, d2y/dx2 = 12x^2.
    /// let x = Variable::from_scalar(2.0);
    /// let y = pow(&x, 4.0);
    /// y.backward_with(false, true);
    ///
    /// let gx = x.grad().expect("first derivative");
    /// assert_eq!(gx.data(), Variable::from_scalar(32.0).data());
    ///
    /// x.cleargrad();
    /// gx.backward();
    /// assert_eq!(x.grad().and_then(|g| g.data()), Variable::from_scalar(48.0).data());
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if this variable holds no data and no gradient has been set, or
    /// if an [`Op::backward`](crate::Op::backward) returns a number of
    /// gradients that does not match its input count.
    #[allow(
        clippy::mutable_key_type,
        reason = "Function hashes and compares by Rc pointer identity (Python's \
                  id()); the RefCell it contains is never observed by Hash or Eq"
    )]
    pub fn backward_with(&self, retain_grad: bool, create_graph: bool) {
        if self.grad().is_none() {
            let ones = self.full_like(1.0).unwrap_or_else(|| {
                panic!("dezero: backward() on a variable that holds no data and no gradient")
            });
            self.set_grad(Some(ones));
        }

        let mut heap: BinaryHeap<Pending> = BinaryHeap::new();
        let mut seen: HashSet<Function> = HashSet::new();
        if let Some(creator) = self.creator() {
            enqueue(&mut heap, &mut seen, creator);
        }

        while let Some(Pending { func, .. }) = heap.pop() {
            let (outputs, gys) = collect_output_grads(&func);

            {
                // Python: `with using_config('enable_backprop', create_graph)`.
                // Both the op's backward *and* the accumulation below run
                // inside it, so with `create_graph` the accumulation is itself
                // differentiable.
                let _guard = config::using_enable_backprop(create_graph);

                let inputs = func.inputs();
                let gxs = func.backward(&outputs, &gys);
                assert!(
                    gxs.len() == inputs.len(),
                    "dezero: {func:?} returned {} gradients for {} inputs",
                    gxs.len(),
                    inputs.len()
                );

                for (x, gx) in inputs.iter().zip(gxs) {
                    // Accumulation goes through Variable addition, not raw
                    // ndarray addition: under `create_graph` the sum has to be
                    // part of the graph too.
                    let accumulated = match x.grad() {
                        None => gx,
                        Some(existing) => ops::add(&existing, &gx),
                    };
                    x.set_grad(Some(accumulated));

                    if let Some(upstream) = x.creator() {
                        enqueue(&mut heap, &mut seen, upstream);
                    }
                }
            }

            if !retain_grad {
                for output in &outputs {
                    output.set_grad(None);
                }
            }
        }
    }
}

/// Gathers `func`'s outputs and their gradients.
///
/// An output the user has dropped is replaced by a detached zero of the
/// recorded shape, and a live output that never received a gradient (possible
/// only for a multi-output op whose other output was the one differentiated)
/// gets a zero gradient. Python dereferences the dead weakref and raises.
fn collect_output_grads(func: &Function) -> (Vec<Variable>, Vec<Variable>) {
    let live = func.outputs();
    let mut outputs = Vec::with_capacity(live.len());
    let mut gys = Vec::with_capacity(live.len());

    for (index, maybe_output) in live.into_iter().enumerate() {
        let output = maybe_output.unwrap_or_else(|| zeros(func.output_shape(index).unwrap_or(&[])));
        let gy = output.grad().unwrap_or_else(|| zeros_like(&output));
        outputs.push(output);
        gys.push(gy);
    }

    (outputs, gys)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{add, exp, mul, no_grad, pow, square};
    use ndarray::{arr1, arr2};

    fn scalar(v: &Variable) -> f64 {
        v.data().expect("data").into_iter().next().expect("element")
    }

    fn grad_of(v: &Variable) -> f64 {
        scalar(&v.grad().expect("gradient"))
    }

    #[test]
    fn seeds_the_starting_gradient_with_ones() {
        let x = Variable::new(arr1(&[1.0, 2.0, 3.0]).into_dyn());
        let y = square(&x);
        y.backward_with(true, false);
        assert_eq!(
            y.grad().and_then(|g| g.data()),
            Some(arr1(&[1.0, 1.0, 1.0]).into_dyn())
        );
        assert_eq!(
            x.grad().and_then(|g| g.data()),
            Some(arr1(&[2.0, 4.0, 6.0]).into_dyn())
        );
    }

    #[test]
    fn chained_functions_apply_the_chain_rule() {
        // y = square(exp(square(x))), the step-8 example.
        let x = Variable::from_scalar(0.5);
        let a = square(&x);
        let b = exp(&a);
        let y = square(&b);
        y.backward();
        assert!((grad_of(&x) - 3.297_442_541_400_256).abs() < 1e-10);
    }

    #[test]
    fn step14_repeated_input_accumulates_gradient() {
        let x = Variable::from_scalar(3.0);
        let y = add(&x, &x);
        y.backward();
        assert_eq!(scalar(&y), 6.0);
        assert_eq!(grad_of(&x), 2.0);
    }

    #[test]
    fn step14_cleargrad_between_passes() {
        let x = Variable::from_scalar(3.0);

        let y = add(&x, &x);
        y.backward();
        assert_eq!(grad_of(&x), 2.0);

        // Without cleargrad the second pass would accumulate onto the first.
        let y = add(&add(&x, &x), &x);
        y.backward();
        assert_eq!(grad_of(&x), 5.0, "gradients accumulate by design");

        x.cleargrad();
        let y = add(&add(&x, &x), &x);
        y.backward();
        assert_eq!(grad_of(&x), 3.0);
    }

    #[test]
    fn step16_generation_order_gives_the_right_gradient() {
        // y = square(a) + square(a) where a = square(x): the diamond that a
        // naive depth-first traversal gets wrong.
        let x = Variable::from_scalar(2.0);
        let a = square(&x);
        let y = add(&square(&a), &square(&a));
        y.backward();

        assert_eq!(scalar(&y), 32.0);
        assert_eq!(grad_of(&x), 64.0);
    }

    #[test]
    fn step16_deeper_diamond() {
        // The join must not run before both of its branches have contributed.
        let x = Variable::from_scalar(2.0);
        let a = square(&x); // gen 1
        let b = square(&a); // gen 2
        let c = square(&b); // gen 3
        let y = add(&c, &a); // gen 4, joins generations 3 and 1
        y.backward();

        // d/dx (x^8 + x^2) = 8x^7 + 2x = 1024 + 4
        assert_eq!(grad_of(&x), 1028.0);
    }

    #[test]
    fn step18_retain_grad_controls_intermediate_gradients() {
        let x = Variable::from_scalar(2.0);
        let a = square(&x);
        let y = square(&a);

        y.backward();
        assert!(y.grad().is_none(), "the seed gradient is dropped as well");
        assert!(a.grad().is_none(), "intermediate gradients are dropped");
        assert_eq!(grad_of(&x), 32.0, "leaf gradients are kept");

        x.cleargrad();
        let a = square(&x);
        let y = square(&a);
        y.backward_with(true, false);
        assert_eq!(grad_of(&y), 1.0);
        assert_eq!(grad_of(&a), 8.0);
        assert_eq!(grad_of(&x), 32.0);
    }

    #[test]
    fn step18_no_grad_disables_the_graph_entirely() {
        let x = Variable::from_scalar(2.0);
        let y = {
            let _guard = no_grad();
            square(&square(&x))
        };
        assert_eq!(scalar(&y), 16.0);
        assert!(y.creator().is_none());

        y.backward(); // no creator: seeds y.grad and stops.
        assert_eq!(grad_of(&y), 1.0);
        assert!(x.grad().is_none());
    }

    #[test]
    fn backward_on_a_leaf_is_a_no_op() {
        let x = Variable::new(arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn());
        x.backward();
        assert_eq!(
            x.grad().and_then(|g| g.data()),
            Some(arr2(&[[1.0, 1.0], [1.0, 1.0]]).into_dyn())
        );
    }

    #[test]
    fn create_graph_enables_second_order_derivatives() {
        // y = x^3 -> dy/dx = 3x^2 = 27, d2y/dx2 = 6x = 18 at x = 3.
        let x = Variable::from_scalar(3.0);
        let y = pow(&x, 3.0);
        y.backward_with(false, true);

        let gx = x.grad().expect("first derivative");
        assert_eq!(scalar(&gx), 27.0);
        assert!(
            gx.creator().is_some(),
            "with create_graph the gradient is itself a graph node"
        );

        x.cleargrad();
        gx.backward();
        assert_eq!(grad_of(&x), 18.0);
    }

    #[test]
    fn without_create_graph_the_gradient_is_detached() {
        let x = Variable::from_scalar(3.0);
        let y = mul(&x, &x);
        y.backward();
        let gx = x.grad().expect("first derivative");
        assert!(gx.creator().is_none());
    }

    #[test]
    fn backward_frees_the_forward_graph_when_the_output_is_dropped() {
        let x = Variable::from_scalar(2.0);
        let creator = {
            let y = square(&x);
            let inner = y.creator().expect("creator");
            y.backward();
            inner
        };
        assert!(creator.outputs()[0].is_none());
        assert_eq!(grad_of(&x), 4.0);
    }
}
