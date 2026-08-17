//! Graphviz DOT rendering of a computation graph (steps 25–26).
//!
//! Port of `_dot_var`, `_dot_func` and `get_dot_graph` in
//! `vendor/dezero-python/dezero/utils.py`.
//!
//! # What is and is not ported
//!
//! Python's `plot_dot_graph` writes the DOT text to `~/.dezero/tmp_graph.dot`,
//! shells out to the Graphviz `dot` binary, and returns a Jupyter `Image`.
//! Only the **text generation** is ported. Rendering needs an external binary
//! that may not be installed, produces a file nobody asserts on, and would put
//! a process spawn in a library — so `get_dot_graph` returns a `String` and the
//! caller decides what to do with it:
//!
//! ```sh
//! # write the string to graph.dot, then:
//! dot -Tpng graph.dot -o graph.png
//! ```
//!
//! Recorded as a divergence in `docs/DIVERGENCES.md`.
//!
//! # Node identity
//!
//! Python labels nodes with `id(v)`, the object address. The port uses
//! [`Variable::id`] and [`Function::id`], which are `Rc::as_ptr` — the same
//! idea, and already the crate's notion of identity everywhere else.

use std::collections::HashSet;
use std::fmt::Write as _;

use crate::core::function::Function;
use crate::core::variable::Variable;

/// Renders one variable as a DOT node.
///
/// `verbose` appends the shape, matching Python — but not the dtype, which is
/// always `f64` here (divergence 1) and would be noise on every node.
fn dot_var(v: &Variable, verbose: bool) -> String {
    let mut label = v.name().unwrap_or_default();
    if verbose && v.data().is_some() {
        if !label.is_empty() {
            label.push_str(": ");
        }
        if let Some(shape) = v.shape() {
            let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
            let _ = write!(label, "({})", dims.join(", "));
        }
    }
    format!(
        "{} [label=\"{}\", color=orange, style=filled]\n",
        v.id(),
        label
    )
}

/// Renders one function as a DOT node plus its in- and out-edges.
///
/// An output whose `Weak` has expired is skipped: the edge would point at a
/// node that no longer exists, and drawing a dangling arrow would be worse
/// than omitting it. Python cannot hit this case because it renders the graph
/// while still holding the output.
fn dot_func(f: &Function) -> String {
    let mut out = format!(
        "{} [label=\"{}\", color=lightblue, style=filled, shape=box]\n",
        f.id(),
        f.op_name()
    );
    for x in f.inputs() {
        let _ = writeln!(out, "{} -> {}", x.id(), f.id());
    }
    for y in f.outputs().into_iter().flatten() {
        let _ = writeln!(out, "{} -> {}", f.id(), y.id());
    }
    out
}

/// The Graphviz DOT source for the graph backward-reachable from `output` —
/// Python's `get_dot_graph`.
///
/// `verbose` adds each variable's shape to its label.
///
/// # Examples
///
/// ```
/// use dezero::{get_dot_graph, Variable};
///
/// let x = Variable::from_scalar(2.0);
/// x.set_name(Some("x".to_owned()));
/// let y = &x * &x;
/// y.set_name(Some("y".to_owned()));
///
/// let dot = get_dot_graph(&y, false);
/// assert!(dot.starts_with("digraph g {\n"));
/// assert!(dot.ends_with("}"));
/// assert!(dot.contains("label=\"y\""));
/// assert!(dot.contains("label=\"Mul\""), "functions are labelled by op name");
/// assert!(dot.contains("->"), "and connected by edges");
/// ```
#[must_use]
#[allow(
    clippy::mutable_key_type,
    reason = "Function hashes and compares by Rc pointer identity (Python's \
              id()); the RefCell it contains is never observed by Hash or Eq"
)]
pub fn get_dot_graph(output: &Variable, verbose: bool) -> String {
    let mut txt = dot_var(output, verbose);
    let mut seen: HashSet<Function> = HashSet::new();
    let mut stack: Vec<Function> = Vec::new();

    if let Some(creator) = output.creator() {
        seen.insert(creator.clone());
        stack.push(creator);
    }

    while let Some(func) = stack.pop() {
        txt.push_str(&dot_func(&func));
        for x in func.inputs() {
            txt.push_str(&dot_var(&x, verbose));
            if let Some(creator) = x.creator()
                && seen.insert(creator.clone())
            {
                stack.push(creator);
            }
        }
    }

    format!("digraph g {{\n{txt}}}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functions::basic_math::exp;

    fn named(value: f64, name: &str) -> Variable {
        let v = Variable::from_scalar(value);
        v.set_name(Some(name.to_owned()));
        v
    }

    #[test]
    fn a_leaf_renders_as_a_single_node_with_no_edges() {
        let x = named(1.0, "x");
        let dot = get_dot_graph(&x, false);

        assert!(dot.contains("label=\"x\""));
        assert!(!dot.contains("->"), "a leaf has no creator, so no edges");
        assert_eq!(dot.matches("color=orange").count(), 1);
    }

    #[test]
    fn every_node_and_edge_of_a_chain_appears() {
        let x = named(1.0, "x");
        let y = exp(&x);
        y.set_name(Some("y".to_owned()));

        let dot = get_dot_graph(&y, false);
        assert!(dot.contains("label=\"Exp\""));
        // x -> Exp -> y
        assert_eq!(dot.matches("->").count(), 2);
        assert_eq!(
            dot.matches("color=orange").count(),
            2,
            "two variables: x and y"
        );
        assert_eq!(dot.matches("shape=box").count(), 1, "one function");
    }

    /// A variable used twice must still be drawn once, and the function that
    /// consumed it must show both edges.
    #[test]
    fn a_reused_variable_is_one_node_with_two_edges() {
        let x = named(3.0, "x");
        let y = &x * &x;

        let dot = get_dot_graph(&y, false);
        let from_x = dot.matches(&format!("{} -> ", x.id())).count();
        assert_eq!(from_x, 2, "both operands of the multiply come from x");
    }

    /// The traversal must not revisit a function reachable by two paths, or a
    /// diamond would render duplicate nodes.
    #[test]
    fn a_diamond_renders_each_function_once() {
        let x = named(2.0, "x");
        let a = exp(&x);
        let y = &a + &a;

        let dot = get_dot_graph(&y, false);
        assert_eq!(dot.matches("label=\"Exp\"").count(), 1);
        assert_eq!(dot.matches("label=\"Add\"").count(), 1);
    }

    #[test]
    fn verbose_adds_shapes_and_plain_does_not() {
        let x = Variable::new(ndarray::Array2::zeros((2, 3)).into_dyn());
        x.set_name(Some("x".to_owned()));
        let y = exp(&x);

        assert!(get_dot_graph(&y, true).contains("(2, 3)"));
        assert!(!get_dot_graph(&y, false).contains("(2, 3)"));
    }

    #[test]
    fn an_unnamed_variable_gets_an_empty_label_not_a_missing_one() {
        let x = Variable::from_scalar(1.0);
        let dot = get_dot_graph(&exp(&x), false);
        assert!(dot.contains("label=\"\""));
    }

    #[test]
    fn the_output_is_a_well_formed_digraph() {
        let x = named(1.0, "x");
        let dot = get_dot_graph(&exp(&x), true);
        assert!(dot.starts_with("digraph g {\n"));
        assert!(dot.ends_with("}"));
    }
}
