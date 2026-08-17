# Steps 25–26 — graph visualization

**Status:** done

Step 25 is Graphviz background with no code. Step 26's deliverable is
`get_dot_graph`, in `utils/dot.rs`.

## What is ported, and what is not

Python's `plot_dot_graph` writes the DOT text to `~/.dezero/tmp_graph.dot`,
shells out to the Graphviz `dot` binary, and returns a Jupyter `Image`.

Only the **text generation** is ported. Rendering needs an external binary that
may not be installed, produces a file nobody asserts on, and would put a
process spawn inside a library. `get_dot_graph` returns a `String`; the caller
runs `dot -Tpng graph.dot -o graph.png` if they want a picture. Divergence 30.

## Node identity

Python labels nodes with `id(v)`. The port uses `Variable::id` / `Function::id`,
which are `Rc::as_ptr` — already the crate's notion of identity everywhere else.

Function labels need Python's `f.__class__.__name__`. Rather than widening the
`Op` trait with a `name` method every implementor would have to write out by
hand, `Function::op_name()` takes the leading identifier of the op's `Debug`
output. Every `Op` is `Debug`, and a derived `Debug` opens with the type's own
name, so the two agree by construction.

## Output matches the reference

Same program, both implementations:

```
digraph g {
<id> [label="y", color=orange, style=filled]
<id> [label="Mul", color=lightblue, style=filled, shape=box]
<id> -> <id>
<id> -> <id>
<id> -> <id>
<id> [label="x", color=orange, style=filled]
<id> [label="x", color=orange, style=filled]
}
```

Node order, the duplicated edge from a variable used twice, and the repeated
`x` node are all identical — only the addresses differ. The repetition is
Python's behaviour, not a bug, and Graphviz collapses duplicate node
declarations.

## One case Python cannot reach

An output whose `Weak` has expired is skipped rather than drawn. Python renders
the graph while still holding the output, so it never sees this; here a
dangling arrow to a node that no longer exists would be worse than omitting it.

## Verification

Eight unit tests: a leaf renders with no edges; a chain renders every node and
edge; a variable used twice is **one node with two edges**; a diamond renders
each function **once** (a traversal that revisited would duplicate); verbose
adds shapes and plain does not; an unnamed variable gets an empty label rather
than a missing one; and the output is a well-formed `digraph`.

```
cargo test    637 passed
cargo clippy --all-targets --all-features -- -D warnings   0 errors
cargo fmt --all --check                                    clean
```

`verbose` appends the shape but not the dtype: this port is `f64` throughout
(divergence 1), so a dtype on every node would be noise.
