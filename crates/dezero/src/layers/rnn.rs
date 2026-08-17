//! Recurrent layers: [`Rnn`] (step 59) and [`Lstm`] (step 60).
//!
//! Port of `RNN` and `LSTM` in `vendor/dezero-python/dezero/layers.py`.
//!
//! # State is the whole point
//!
//! Unlike every layer before them, these carry a hidden state **across calls**.
//! `forward` sees `&self`, so the state lives in a `RefCell`. That is the same
//! interior-mutability choice [`Linear`](crate::Linear) makes for its lazily
//! shaped weight, and for the same reason: pushing `&mut self` out to the
//! caller would put a mutable borrow through the whole training loop.
//!
//! # Why [`reset_state`](Rnn::reset_state) matters
//!
//! Each call links the new state to the previous one, so an unbroken sequence
//! of `n` steps builds a graph `n` layers deep and `backward` walks all of it —
//! this is BPTT. Nothing frees that graph while the state still references it,
//! so a training loop that never resets grows without bound.
//!
//! Two ways to stop it, both from the book:
//!
//! * [`reset_state`](Rnn::reset_state) — forget the state entirely, at a
//!   sequence boundary;
//! * [`Variable::unchain_backward`](crate::Variable::unchain_backward) — keep
//!   the state's *value* but cut its history, which is truncated BPTT.

use std::cell::RefCell;

use crate::core::parameter::Parameter;
use crate::core::variable::Variable;
use crate::functions::activation::sigmoid;
use crate::functions::basic_math::tanh;
use crate::layers::{Layer, Linear};

/// An Elman RNN with `tanh` — Python's `L.RNN`.
///
/// `h' = tanh(x·Wx + b + h·Wh)`, with the `h·Wh` term dropped on the first
/// step, when there is no previous state. The recurrent transform has no bias
/// of its own: one bias for the pair is enough, and Python spells that
/// `nobias=True`.
///
/// # Examples
///
/// ```
/// use dezero::{Layer, Rnn, Variable};
/// use ndarray::arr2;
///
/// let rnn = Rnn::with_in_size(3, 4); // 3 features in, 4 hidden
/// let x = Variable::new(arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn());
///
/// let h1 = rnn.forward(&x);
/// assert_eq!(h1.shape(), Some(vec![2, 4]));
///
/// // The second step sees the first one's state, so it differs.
/// let h2 = rnn.forward(&x);
/// assert_ne!(h1.data(), h2.data());
///
/// // ...unless the state is cleared, which makes it the first step again.
/// rnn.reset_state();
/// assert_eq!(rnn.forward(&x).data(), h1.data());
/// ```
#[derive(Debug)]
pub struct Rnn {
    x2h: Linear,
    h2h: Linear,
    h: RefCell<Option<Variable>>,
}

impl Rnn {
    /// A layer with `hidden_size` units whose input width is settled by the
    /// first batch — Python's `L.RNN(hidden_size)`.
    #[must_use]
    pub fn new(hidden_size: usize) -> Self {
        Self {
            x2h: Linear::new(hidden_size),
            h2h: Linear::new(hidden_size).without_bias(),
            h: RefCell::new(None),
        }
    }

    /// A layer whose weights exist immediately — Python's
    /// `L.RNN(hidden_size, in_size=in_size)`.
    #[must_use]
    pub fn with_in_size(in_size: usize, hidden_size: usize) -> Self {
        Self {
            x2h: Linear::with_in_size(in_size, hidden_size),
            // The recurrent weight maps hidden to hidden, so its input width is
            // the hidden size -- not `in_size`.
            h2h: Linear::with_in_size(hidden_size, hidden_size).without_bias(),
            h: RefCell::new(None),
        }
    }

    /// Forgets the hidden state — Python's `reset_state`.
    ///
    /// Call this at a sequence boundary. It also releases the BPTT graph the
    /// state was holding alive.
    pub fn reset_state(&self) {
        *self.h.borrow_mut() = None;
    }

    /// The current hidden state, if any step has run since the last reset.
    #[must_use]
    pub fn state(&self) -> Option<Variable> {
        self.h.borrow().clone()
    }

    /// The input-to-hidden transform, `x2h`.
    #[must_use]
    pub fn x2h(&self) -> &Linear {
        &self.x2h
    }

    /// The hidden-to-hidden transform, `h2h`.
    #[must_use]
    pub fn h2h(&self) -> &Linear {
        &self.h2h
    }
}

impl Layer for Rnn {
    fn own_params(&self) -> Vec<Parameter> {
        Vec::new()
    }

    fn sublayers(&self) -> Vec<&dyn Layer> {
        vec![&self.x2h, &self.h2h]
    }

    /// One timestep. The returned state is also retained for the next call.
    fn forward(&self, x: &Variable) -> Variable {
        // Read the previous state and drop the borrow before running anything:
        // the ops below allocate and could otherwise re-enter this cell.
        let previous = self.h.borrow().clone();
        let h_new = match previous {
            None => tanh(&self.x2h.forward(x)),
            Some(h) => tanh(&(&self.x2h.forward(x) + &self.h2h.forward(&h))),
        };
        *self.h.borrow_mut() = Some(h_new.clone());
        h_new
    }
}

/// An LSTM cell — Python's `L.LSTM`.
///
/// Four gates, each an affine map of the input plus (after the first step) one
/// of the previous hidden state:
///
/// ```text
/// f = sigmoid(x·Wxf + bf + h·Whf)     forget
/// i = sigmoid(x·Wxi + bi + h·Whi)     input
/// o = sigmoid(x·Wxo + bo + h·Who)     output
/// u = tanh   (x·Wxu + bu + h·Whu)     candidate
///
/// c' = f * c + i * u                  (just i * u on the first step)
/// h' = o * tanh(c')
/// ```
///
/// As with [`Rnn`], only the `x`-side transforms carry a bias.
///
/// # Examples
///
/// ```
/// use dezero::{Layer, Lstm, Variable};
/// use ndarray::arr2;
///
/// let lstm = Lstm::with_in_size(3, 4);
/// let x = Variable::new(arr2(&[[1.0, 2.0, 3.0]]).into_dyn());
///
/// let h = lstm.forward(&x);
/// assert_eq!(h.shape(), Some(vec![1, 4]));
/// assert!(lstm.cell_state().is_some(), "the cell state is carried too");
///
/// lstm.reset_state();
/// assert!(lstm.cell_state().is_none());
/// ```
#[derive(Debug)]
pub struct Lstm {
    x2f: Linear,
    x2i: Linear,
    x2o: Linear,
    x2u: Linear,
    h2f: Linear,
    h2i: Linear,
    h2o: Linear,
    h2u: Linear,
    h: RefCell<Option<Variable>>,
    c: RefCell<Option<Variable>>,
}

impl Lstm {
    /// A cell with `hidden_size` units whose input width is settled by the
    /// first batch — Python's `L.LSTM(hidden_size)`.
    #[must_use]
    pub fn new(hidden_size: usize) -> Self {
        Self {
            x2f: Linear::new(hidden_size),
            x2i: Linear::new(hidden_size),
            x2o: Linear::new(hidden_size),
            x2u: Linear::new(hidden_size),
            h2f: Linear::new(hidden_size).without_bias(),
            h2i: Linear::new(hidden_size).without_bias(),
            h2o: Linear::new(hidden_size).without_bias(),
            h2u: Linear::new(hidden_size).without_bias(),
            h: RefCell::new(None),
            c: RefCell::new(None),
        }
    }

    /// A cell whose weights exist immediately — Python's
    /// `L.LSTM(hidden_size, in_size=in_size)`.
    #[must_use]
    pub fn with_in_size(in_size: usize, hidden_size: usize) -> Self {
        Self {
            x2f: Linear::with_in_size(in_size, hidden_size),
            x2i: Linear::with_in_size(in_size, hidden_size),
            x2o: Linear::with_in_size(in_size, hidden_size),
            x2u: Linear::with_in_size(in_size, hidden_size),
            h2f: Linear::with_in_size(hidden_size, hidden_size).without_bias(),
            h2i: Linear::with_in_size(hidden_size, hidden_size).without_bias(),
            h2o: Linear::with_in_size(hidden_size, hidden_size).without_bias(),
            h2u: Linear::with_in_size(hidden_size, hidden_size).without_bias(),
            h: RefCell::new(None),
            c: RefCell::new(None),
        }
    }

    /// Forgets **both** states — Python's `reset_state`.
    pub fn reset_state(&self) {
        *self.h.borrow_mut() = None;
        *self.c.borrow_mut() = None;
    }

    /// The current hidden state `h`, if any step has run since the last reset.
    #[must_use]
    pub fn state(&self) -> Option<Variable> {
        self.h.borrow().clone()
    }

    /// The current cell state `c`, if any step has run since the last reset.
    #[must_use]
    pub fn cell_state(&self) -> Option<Variable> {
        self.c.borrow().clone()
    }

    /// The four input-side gate transforms, in Python's field order:
    /// forget, input, output, candidate.
    #[must_use]
    pub fn input_gates(&self) -> [&Linear; 4] {
        [&self.x2f, &self.x2i, &self.x2o, &self.x2u]
    }

    /// The four recurrent gate transforms, in the same order.
    #[must_use]
    pub fn recurrent_gates(&self) -> [&Linear; 4] {
        [&self.h2f, &self.h2i, &self.h2o, &self.h2u]
    }
}

impl Layer for Lstm {
    fn own_params(&self) -> Vec<Parameter> {
        Vec::new()
    }

    fn sublayers(&self) -> Vec<&dyn Layer> {
        vec![
            &self.x2f, &self.x2i, &self.x2o, &self.x2u, &self.h2f, &self.h2i, &self.h2o, &self.h2u,
        ]
    }

    /// One timestep. Both states are retained for the next call.
    fn forward(&self, x: &Variable) -> Variable {
        let previous_h = self.h.borrow().clone();
        let previous_c = self.c.borrow().clone();

        let (f, i, o, u) = match previous_h {
            None => (
                sigmoid(&self.x2f.forward(x)),
                sigmoid(&self.x2i.forward(x)),
                sigmoid(&self.x2o.forward(x)),
                tanh(&self.x2u.forward(x)),
            ),
            Some(h) => (
                sigmoid(&(&self.x2f.forward(x) + &self.h2f.forward(&h))),
                sigmoid(&(&self.x2i.forward(x) + &self.h2i.forward(&h))),
                sigmoid(&(&self.x2o.forward(x) + &self.h2o.forward(&h))),
                tanh(&(&self.x2u.forward(x) + &self.h2u.forward(&h))),
            ),
        };

        let c_new = match previous_c {
            None => &i * &u,
            Some(c) => &(&f * &c) + &(&i * &u),
        };
        let h_new = &o * &tanh(&c_new);

        *self.h.borrow_mut() = Some(h_new.clone());
        *self.c.borrow_mut() = Some(c_new);
        h_new
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    fn input() -> Variable {
        Variable::new(arr2(&[[0.5, -1.0, 2.0], [1.5, 0.25, -0.5]]).into_dyn())
    }

    // -- Rnn ---------------------------------------------------------------

    #[test]
    fn an_rnn_registers_both_of_its_transforms() {
        let rnn = Rnn::with_in_size(3, 4);
        // x2h contributes W and b; h2h is nobias, so W only.
        assert_eq!(rnn.params().len(), 3);
        assert!(rnn.own_params().is_empty(), "an Rnn owns no direct weights");
    }

    #[test]
    fn the_recurrent_weight_is_hidden_by_hidden() {
        let rnn = Rnn::with_in_size(3, 4);
        assert_eq!(rnn.x2h().weight().shape(), Some(vec![3, 4]));
        assert_eq!(
            rnn.h2h().weight().shape(),
            Some(vec![4, 4]),
            "h2h maps hidden to hidden, so in_size is the hidden size"
        );
        assert!(rnn.h2h().bias().is_none(), "the recurrent path has no bias");
    }

    #[test]
    fn state_carries_between_steps_and_clears_on_reset() {
        let rnn = Rnn::with_in_size(3, 4);
        assert!(rnn.state().is_none());

        let first = rnn.forward(&input());
        assert_eq!(rnn.state().map(|h| h.data()), Some(first.data()));

        let second = rnn.forward(&input());
        assert_ne!(
            first.data(),
            second.data(),
            "the second step sees the first one's state"
        );

        rnn.reset_state();
        assert!(rnn.state().is_none());
        assert_eq!(
            rnn.forward(&input()).data(),
            first.data(),
            "after a reset the next step is a first step again"
        );
    }

    #[test]
    fn an_unrolled_rnn_back_propagates_through_every_step() {
        let rnn = Rnn::with_in_size(3, 4);
        let x = input();

        let mut h = rnn.forward(&x);
        for _ in 0..2 {
            h = rnn.forward(&x);
        }
        crate::functions::reduce::sum_all(&h).backward();

        // BPTT reaches the weights through all three steps.
        for p in rnn.params() {
            assert!(
                p.grad().is_some(),
                "every parameter takes a gradient from the unrolled graph"
            );
        }
    }

    #[test]
    fn unchaining_the_state_truncates_the_history() {
        let rnn = Rnn::with_in_size(3, 4);
        let x = input();

        rnn.forward(&x);
        let h = rnn.forward(&x);
        assert!(h.creator().is_some(), "the state has a history to cut");

        h.unchain_backward();

        // `unchain_backward` cuts the *ancestors*, not the node it is called
        // on -- so `h` keeps the step that produced it and loses everything
        // before that. This matches the reference, whose loop unchains
        // `f.inputs`, never `self`.
        let creator = h.creator().expect("the producing step survives the cut");
        assert!(
            creator
                .inputs()
                .iter()
                .all(|input| input.creator().is_none()),
            "everything upstream of the last step is severed"
        );

        // The value survives the cut -- that is the point of truncated BPTT.
        assert!(h.data().is_some());
    }

    /// The two transforms settle at *different* times, because the first step
    /// never touches `h2h` — there is no previous state to feed it. Verified
    /// against the reference, which does the same.
    #[test]
    fn a_lazily_shaped_rnn_settles_one_transform_per_step() {
        let rnn = Rnn::new(4);
        assert!(rnn.x2h().weight().data().is_none());

        rnn.forward(&input());
        assert_eq!(rnn.x2h().weight().shape(), Some(vec![3, 4]));
        assert!(
            rnn.h2h().weight().data().is_none(),
            "the first step has no state to run through h2h, so it stays unshaped"
        );

        rnn.forward(&input());
        assert_eq!(
            rnn.h2h().weight().shape(),
            Some(vec![4, 4]),
            "the second step is the first one to use the recurrent path"
        );
    }

    // -- Lstm --------------------------------------------------------------

    #[test]
    fn an_lstm_registers_all_eight_transforms() {
        let lstm = Lstm::with_in_size(3, 4);
        // Four x-side gates with W and b, four recurrent gates with W only.
        assert_eq!(lstm.params().len(), 4 * 2 + 4);
        assert_eq!(lstm.sublayers().len(), 8);
    }

    #[test]
    fn only_the_input_side_gates_carry_a_bias() {
        let lstm = Lstm::with_in_size(3, 4);
        assert!(lstm.input_gates().iter().all(|g| g.bias().is_some()));
        assert!(lstm.recurrent_gates().iter().all(|g| g.bias().is_none()));
    }

    #[test]
    fn both_lstm_states_carry_and_clear_together() {
        let lstm = Lstm::with_in_size(3, 4);
        assert!(lstm.state().is_none() && lstm.cell_state().is_none());

        let first = lstm.forward(&input());
        assert!(lstm.state().is_some() && lstm.cell_state().is_some());

        let second = lstm.forward(&input());
        assert_ne!(first.data(), second.data());

        lstm.reset_state();
        assert!(lstm.state().is_none() && lstm.cell_state().is_none());
        assert_eq!(
            lstm.forward(&input()).data(),
            first.data(),
            "clearing both states makes the next step a first step"
        );
    }

    #[test]
    fn an_unrolled_lstm_back_propagates_through_every_step() {
        let lstm = Lstm::with_in_size(3, 4);
        let x = input();

        let mut h = lstm.forward(&x);
        for _ in 0..2 {
            h = lstm.forward(&x);
        }
        crate::functions::reduce::sum_all(&h).backward();

        for p in lstm.params() {
            assert!(p.grad().is_some(), "BPTT reaches every gate");
        }
    }

    /// The output gate squashes through `tanh`, so `h` is bounded even when the
    /// cell state is not.
    #[test]
    fn the_hidden_state_stays_within_the_unit_interval() {
        let lstm = Lstm::with_in_size(3, 4);
        let big = Variable::new(arr2(&[[50.0, -50.0, 50.0]]).into_dyn());

        for _ in 0..5 {
            let h = lstm.forward(&big);
            assert!(
                h.data().expect("data").iter().all(|v| v.abs() <= 1.0),
                "h = o * tanh(c), and both factors are bounded by 1"
            );
        }
    }
}
