//! [`DataLoader`]: a [`Dataset`] cut into mini-batches (step 50).
//!
//! Port of `DataLoader` in `vendor/dezero-python/dezero/dataloaders.py`.
//!
//! ```python
//! self.max_iter = math.ceil(self.data_size / batch_size)
//! ...
//! batch_index = self.index[i * batch_size:(i + 1) * batch_size]
//! ```
//!
//! `ceil`, not floor: the last batch of an epoch is **short**, never dropped.
//! `300 / 32` is nine batches of 32 and one of 12, and every example is seen
//! exactly once per epoch. Python's slice clamps at the end of the index array;
//! the port clamps explicitly, which is the same arithmetic said out loud.
//!
//! # Why it is an [`Iterator`] but does not restart itself
//!
//! Python's `__next__` calls `reset()` *and then* raises `StopIteration`, so a
//! `for` loop over the same loader object silently begins a fresh, re-shuffled
//! epoch the next time round. Rust's iterator protocol has no equivalent
//! escape hatch: an iterator that starts yielding again after returning `None`
//! violates the [`FusedIterator`] contract that `std`'s adapters are entitled to
//! rely on, and `.chain()`, `.zip()` or `.by_ref()` would quietly produce
//! nonsense.
//!
//! So the port keeps the epoch boundary explicit: [`next`](Iterator::next)
//! returns `None` for good, and [`reset`](DataLoader::reset) — the same method
//! Python has, doing the same job — begins the next epoch.
//!
//! ```
//! use dezero::{DataLoader, Spiral};
//!
//! let train = Spiral::generate(0);
//! let mut loader = DataLoader::new(&train, 32, true).with_seed(1);
//!
//! for _epoch in 0..3 {
//!     for batch in &mut loader {
//!         assert!(batch.len() <= 32);
//!     }
//!     loader.reset(); // Python does this for you on the way out of the loop
//! }
//! ```

use std::fmt;
use std::iter::FusedIterator;

use ndarray::ArrayD;

use crate::core::variable::Variable;
use crate::data::{Dataset, stack_rows};
use crate::utils::random::Rng;

/// The seed [`DataLoader`]'s shuffle stream starts from when none is given.
///
/// Fixed rather than drawn from the clock: two runs of the same program shuffle
/// the same way unless the caller asks otherwise, which is what makes a
/// training run reproducible. `np.random.permutation` reads a process-global
/// stream instead; see [`with_seed`](DataLoader::with_seed).
const DEFAULT_SHUFFLE_SEED: u64 = 0x5EED_0DA7_A10A_DE12;

/// One mini-batch: stacked inputs and their labels.
///
/// Python returns the pair `(x, t)` from `__next__`; the port names them,
/// because `batch.t` at the call site reads better than `batch.1`.
#[derive(Debug, Clone, PartialEq)]
pub struct Batch {
    /// The stacked inputs, `[batch_size, ..example shape]`.
    pub x: ArrayD<f64>,
    /// One class label per row, in the same order as `x`.
    ///
    /// Empty when the dataset is unlabelled — Python stacks a list of `None`
    /// into an `object`-dtype array there, which nothing can consume.
    pub t: Vec<usize>,
}

impl Batch {
    /// The number of examples in the batch.
    ///
    /// The final batch of an epoch is shorter than `batch_size` whenever the
    /// batch size does not divide the dataset.
    #[must_use]
    pub fn len(&self) -> usize {
        self.x.shape().first().copied().unwrap_or(0)
    }

    /// Whether the batch holds no examples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The inputs as a graph node, ready for a forward pass — Python's
    /// `Variable(x)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use dezero::{DataLoader, Spiral};
    ///
    /// let train = Spiral::generate(0);
    /// let mut loader = DataLoader::new(&train, 16, false);
    /// let batch = loader.next().expect("300 examples make 19 batches");
    /// assert_eq!(batch.input().shape(), Some(vec![16, 2]));
    /// ```
    #[must_use]
    pub fn input(&self) -> Variable {
        Variable::new(self.x.clone())
    }
}

/// Mini-batch iteration over a dataset — Python's `DataLoader`.
///
/// Borrows the dataset rather than owning it, so the same data can back several
/// loaders (a shuffled one for training and an ordered one for evaluation, say)
/// without being copied. Python holds a reference for the same reason; Rust
/// just says so in the type.
///
/// # Examples
///
/// ```
/// use dezero::{DataLoader, Dataset, Spiral};
///
/// let train = Spiral::generate(0);
/// let loader = DataLoader::new(&train, 32, false);
///
/// assert_eq!(loader.data_size(), 300);
/// assert_eq!(loader.max_iter(), 10, "ceil(300 / 32)");
///
/// let sizes: Vec<usize> = loader.map(|batch| batch.len()).collect();
/// assert_eq!(sizes, vec![32, 32, 32, 32, 32, 32, 32, 32, 32, 12]);
/// ```
#[derive(Clone)]
pub struct DataLoader<'a> {
    dataset: &'a dyn Dataset,
    batch_size: usize,
    shuffle: bool,
    data_size: usize,
    max_iter: usize,
    iteration: usize,
    /// The order examples are visited in — Python's `self.index`, either
    /// `arange` or a permutation.
    index: Vec<usize>,
    rng: Rng,
}

impl<'a> DataLoader<'a> {
    /// Wraps a dataset — Python's `DataLoader(dataset, batch_size, shuffle)`.
    ///
    /// `shuffle` has no default, unlike Python's `shuffle=True`: whether the
    /// examples of an epoch are permuted is the difference between a training
    /// loader and an evaluation one, and it is not something to leave implicit.
    ///
    /// # Panics
    ///
    /// Panics if `batch_size` is 0 — `ceil(n / 0)` has no answer, and Python
    /// raises `ZeroDivisionError` at the same point.
    #[must_use]
    pub fn new(dataset: &'a dyn Dataset, batch_size: usize, shuffle: bool) -> Self {
        assert!(
            batch_size > 0,
            "dezero: a DataLoader needs a batch size of at least 1"
        );

        let data_size = dataset.len();
        let mut loader = Self {
            dataset,
            batch_size,
            shuffle,
            data_size,
            max_iter: data_size.div_ceil(batch_size),
            iteration: 0,
            index: Vec::new(),
            rng: Rng::new(DEFAULT_SHUFFLE_SEED),
        };
        loader.reset();
        loader
    }

    /// Pins the shuffle stream, and re-draws the current epoch's order from it.
    ///
    /// Python has no counterpart: `reset()` reads `np.random`, so pinning the
    /// order there means seeding the whole process. A loader-local stream keeps
    /// one reproducible training run from dictating everything else's
    /// randomness — the same reasoning that makes [`crate::seed`]
    /// `thread_local!`.
    ///
    /// # Examples
    ///
    /// ```
    /// use dezero::{DataLoader, Spiral};
    ///
    /// let train = Spiral::generate(0);
    /// let first: Vec<usize> = DataLoader::new(&train, 300, true).with_seed(42).order().to_vec();
    /// let again: Vec<usize> = DataLoader::new(&train, 300, true).with_seed(42).order().to_vec();
    /// assert_eq!(first, again);
    /// ```
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Rng::new(seed);
        self.reset();
        self
    }

    /// Starts a fresh epoch — Python's `DataLoader.reset`.
    ///
    /// Rewinds to the first batch and, when shuffling, draws a new permutation.
    pub fn reset(&mut self) {
        self.iteration = 0;
        self.index = if self.shuffle {
            self.rng.permutation(self.data_size)
        } else {
            (0..self.data_size).collect()
        };
    }

    /// The number of examples in the underlying dataset — Python's
    /// `data_size`.
    #[must_use]
    pub fn data_size(&self) -> usize {
        self.data_size
    }

    /// The number of batches in one epoch — Python's
    /// `max_iter = ceil(data_size / batch_size)`.
    #[must_use]
    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    /// The requested batch size. Only the last batch of an epoch may be
    /// smaller.
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Whether each epoch permutes the examples.
    #[must_use]
    pub fn shuffle(&self) -> bool {
        self.shuffle
    }

    /// How many batches this epoch has already yielded — Python's `iteration`.
    #[must_use]
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// The order this epoch visits examples in — Python's `self.index`.
    ///
    /// `0, 1, 2, ...` without shuffling, a permutation of the same with it.
    #[must_use]
    pub fn order(&self) -> &[usize] {
        &self.index
    }

    /// Assembles the batch at `iteration`, without advancing.
    ///
    /// # Panics
    ///
    /// Panics if the dataset reports a label for some examples of the batch and
    /// not others: the batch would silently lose the alignment between `x` and
    /// `t`, which is a wrong training signal rather than an error.
    fn batch_at(&self, iteration: usize) -> Batch {
        let start = iteration * self.batch_size;
        // Python relies on a slice past the end clamping; say it explicitly.
        let end = (start + self.batch_size).min(self.data_size);
        let batch_index = &self.index[start..end];

        let inputs: Vec<ArrayD<f64>> = batch_index.iter().map(|&i| self.dataset.input(i)).collect();

        let mut labels = Vec::with_capacity(batch_index.len());
        let mut unlabelled = 0_usize;
        for &i in batch_index {
            match self.dataset.label(i) {
                Some(label) => labels.push(label),
                None => unlabelled += 1,
            }
        }
        assert!(
            labels.is_empty() || unlabelled == 0,
            "dezero: a dataset must label either all of its examples or none of them, \
             but {} of the {} in this batch had no label",
            unlabelled,
            batch_index.len()
        );

        Batch {
            x: stack_rows(&inputs),
            t: labels,
        }
    }
}

impl fmt::Debug for DataLoader<'_> {
    /// Written by hand because a [`Dataset`] is not required to be [`Debug`] —
    /// requiring it would tax every implementor to make one struct printable.
    /// What a reader wants from a loader anyway is where it is in the epoch,
    /// not a dump of the data.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DataLoader")
            .field("data_size", &self.data_size)
            .field("batch_size", &self.batch_size)
            .field("shuffle", &self.shuffle)
            .field("iteration", &self.iteration)
            .field("max_iter", &self.max_iter)
            .finish_non_exhaustive()
    }
}

impl Iterator for DataLoader<'_> {
    type Item = Batch;

    /// The next batch of this epoch, or `None` once it is exhausted.
    ///
    /// Unlike Python's `__next__`, this does **not** call
    /// [`reset`](DataLoader::reset) on the way out — see the module docs.
    fn next(&mut self) -> Option<Batch> {
        if self.iteration >= self.max_iter {
            return None;
        }
        let batch = self.batch_at(self.iteration);
        self.iteration += 1;
        Some(batch)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.max_iter.saturating_sub(self.iteration);
        (remaining, Some(remaining))
    }
}

/// The batch count of an epoch is known exactly, from the moment the loader is
/// built.
impl ExactSizeIterator for DataLoader<'_> {}

/// Exhaustion is final until [`reset`](DataLoader::reset) — which is precisely
/// what this marker promises and what Python's self-restarting `__next__` could
/// not.
impl FusedIterator for DataLoader<'_> {}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn, arr1};

    /// `input(i) == [i]`, so a batch's contents name the examples in it.
    struct Counting {
        count: usize,
        labelled: bool,
    }

    impl Counting {
        fn labelled(count: usize) -> Self {
            Self {
                count,
                labelled: true,
            }
        }
    }

    impl Dataset for Counting {
        fn len(&self) -> usize {
            self.count
        }

        #[allow(
            clippy::cast_precision_loss,
            reason = "test indices are far below 2^53"
        )]
        fn input(&self, index: usize) -> ArrayD<f64> {
            assert!(index < self.count, "index {index} out of range");
            arr1(&[index as f64]).into_dyn()
        }

        fn label(&self, index: usize) -> Option<usize> {
            assert!(index < self.count, "index {index} out of range");
            self.labelled.then_some(index * 10)
        }
    }

    /// Labels only the even-numbered examples — the alignment hazard.
    struct HalfLabelled;

    impl Dataset for HalfLabelled {
        fn len(&self) -> usize {
            4
        }

        fn input(&self, _index: usize) -> ArrayD<f64> {
            ArrayD::zeros(IxDyn(&[1]))
        }

        fn label(&self, index: usize) -> Option<usize> {
            index.is_multiple_of(2).then_some(index)
        }
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the values are the small non-negative indices put in by Counting"
    )]
    fn visited(loader: DataLoader<'_>) -> Vec<Vec<usize>> {
        loader
            .map(|batch| batch.x.iter().map(|v| *v as usize).collect())
            .collect()
    }

    // -- batch boundaries --------------------------------------------------

    #[test]
    fn a_dividing_batch_size_gives_whole_batches() {
        let set = Counting::labelled(6);
        let loader = DataLoader::new(&set, 3, false);
        assert_eq!(loader.max_iter(), 2);
        assert_eq!(visited(loader), vec![vec![0, 1, 2], vec![3, 4, 5]]);
    }

    #[test]
    fn the_last_batch_is_short_rather_than_dropped() {
        let set = Counting::labelled(7);
        let loader = DataLoader::new(&set, 3, false);
        assert_eq!(loader.max_iter(), 3, "ceil(7 / 3)");
        assert_eq!(visited(loader), vec![vec![0, 1, 2], vec![3, 4, 5], vec![6]]);
    }

    #[test]
    fn a_batch_size_larger_than_the_dataset_gives_one_short_batch() {
        let set = Counting::labelled(3);
        let mut loader = DataLoader::new(&set, 100, false);
        assert_eq!(loader.max_iter(), 1);

        let batch = loader.next().expect("one batch");
        assert_eq!(batch.len(), 3);
        assert_eq!(batch.x.shape(), &[3, 1]);
        assert_eq!(batch.t, vec![0, 10, 20]);
        assert!(loader.next().is_none());
    }

    #[test]
    fn a_batch_size_of_one_yields_one_example_at_a_time() {
        let set = Counting::labelled(3);
        let loader = DataLoader::new(&set, 1, false);
        assert_eq!(loader.max_iter(), 3);
        assert_eq!(visited(loader), vec![vec![0], vec![1], vec![2]]);
    }

    #[test]
    fn an_empty_dataset_yields_nothing() {
        let set = Counting::labelled(0);
        let mut loader = DataLoader::new(&set, 8, false);
        assert_eq!(loader.data_size(), 0);
        assert_eq!(loader.max_iter(), 0, "ceil(0 / 8)");
        assert_eq!(loader.len(), 0);
        assert!(loader.next().is_none());
        loader.reset();
        assert!(loader.next().is_none(), "and still nothing after a reset");
    }

    #[test]
    #[should_panic(expected = "batch size of at least 1")]
    fn a_zero_batch_size_is_rejected() {
        let set = Counting::labelled(4);
        let _ = DataLoader::new(&set, 0, false);
    }

    // -- labels ------------------------------------------------------------

    #[test]
    fn labels_travel_with_their_rows() {
        let set = Counting::labelled(5);
        let batches: Vec<Batch> = DataLoader::new(&set, 2, false).collect();
        assert_eq!(batches[0].t, vec![0, 10]);
        assert_eq!(batches[1].t, vec![20, 30]);
        assert_eq!(batches[2].t, vec![40]);
    }

    #[test]
    fn an_unlabelled_dataset_gives_batches_with_no_labels() {
        let set = Counting {
            count: 4,
            labelled: false,
        };
        let batch = DataLoader::new(&set, 4, false)
            .next()
            .expect("one full batch");
        assert_eq!(batch.len(), 4);
        assert!(batch.t.is_empty());
    }

    #[test]
    #[should_panic(expected = "either all of its examples or none")]
    fn a_partly_labelled_dataset_is_rejected() {
        let _ = DataLoader::new(&HalfLabelled, 4, false).next();
    }

    // -- shuffling ---------------------------------------------------------

    #[test]
    fn shuffling_off_visits_examples_in_order() {
        let set = Counting::labelled(10);
        let loader = DataLoader::new(&set, 4, false);
        assert_eq!(loader.order(), (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn shuffling_on_permutes_without_losing_or_repeating_an_example() {
        let set = Counting::labelled(50);
        let loader = DataLoader::new(&set, 7, true).with_seed(3);

        let mut seen: Vec<usize> = visited(loader).into_iter().flatten().collect();
        assert_eq!(seen.len(), 50, "every example appears exactly once");
        seen.sort_unstable();
        assert_eq!(seen, (0..50).collect::<Vec<_>>());
    }

    #[test]
    fn a_reset_reshuffles_but_an_unshuffled_loader_keeps_its_order() {
        let set = Counting::labelled(30);

        let mut shuffled = DataLoader::new(&set, 5, true).with_seed(9);
        let first_epoch = shuffled.order().to_vec();
        shuffled.reset();
        assert_ne!(shuffled.order(), first_epoch, "a new epoch, a new order");

        let mut ordered = DataLoader::new(&set, 5, false);
        let before = ordered.order().to_vec();
        ordered.reset();
        assert_eq!(ordered.order(), before);
    }

    #[test]
    fn the_seed_pins_the_shuffle() {
        let set = Counting::labelled(20);
        let a = DataLoader::new(&set, 4, true).with_seed(77);
        let b = DataLoader::new(&set, 4, true).with_seed(77);
        let c = DataLoader::new(&set, 4, true).with_seed(78);
        assert_eq!(a.order(), b.order());
        assert_ne!(a.order(), c.order());
    }

    // -- iterator protocol -------------------------------------------------

    #[test]
    fn exhaustion_is_final_until_reset() {
        let set = Counting::labelled(4);
        let mut loader = DataLoader::new(&set, 2, false);

        assert_eq!(loader.by_ref().count(), 2);
        assert!(loader.next().is_none());
        assert!(loader.next().is_none(), "fused: still nothing");

        loader.reset();
        assert_eq!(loader.iteration(), 0);
        assert_eq!(loader.count(), 2, "the next epoch is a full one");
    }

    #[test]
    fn the_remaining_batch_count_is_exact() {
        let set = Counting::labelled(10);
        let mut loader = DataLoader::new(&set, 3, false);
        assert_eq!(loader.len(), 4);
        assert_eq!(loader.size_hint(), (4, Some(4)));

        let _ = loader.next();
        assert_eq!(loader.len(), 3);
        assert_eq!(loader.iteration(), 1);

        while loader.next().is_some() {}
        assert_eq!(loader.len(), 0);
    }

    #[test]
    fn several_epochs_cover_the_dataset_each_time() {
        let set = Counting::labelled(9);
        let mut loader = DataLoader::new(&set, 4, true).with_seed(5);

        for epoch in 0..3 {
            let mut seen: Vec<usize> = (&mut loader)
                .flat_map(|batch| batch.t.into_iter().map(|label| label / 10))
                .collect();
            seen.sort_unstable();
            assert_eq!(seen, (0..9).collect::<Vec<_>>(), "epoch {epoch}");
            loader.reset();
        }
    }

    // -- accessors and Batch -----------------------------------------------

    #[test]
    fn the_loader_reports_how_it_was_built() {
        let set = Counting::labelled(11);
        let loader = DataLoader::new(&set, 4, true).with_seed(1);
        assert_eq!(loader.batch_size(), 4);
        assert_eq!(loader.data_size(), 11);
        assert_eq!(loader.max_iter(), 3);
        assert!(loader.shuffle());
    }

    #[test]
    fn a_batch_converts_to_a_graph_node() {
        let set = Counting::labelled(4);
        let batch = DataLoader::new(&set, 4, false).next().expect("a batch");
        assert!(!batch.is_empty());

        let x = batch.input();
        assert_eq!(x.shape(), Some(vec![4, 1]));
        assert_eq!(x.data(), Some(batch.x.clone()));
        assert!(
            x.creator().is_none(),
            "a batch is a leaf, not a computation"
        );
    }

    #[test]
    fn a_loader_can_be_built_over_a_trait_object() {
        let set = Counting::labelled(5);
        let erased: &dyn Dataset = &set;
        let loader = DataLoader::new(erased, 2, false);
        assert_eq!(loader.max_iter(), 3);
    }

    #[test]
    fn two_loaders_can_share_one_dataset() {
        let set = Counting::labelled(8);
        let train = DataLoader::new(&set, 4, true).with_seed(2);
        let evaluate = DataLoader::new(&set, 8, false);
        assert_eq!(train.max_iter(), 2);
        assert_eq!(evaluate.max_iter(), 1);
    }
}
