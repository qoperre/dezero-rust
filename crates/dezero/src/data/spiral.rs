//! [`Spiral`]: the three-armed toy classification set of step 48.
//!
//! Port of `get_spiral` / `class Spiral` in
//! `vendor/dezero-python/dezero/datasets.py`.
//!
//! Three hundred points in the plane, a hundred per class, each class an arm of
//! a spiral. It is the smallest dataset that is obviously *not* linearly
//! separable, which is the whole point of step 48: a single [`Linear`] cannot
//! fit it and an [`Mlp`](crate::Mlp) with one hidden layer can.
//!
//! [`Linear`]: crate::Linear
//!
//! # Why the data has to be handed in
//!
//! Python builds it from `np.random.randn` and shuffles with
//! `np.random.permutation`:
//!
//! ```python
//! seed = 1984 if train else 2020
//! np.random.seed(seed=seed)
//! ...
//! theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
//! ...
//! indices = np.random.permutation(num_data * num_class)
//! ```
//!
//! Both read the Mersenne Twister, which no generator in this crate reproduces
//! (`docs/ARCHITECTURE.md`). Seeding both sides and hoping is not an option and
//! never was, so:
//!
//! * [`Spiral::new`] takes the arrays explicitly. This is what the parity
//!   fixture `spiral_data` feeds, and it is the only constructor that agrees
//!   with Python's numbers.
//! * [`Spiral::generate`] runs the same *algorithm* on this crate's [`Rng`].
//!   The distribution matches; the values do not, and never will. Use it to
//!   have a spiral to play with, never to reproduce a Python run.

use ndarray::{ArrayD, IxDyn};

use crate::data::Dataset;
use crate::utils::random::Rng;

/// Spiral arms, one per class — Python's `num_class`.
pub const CLASSES: usize = 3;

/// Points per arm — Python's `num_data`.
pub const POINTS_PER_CLASS: usize = 100;

/// Coordinates per point — Python's `input_dim`.
pub const INPUT_DIM: usize = 2;

/// The angular jitter applied to each point, in radians — Python's `* 0.2`.
const THETA_NOISE: f64 = 0.2;

/// The three-armed spiral — Python's `datasets.Spiral`.
///
/// # Examples
///
/// ```
/// use dezero::{Dataset, Spiral};
/// use ndarray::arr2;
///
/// // Explicit data: what a parity fixture supplies.
/// let tiny = Spiral::new(
///     arr2(&[[0.0, 0.0], [1.0, 0.5], [-1.0, 0.25]]).into_dyn(),
///     vec![0, 1, 2],
/// );
/// assert_eq!(tiny.len(), 3);
/// assert_eq!(tiny.input(1), ndarray::arr1(&[1.0, 0.5]).into_dyn());
/// assert_eq!(tiny.label(2), Some(2));
///
/// // Generated data: the same shape, different numbers from Python's.
/// let train = Spiral::generate(1984);
/// assert_eq!(train.len(), 300);
/// assert_eq!(train.inputs().shape(), &[300, 2]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Spiral {
    x: ArrayD<f64>,
    t: Vec<usize>,
}

impl Spiral {
    /// Builds the dataset from explicit arrays — the constructor a fixture
    /// uses.
    ///
    /// `x` is `[n, 2]` and `t` is the matching `n` class indices.
    ///
    /// # Panics
    ///
    /// Panics if `x` is not two-dimensional, if its row count differs from
    /// `t.len()`, or if any label is not below [`CLASSES`]. A dataset that is
    /// wrong in any of those ways produces a wrong training run rather than an
    /// error, which is much harder to notice.
    #[must_use]
    pub fn new(x: ArrayD<f64>, t: Vec<usize>) -> Self {
        assert!(
            x.ndim() == 2,
            "dezero: Spiral inputs must be a 2-D [n, features] array, got shape {:?}",
            x.shape()
        );
        assert!(
            x.shape()[0] == t.len(),
            "dezero: Spiral has {} input rows but {} labels",
            x.shape()[0],
            t.len()
        );
        if let Some(bad) = t.iter().find(|&&label| label >= CLASSES) {
            panic!("dezero: the spiral has {CLASSES} classes, but a label is {bad}");
        }
        Self { x, t }
    }

    /// Generates a spiral with this crate's [`Rng`] — Python's
    /// `get_spiral()`, algorithm for algorithm.
    ///
    /// **The values differ from Python's.** The arms, the noise scale, the
    /// class balance and the final shuffle are all the reference's; the
    /// underlying stream is not, and no seed makes it so. Every parity test
    /// builds its `Spiral` with [`new`](Spiral::new) instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use dezero::{Dataset, Spiral};
    ///
    /// let a = Spiral::generate(1984);
    /// assert_eq!(a, Spiral::generate(1984), "a seed replays a spiral exactly");
    /// assert_ne!(a, Spiral::generate(2020), "and a different one does not");
    ///
    /// // 100 points on each of the three arms, shuffled together.
    /// let mut counts = [0_usize; 3];
    /// for i in 0..a.len() {
    ///     counts[a.label(i).expect("labelled")] += 1;
    /// }
    /// assert_eq!(counts, [100, 100, 100]);
    /// ```
    #[must_use]
    pub fn generate(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let data_size = CLASSES * POINTS_PER_CLASS;

        let mut flat = vec![0.0; data_size * INPUT_DIM];
        let mut labels = vec![0_usize; data_size];

        #[allow(
            clippy::cast_precision_loss,
            reason = "the loop bounds are 3 and 100, exactly representable"
        )]
        for class in 0..CLASSES {
            for point in 0..POINTS_PER_CLASS {
                let rate = point as f64 / POINTS_PER_CLASS as f64;
                let radius = rate;
                let theta = class as f64 * 4.0 + 4.0 * rate + rng.standard_normal() * THETA_NOISE;

                let index = POINTS_PER_CLASS * class + point;
                flat[index * INPUT_DIM] = radius * theta.sin();
                flat[index * INPUT_DIM + 1] = radius * theta.cos();
                labels[index] = class;
            }
        }

        // Python's `x = x[indices]; t = t[indices]` — the shuffle is part of the
        // dataset, not of the loader, so an unshuffled DataLoader still sees the
        // classes interleaved.
        let order = rng.permutation(data_size);
        let mut shuffled = vec![0.0; flat.len()];
        let mut shuffled_labels = vec![0_usize; data_size];
        for (destination, &source) in order.iter().enumerate() {
            for component in 0..INPUT_DIM {
                shuffled[destination * INPUT_DIM + component] =
                    flat[source * INPUT_DIM + component];
            }
            shuffled_labels[destination] = labels[source];
        }

        Self {
            x: ArrayD::from_shape_vec(IxDyn(&[data_size, INPUT_DIM]), shuffled)
                .expect("the buffer holds exactly data_size * INPUT_DIM values"),
            t: shuffled_labels,
        }
    }

    /// Every input, as one `[n, 2]` array — Python's `self.data`.
    #[must_use]
    pub fn inputs(&self) -> &ArrayD<f64> {
        &self.x
    }

    /// Every label — Python's `self.label`.
    #[must_use]
    pub fn labels(&self) -> &[usize] {
        &self.t
    }
}

impl Dataset for Spiral {
    fn len(&self) -> usize {
        self.t.len()
    }

    /// Row `index` of the input array.
    ///
    /// # Panics
    ///
    /// Panics if `index` is not below [`len`](Dataset::len).
    fn input(&self, index: usize) -> ArrayD<f64> {
        assert!(
            index < self.len(),
            "dezero: Spiral has {} examples, so index {index} is out of range",
            self.len()
        );
        self.x
            .index_axis(ndarray::Axis(0), index)
            .to_owned()
            .into_dyn()
    }

    /// # Panics
    ///
    /// Panics if `index` is not below [`len`](Dataset::len).
    fn label(&self, index: usize) -> Option<usize> {
        Some(*self.t.get(index).unwrap_or_else(|| {
            panic!(
                "dezero: Spiral has {} examples, so index {index} is out of range",
                self.len()
            )
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    fn tiny() -> Spiral {
        Spiral::new(
            arr2(&[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]).into_dyn(),
            vec![0, 1, 2],
        )
    }

    #[test]
    fn explicit_data_is_handed_back_row_by_row() {
        let set = tiny();
        assert_eq!(set.len(), 3);
        assert!(!set.is_empty());
        assert_eq!(set.input(0), arr1(&[0.0, 1.0]).into_dyn());
        assert_eq!(set.input(2), arr1(&[4.0, 5.0]).into_dyn());
        assert_eq!(set.labels(), &[0, 1, 2]);
        assert_eq!(set.inputs().shape(), &[3, 2]);
    }

    #[test]
    fn an_input_row_is_rank_one() {
        assert_eq!(tiny().input(1).ndim(), 1, "an example, not a 1-row matrix");
    }

    #[test]
    #[should_panic(expected = "index 3 is out of range")]
    fn reading_past_the_end_panics() {
        let _ = tiny().input(3);
    }

    #[test]
    #[should_panic(expected = "index 9 is out of range")]
    fn reading_a_label_past_the_end_panics() {
        let _ = tiny().label(9);
    }

    #[test]
    #[should_panic(expected = "must be a 2-D")]
    fn a_one_dimensional_input_array_is_rejected() {
        let _ = Spiral::new(arr1(&[1.0, 2.0]).into_dyn(), vec![0, 1]);
    }

    #[test]
    #[should_panic(expected = "3 input rows but 2 labels")]
    fn a_label_count_mismatch_is_rejected() {
        let _ = Spiral::new(
            arr2(&[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]).into_dyn(),
            vec![0, 1],
        );
    }

    #[test]
    #[should_panic(expected = "3 classes, but a label is 3")]
    fn a_label_outside_the_class_range_is_rejected() {
        let _ = Spiral::new(arr2(&[[0.0, 1.0]]).into_dyn(), vec![3]);
    }

    #[test]
    fn an_empty_spiral_is_allowed() {
        let empty = Spiral::new(ArrayD::zeros(IxDyn(&[0, 2])), Vec::new());
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
    }

    // -- generate ----------------------------------------------------------

    #[test]
    fn generate_produces_the_reference_shape() {
        let set = Spiral::generate(1984);
        assert_eq!(set.len(), 300);
        assert_eq!(set.inputs().shape(), &[300, 2]);
        assert_eq!(set.labels().len(), 300);
    }

    #[test]
    fn generate_balances_the_three_arms() {
        let set = Spiral::generate(11);
        let mut counts = [0_usize; CLASSES];
        for &label in set.labels() {
            counts[label] += 1;
        }
        assert_eq!(counts, [POINTS_PER_CLASS; CLASSES]);
    }

    #[test]
    fn generate_is_reproducible_and_seed_dependent() {
        assert_eq!(Spiral::generate(7), Spiral::generate(7));
        assert_ne!(Spiral::generate(7), Spiral::generate(8));
    }

    #[test]
    fn generate_shuffles_the_classes_together() {
        let set = Spiral::generate(3);
        let first_hundred = &set.labels()[..POINTS_PER_CLASS];
        assert!(
            first_hundred.iter().any(|&label| label != first_hundred[0]),
            "unshuffled, the first hundred labels would all be class 0"
        );
    }

    /// Every point lies inside the unit disc: `radius = i / 100 < 1` and the
    /// noise only rotates it.
    #[test]
    fn generated_points_stay_inside_the_unit_disc() {
        let set = Spiral::generate(2);
        for index in 0..set.len() {
            let point = set.input(index);
            let radius = point.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(radius < 1.0, "point {index} has radius {radius}");
        }
    }

    /// The arms are what makes the problem non-linear: at the same radius, the
    /// three classes sit at angles roughly four radians apart.
    #[test]
    fn the_three_arms_are_separated_in_angle() {
        let set = Spiral::generate(4);
        let mut outermost = [(0.0_f64, 0.0_f64); CLASSES];
        for index in 0..set.len() {
            let point = set.input(index);
            let (x, y) = (point[[0]], point[[1]]);
            let class = set.label(index).expect("labelled");
            if x.hypot(y) > outermost[class].0.hypot(outermost[class].1) {
                outermost[class] = (x, y);
            }
        }

        let angles: Vec<f64> = outermost.iter().map(|(x, y)| x.atan2(*y)).collect();
        for (a, b) in [(0, 1), (1, 2), (0, 2)] {
            let gap = (angles[a] - angles[b]).abs();
            assert!(gap > 0.5, "classes {a} and {b} share an angle: {angles:?}");
        }
    }

    #[test]
    fn a_generated_spiral_is_a_dataset() {
        let set = Spiral::generate(0);
        let erased: &dyn Dataset = &set;
        assert_eq!(erased.len(), 300);
        assert_eq!(erased.input(299).shape(), &[2]);
        assert!(erased.label(299).is_some_and(|label| label < CLASSES));
    }
}
