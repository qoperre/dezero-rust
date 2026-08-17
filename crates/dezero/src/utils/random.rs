//! A small deterministic random number generator, for weight initialisation.
//!
//! Python's `Linear._init_W` calls `np.random.randn`, which draws from numpy's
//! global Mersenne Twister stream. `docs/ARCHITECTURE.md` already records that
//! **no Rust generator will ever reproduce that stream**, so every fixture that
//! depends on random values ships the values explicitly. What is still needed is
//! *a* source of normal deviates, so that a lazily-shaped
//! [`Linear`](crate::Linear) layer can fill in its weights on the first forward
//! pass without breaking symmetry.
//!
//! `ndarray` is this crate's only runtime dependency, so the generator is
//! written here rather than pulled in:
//!
//! * uniforms come from **`SplitMix64`** — Steele, Lea and Flood's
//!   fixed-increment mixer, the seeding routine of the `xoshiro`/`xoroshiro`
//!   family. It passes BigCrush, is four lines long and needs no state beyond a
//!   single `u64`;
//! * normals come from the **Box–Muller transform**.
//!
//! This is a weight initialiser, not a simulation engine: it must be
//! reproducible and free of obvious structure, and it is both. It is not
//! cryptographic, and it does not try to be.
//!
//! # The global stream
//!
//! [`randn`] and [`seed`] mirror `np.random.randn` and `np.random.seed`: a
//! process-wide stream so that constructing two layers in a row gives them
//! different weights without the caller managing a generator. It is
//! `thread_local!` for the same reason [`Config`](crate::no_grad) is — one
//! test's [`seed`] must not perturb another's, and `cargo test` is
//! multi-threaded.
//!
//! ```
//! use dezero::{randn, seed};
//!
//! seed(7);
//! let a = randn(&[2, 3]);
//! seed(7);
//! assert_eq!(randn(&[2, 3]), a, "the same seed replays the same stream");
//! ```

use std::cell::RefCell;

use ndarray::{ArrayD, IxDyn};

/// The seed the global stream starts from, so an unseeded program is still
/// reproducible run to run.
const DEFAULT_SEED: u64 = 0x2545_F491_4F6C_DD1D;

/// `SplitMix64`'s golden-ratio increment.
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// Scales a 53-bit integer into `[0, 1)`; `2^-53`.
const TWO_POW_MINUS_53: f64 = 1.0 / 9_007_199_254_740_992.0;

/// A reproducible stream of pseudo-random numbers.
///
/// # Examples
///
/// ```
/// use dezero::Rng;
///
/// let mut rng = Rng::new(42);
/// let x = rng.randn(&[3, 2]);
/// assert_eq!(x.shape(), &[3, 2]);
///
/// // Same seed, same numbers -- on any platform, in any build.
/// assert_eq!(Rng::new(42).randn(&[3, 2]), x);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Starts a stream from `seed`.
    ///
    /// Every seed is valid, including 0: `SplitMix64` advances by a constant
    /// rather than by a feedback shift register, so it has no zero state to
    /// fall into.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// The next 64 raw bits — one `SplitMix64` step.
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// The next uniform deviate in `[0, 1)`.
    ///
    /// Built from the top 53 bits, which is exactly the mantissa width of an
    /// `f64`: every representable value in the range is reachable and none is
    /// favoured.
    #[allow(
        clippy::cast_precision_loss,
        reason = "the value is masked to 53 bits first, which is precisely the \
                  f64 mantissa width, so the conversion is exact"
    )]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * TWO_POW_MINUS_53
    }

    /// The next standard normal deviate (mean 0, variance 1) — numpy's
    /// `np.random.randn()`.
    ///
    /// Box–Muller: two uniforms in, one normal out. The transform actually
    /// produces a *pair* of independent normals and this keeps only the first;
    /// the second is discarded rather than cached, so a stream's value at step
    /// `n` never depends on whether an earlier call was made from `randn` or
    /// from `next_f64`.
    pub fn standard_normal(&mut self) -> f64 {
        // `u1` must exclude 0, or `ln` is -inf. `next_f64` covers [0, 1), so
        // the complement covers (0, 1].
        let u1 = 1.0 - self.next_f64();
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// A uniformly distributed integer in `0..n` — numpy's
    /// `np.random.randint(n)`.
    ///
    /// Unbiased: the raw 64-bit draw is rejected when it falls in the short
    /// final block that `% n` would over-represent. The plain `next_u64() % n`
    /// everyone writes first is skewed towards the low residues by a factor of
    /// `n / 2^64`, which is invisible for `n = 3` and catastrophic for a `n`
    /// near `2^63`.
    ///
    /// # Panics
    ///
    /// Panics if `n` is 0: there is no integer below zero to return.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the residue is smaller than n, which came from a usize"
    )]
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "dezero: Rng::below needs a bound of at least 1");
        let bound = n as u64;
        // `bound.wrapping_neg() % bound` is `2^64 % bound` — the size of the
        // incomplete final block of the 64-bit range.
        let reject_below = bound.wrapping_neg() % bound;
        loop {
            let draw = self.next_u64();
            if draw >= reject_below {
                return (draw % bound) as usize;
            }
        }
    }

    /// A uniformly random permutation of `0..n` — numpy's
    /// `np.random.permutation(n)`.
    ///
    /// Fisher–Yates, drawing from this stream. It is the *same distribution* as
    /// numpy's and emphatically **not** the same sequence: numpy shuffles from
    /// the Mersenne Twister, which no generator here reproduces
    /// (`docs/ARCHITECTURE.md`). Anything that has to agree with Python on a
    /// specific ordering must ship the ordering, not a seed.
    ///
    /// # Examples
    ///
    /// ```
    /// use dezero::Rng;
    ///
    /// let mut rng = Rng::new(3);
    /// let mut p = rng.permutation(6);
    /// assert_eq!(p.len(), 6);
    /// p.sort_unstable();
    /// assert_eq!(p, vec![0, 1, 2, 3, 4, 5], "every index appears exactly once");
    /// ```
    #[must_use]
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            order.swap(i, self.below(i + 1));
        }
        order
    }

    /// An array of `shape` filled with standard normal deviates — numpy's
    /// `np.random.randn(*shape)`.
    ///
    /// Values are written in C (row-major) order.
    #[must_use]
    pub fn randn(&mut self, shape: &[usize]) -> ArrayD<f64> {
        let count: usize = shape.iter().product();
        let values: Vec<f64> = (0..count).map(|_| self.standard_normal()).collect();
        ArrayD::from_shape_vec(IxDyn(shape), values)
            .expect("the buffer was built with exactly this many elements")
    }
}

impl Default for Rng {
    /// A stream from the crate's built-in seed — the same one [`randn`]'s
    /// global stream starts on.
    fn default() -> Self {
        Self::new(DEFAULT_SEED)
    }
}

thread_local! {
    /// The stream [`randn`] draws from — numpy's global `np.random` state.
    static GLOBAL: RefCell<Rng> = RefCell::new(Rng::default());
}

/// Restarts the global stream from `seed` — numpy's `np.random.seed`.
///
/// Scoped to the calling thread, so seeding in one test cannot perturb another.
///
/// # Examples
///
/// ```
/// use dezero::{randn, seed};
///
/// seed(0);
/// let first = randn(&[4]);
/// seed(0);
/// assert_eq!(randn(&[4]), first);
/// ```
pub fn seed(seed: u64) {
    GLOBAL.with(|rng| *rng.borrow_mut() = Rng::new(seed));
}

/// Draws an array of standard normal deviates from the global stream —
/// numpy's `np.random.randn(*shape)`.
///
/// This is what a lazily-shaped [`Linear`](crate::Linear) layer initialises its
/// weights from. Use [`Rng`] directly when a private, explicitly seeded stream
/// is wanted instead.
///
/// # Examples
///
/// ```
/// use dezero::randn;
///
/// assert_eq!(randn(&[2, 5]).shape(), &[2, 5]);
/// assert_eq!(randn(&[]).shape(), &[] as &[usize], "0-d is a single deviate");
/// ```
#[must_use]
pub fn randn(shape: &[usize]) -> ArrayD<f64> {
    GLOBAL.with(|rng| rng.borrow_mut().randn(shape))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Enough samples that the moment checks below are meaningful, but still
    /// instant in a debug build.
    const SAMPLES: usize = 20_000;

    #[allow(
        clippy::cast_precision_loss,
        reason = "SAMPLES is 20000, exactly representable as an f64"
    )]
    fn sample_count() -> f64 {
        SAMPLES as f64
    }

    #[test]
    fn the_same_seed_replays_the_same_stream() {
        let a: Vec<f64> = (0..8).map(|_| Rng::new(1).next_f64()).collect();
        assert!(
            a.windows(2).all(|w| (w[0] - w[1]).abs() < f64::EPSILON),
            "a fresh Rng::new(1) each time must give the same first value"
        );

        let mut one = Rng::new(99);
        let mut two = Rng::new(99);
        for _ in 0..32 {
            assert_eq!(one.next_u64(), two.next_u64());
        }
    }

    #[test]
    fn different_seeds_give_different_streams() {
        assert_ne!(Rng::new(1).next_u64(), Rng::new(2).next_u64());
        assert_ne!(Rng::new(0).randn(&[4]), Rng::new(1).randn(&[4]));
    }

    #[test]
    fn a_zero_seed_is_not_a_fixed_point() {
        // A feedback-shift generator would be stuck at 0 forever; SplitMix64
        // advances by a constant, so it is not.
        let mut rng = Rng::new(0);
        let values: Vec<u64> = (0..4).map(|_| rng.next_u64()).collect();
        assert!(values.iter().all(|&v| v != 0));
        assert_ne!(values[0], values[1]);
    }

    #[test]
    fn uniforms_stay_inside_the_half_open_unit_interval() {
        let mut rng = Rng::new(7);
        for _ in 0..SAMPLES {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v), "{v} escaped [0, 1)");
        }
    }

    #[test]
    fn uniforms_have_the_right_mean_and_spread() {
        let mut rng = Rng::new(11);
        let values: Vec<f64> = (0..SAMPLES).map(|_| rng.next_f64()).collect();
        let mean = values.iter().sum::<f64>() / sample_count();
        assert!((mean - 0.5).abs() < 0.01, "mean was {mean}");

        // Every decile should hold roughly a tenth of the draws.
        let mut deciles = [0_usize; 10];
        for v in &values {
            let bucket = (v * 10.0).floor().clamp(0.0, 9.0);
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "clamped to 0..=9 immediately above"
            )]
            let index = bucket as usize;
            deciles[index] += 1;
        }
        for (index, count) in deciles.iter().enumerate() {
            let share = f64::from(u32::try_from(*count).expect("fits")) / sample_count();
            assert!(
                (share - 0.1).abs() < 0.02,
                "decile {index} held {share} of the draws"
            );
        }
    }

    #[test]
    fn normals_have_mean_zero_and_variance_one() {
        let mut rng = Rng::new(3);
        let values: Vec<f64> = (0..SAMPLES).map(|_| rng.standard_normal()).collect();

        let mean = values.iter().sum::<f64>() / sample_count();
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / sample_count();

        assert!(mean.abs() < 0.03, "mean was {mean}");
        assert!((variance - 1.0).abs() < 0.05, "variance was {variance}");
        assert!(values.iter().all(|v| v.is_finite()), "no inf or NaN");
    }

    #[test]
    fn normals_have_a_normal_shaped_tail() {
        // ~68% within one standard deviation, ~95% within two.
        let mut rng = Rng::new(5);
        let values: Vec<f64> = (0..SAMPLES).map(|_| rng.standard_normal()).collect();

        let share = |limit: f64| {
            let inside = values.iter().filter(|v| v.abs() < limit).count();
            f64::from(u32::try_from(inside).expect("fits")) / sample_count()
        };
        assert!(
            (share(1.0) - 0.6827).abs() < 0.02,
            "1 sigma: {}",
            share(1.0)
        );
        assert!(
            (share(2.0) - 0.9545).abs() < 0.01,
            "2 sigma: {}",
            share(2.0)
        );
    }

    #[test]
    fn randn_fills_the_requested_shape_in_c_order() {
        let mut rng = Rng::new(13);
        let flat = rng.randn(&[6]);

        let mut rng = Rng::new(13);
        let shaped = rng.randn(&[2, 3]);

        assert_eq!(shaped.shape(), &[2, 3]);
        for (index, value) in flat.iter().enumerate() {
            assert_eq!(shaped[[index / 3, index % 3]], *value);
        }
    }

    #[test]
    fn randn_of_no_shape_is_a_single_deviate() {
        let x = Rng::new(1).randn(&[]);
        assert_eq!(x.shape(), &[] as &[usize]);
        assert_eq!(x.len(), 1);
    }

    #[test]
    fn the_global_stream_advances_and_reseeds() {
        seed(123);
        let first = randn(&[3]);
        let second = randn(&[3]);
        assert_ne!(first, second, "the stream advances between calls");

        seed(123);
        assert_eq!(randn(&[3]), first, "reseeding replays it");
    }

    #[test]
    fn the_global_stream_matches_an_explicit_one() {
        seed(2024);
        let global = randn(&[5]);
        assert_eq!(Rng::new(2024).randn(&[5]), global);
    }

    // -- below / permutation ----------------------------------------------

    #[test]
    fn below_stays_inside_its_bound() {
        let mut rng = Rng::new(17);
        assert!((0..1000).all(|_| rng.below(7) < 7));
        assert!(
            (0..10).all(|_| rng.below(1) == 0),
            "a bound of 1 is constant"
        );
    }

    #[test]
    fn below_covers_its_range_roughly_evenly() {
        let mut rng = Rng::new(31);
        let mut counts = [0_u32; 6];
        for _ in 0..SAMPLES {
            counts[rng.below(6)] += 1;
        }
        let expected = sample_count() / 6.0;
        for (face, count) in counts.iter().enumerate() {
            let share = f64::from(*count) / expected;
            assert!(
                (share - 1.0).abs() < 0.1,
                "face {face} came up {count} times, expected about {expected}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "bound of at least 1")]
    fn below_rejects_a_zero_bound() {
        let _ = Rng::new(0).below(0);
    }

    #[test]
    fn a_permutation_is_a_bijection() {
        let mut rng = Rng::new(11);
        for n in [0, 1, 2, 5, 64] {
            let mut order = rng.permutation(n);
            assert_eq!(order.len(), n);
            order.sort_unstable();
            assert_eq!(order, (0..n).collect::<Vec<_>>(), "n = {n}");
        }
    }

    #[test]
    fn permutations_are_reproducible_and_not_the_identity() {
        assert_eq!(Rng::new(4).permutation(50), Rng::new(4).permutation(50));
        assert_ne!(
            Rng::new(4).permutation(50),
            (0..50).collect::<Vec<_>>(),
            "50! makes an accidental identity impossible in practice"
        );
        assert_ne!(Rng::new(4).permutation(50), Rng::new(5).permutation(50));
    }
}
