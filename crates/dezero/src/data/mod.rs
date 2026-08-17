//! [`Dataset`] and [`DataLoader`]: examples, and the batches a training loop
//! eats (steps 48–51).
//!
//! Port of `vendor/dezero-python/dezero/datasets.py` and
//! `vendor/dezero-python/dezero/dataloaders.py`.
//!
//! # The trait
//!
//! Python's `Dataset` is a base class with two dunder methods and two optional
//! callables:
//!
//! ```python
//! def __getitem__(self, index):
//!     return self.transform(self.data[index]), \
//!            self.target_transform(self.label[index])
//! def __len__(self):  return len(self.data)
//! ```
//!
//! The port splits `__getitem__` in two — [`input`](Dataset::input) and
//! [`label`](Dataset::label) — rather than returning a tuple. A batch needs the
//! two halves in different container types (a stacked [`ArrayD`] and a
//! `Vec<usize>`), so a tuple would be built only to be taken apart again; and
//! `label` returning [`Option`] states in the type what Python states by
//! convention, that a dataset may be unlabelled.
//!
//! The trait is **object-safe**: [`DataLoader`] holds a `&dyn Dataset` so that
//! one loader type serves every dataset, exactly as Python's does. The two
//! combinators that would break that ([`map_input`](Dataset::map_input) and
//! [`map_label`](Dataset::map_label)) are `where Self: Sized`, the same trick
//! [`Iterator`] uses.
//!
//! # `transform` / `target_transform`
//!
//! Python stores the two callables as *fields* and applies them inside
//! `__getitem__`; every subclass therefore inherits a pair of `lambda x: x`
//! defaults it usually does not want. The port makes them **composition**
//! instead: [`Dataset::map_input`] and [`Dataset::map_label`] wrap a dataset in
//! one that transforms on the way out.
//!
//! ```
//! use dezero::{Dataset, Spiral};
//!
//! let raw = Spiral::generate(7);
//! let doubled = raw.map_input(|x| x * 2.0);
//! assert_eq!(doubled.len(), 300);
//! ```
//!
//! A dataset whose transform is not optional bakes it in instead — which is
//! what [`Mnist`] does with Python's
//! `Compose([Flatten(), ToFloat(), Normalize(0., 255.)])`, because an MNIST
//! that handed out raw bytes would be a footgun rather than a feature.
//!
//! # Step 52 — GPU
//!
//! Python's `DataLoader` carries a `gpu` flag plus `to_cpu()`/`to_gpu()`, which
//! pick `numpy` or `cupy` for the array that a batch is stacked into. There is
//! no CUDA backend in this port and none is faked, so none of the three is
//! present: batches are always [`ArrayD<f64>`]. See `docs/DIVERGENCES.md`.

pub mod dataloader;
pub mod idx;
pub mod mnist;
pub mod spiral;

pub use crate::data::dataloader::{Batch, DataLoader};
pub use crate::data::idx::{IdxArray, IdxError};
pub use crate::data::mnist::{Mnist, MnistError};
pub use crate::data::spiral::Spiral;

use ndarray::{ArrayD, IxDyn};

/// A finite, indexable collection of examples — Python's `Dataset`.
///
/// Implementors supply a length and the two halves of an example. Indices run
/// `0..len()`; anything else panics, mirroring Python's `IndexError`.
///
/// # Examples
///
/// The whole of a dataset over data already in memory:
///
/// ```
/// use dezero::{DataLoader, Dataset};
/// use ndarray::{ArrayD, IxDyn};
///
/// struct Squares;
///
/// impl Dataset for Squares {
///     fn len(&self) -> usize {
///         4
///     }
///
///     fn input(&self, index: usize) -> ArrayD<f64> {
///         let square = u32::try_from(index * index).expect("small");
///         ArrayD::from_elem(IxDyn(&[1]), f64::from(square))
///     }
///
///     fn label(&self, index: usize) -> Option<usize> {
///         Some(index % 2)
///     }
/// }
///
/// let batches: Vec<_> = DataLoader::new(&Squares, 3, false).collect();
/// assert_eq!(batches.len(), 2, "ceil(4 / 3): the last batch is short, not dropped");
/// assert_eq!(batches[0].x.shape(), &[3, 1]);
/// assert_eq!(batches[1].x.shape(), &[1, 1]);
/// assert_eq!(batches[0].t, vec![0, 1, 0]);
/// ```
pub trait Dataset {
    /// The number of examples — Python's `__len__`.
    fn len(&self) -> usize;

    /// The input half of example `index`, after any transform.
    ///
    /// Every index must produce the same shape: [`DataLoader`] stacks them.
    ///
    /// # Panics
    ///
    /// Panics if `index` is not below [`len`](Dataset::len).
    fn input(&self, index: usize) -> ArrayD<f64>;

    /// The class label of example `index`, or `None` for an unlabelled dataset
    /// — Python's `self.label is None` branch.
    ///
    /// # Panics
    ///
    /// Panics if `index` is not below [`len`](Dataset::len).
    fn label(&self, index: usize) -> Option<usize>;

    /// Whether the dataset holds no examples at all.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Wraps this dataset in one that transforms every input — Python's
    /// `transform`.
    ///
    /// # Examples
    ///
    /// ```
    /// use dezero::{Dataset, Spiral};
    ///
    /// let scaled = Spiral::generate(1).map_input(|x| x * 10.0);
    /// assert_eq!(scaled.input(0).shape(), &[2]);
    /// ```
    fn map_input<F>(self, transform: F) -> MapInput<Self, F>
    where
        Self: Sized,
        F: Fn(ArrayD<f64>) -> ArrayD<f64>,
    {
        MapInput {
            inner: self,
            transform,
        }
    }

    /// Wraps this dataset in one that transforms every label — Python's
    /// `target_transform`.
    ///
    /// # Examples
    ///
    /// ```
    /// use dezero::{Dataset, Spiral};
    ///
    /// // Collapse three spiral arms into "arm 0" and "not arm 0".
    /// let binary = Spiral::generate(1).map_label(|t| usize::from(t != 0));
    /// assert!(binary.label(0).is_some_and(|t| t < 2));
    /// ```
    fn map_label<F>(self, transform: F) -> MapLabel<Self, F>
    where
        Self: Sized,
        F: Fn(usize) -> usize,
    {
        MapLabel {
            inner: self,
            transform,
        }
    }
}

/// A dataset whose inputs pass through a transform — see
/// [`Dataset::map_input`].
#[derive(Debug, Clone)]
pub struct MapInput<D, F> {
    inner: D,
    transform: F,
}

impl<D, F> MapInput<D, F> {
    /// The dataset underneath the transform.
    pub const fn inner(&self) -> &D {
        &self.inner
    }
}

impl<D, F> Dataset for MapInput<D, F>
where
    D: Dataset,
    F: Fn(ArrayD<f64>) -> ArrayD<f64>,
{
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn input(&self, index: usize) -> ArrayD<f64> {
        (self.transform)(self.inner.input(index))
    }

    fn label(&self, index: usize) -> Option<usize> {
        self.inner.label(index)
    }
}

/// A dataset whose labels pass through a transform — see
/// [`Dataset::map_label`].
#[derive(Debug, Clone)]
pub struct MapLabel<D, F> {
    inner: D,
    transform: F,
}

impl<D, F> MapLabel<D, F> {
    /// The dataset underneath the transform.
    pub const fn inner(&self) -> &D {
        &self.inner
    }
}

impl<D, F> Dataset for MapLabel<D, F>
where
    D: Dataset,
    F: Fn(usize) -> usize,
{
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn input(&self, index: usize) -> ArrayD<f64> {
        self.inner.input(index)
    }

    fn label(&self, index: usize) -> Option<usize> {
        self.inner.label(index).map(&self.transform)
    }
}

/// Stacks equally-shaped examples along a new leading axis — numpy's
/// `np.array([...])` over a list of arrays.
///
/// An empty list gives a `[0]` array, which is what `np.array([])` produces.
/// [`DataLoader`] never asks for one: `max_iter` is zero when the dataset is.
///
/// # Panics
///
/// Panics if the rows do not all share a shape. `ndarray` would otherwise have
/// to guess, and a silently ragged batch is a wrong gradient rather than an
/// error.
fn stack_rows(rows: &[ArrayD<f64>]) -> ArrayD<f64> {
    let Some(first) = rows.first() else {
        return ArrayD::zeros(IxDyn(&[0]));
    };

    let element_shape = first.shape();
    let mut shape = Vec::with_capacity(element_shape.len() + 1);
    shape.push(rows.len());
    shape.extend_from_slice(element_shape);

    let mut flat = Vec::with_capacity(rows.len() * first.len());
    for (position, row) in rows.iter().enumerate() {
        assert!(
            row.shape() == element_shape,
            "dezero: example {position} of a batch has shape {:?}, but example 0 has {element_shape:?}",
            row.shape()
        );
        flat.extend(row.iter().copied());
    }

    ArrayD::from_shape_vec(IxDyn(&shape), flat)
        .expect("the buffer holds exactly one element per position of the stacked shape")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    /// A dataset with a shape and labels chosen so that every value identifies
    /// the example it came from.
    struct Counting {
        count: usize,
        labelled: bool,
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
            assert!(index < self.count, "index {index} is out of range");
            arr1(&[index as f64, index as f64 + 0.5]).into_dyn()
        }

        fn label(&self, index: usize) -> Option<usize> {
            assert!(index < self.count, "index {index} is out of range");
            self.labelled.then_some(index % 3)
        }
    }

    #[test]
    fn is_empty_follows_len() {
        assert!(
            Counting {
                count: 0,
                labelled: true
            }
            .is_empty()
        );
        assert!(
            !Counting {
                count: 1,
                labelled: true
            }
            .is_empty()
        );
    }

    #[test]
    fn an_unlabelled_dataset_reports_no_label() {
        let set = Counting {
            count: 3,
            labelled: false,
        };
        assert!((0..3).all(|i| set.label(i).is_none()));
        assert_eq!(set.input(2), arr1(&[2.0, 2.5]).into_dyn());
    }

    #[test]
    fn map_input_transforms_only_the_input() {
        let set = Counting {
            count: 3,
            labelled: true,
        }
        .map_input(|x| x * 10.0);

        assert_eq!(set.len(), 3);
        assert_eq!(set.input(1), arr1(&[10.0, 15.0]).into_dyn());
        assert_eq!(set.label(1), Some(1), "the label is untouched");
        assert_eq!(set.inner().len(), 3);
    }

    #[test]
    fn map_label_transforms_only_the_label() {
        let set = Counting {
            count: 4,
            labelled: true,
        }
        .map_label(|t| t * 100);

        assert_eq!(set.input(1), arr1(&[1.0, 1.5]).into_dyn());
        assert_eq!(set.label(1), Some(100));
        assert_eq!(set.inner().len(), 4);
    }

    #[test]
    fn map_label_leaves_an_absent_label_absent() {
        let set = Counting {
            count: 2,
            labelled: false,
        }
        .map_label(|t| t + 1);
        assert_eq!(set.label(0), None);
    }

    #[test]
    fn the_combinators_compose() {
        let set = Counting {
            count: 2,
            labelled: true,
        }
        .map_input(|x| x + 1.0)
        .map_label(|t| t + 10);

        assert_eq!(set.input(0), arr1(&[1.0, 1.5]).into_dyn());
        assert_eq!(set.label(1), Some(11));
    }

    #[test]
    fn a_dataset_works_behind_a_trait_object() {
        let set = Counting {
            count: 5,
            labelled: true,
        };
        let erased: &dyn Dataset = &set;
        assert_eq!(erased.len(), 5);
        assert_eq!(erased.label(4), Some(1));
    }

    // -- stack_rows --------------------------------------------------------

    #[test]
    fn stacking_adds_a_leading_axis() {
        let rows = [
            arr1(&[1.0, 2.0]).into_dyn(),
            arr1(&[3.0, 4.0]).into_dyn(),
            arr1(&[5.0, 6.0]).into_dyn(),
        ];
        assert_eq!(
            stack_rows(&rows),
            arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).into_dyn()
        );
    }

    #[test]
    fn stacking_preserves_higher_rank_examples() {
        let rows = [
            arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn(),
            arr2(&[[5.0, 6.0], [7.0, 8.0]]).into_dyn(),
        ];
        let stacked = stack_rows(&rows);
        assert_eq!(stacked.shape(), &[2, 2, 2]);
        assert_eq!(stacked[[1, 0, 1]], 6.0);
    }

    #[test]
    fn stacking_a_scalar_example_gives_a_vector() {
        let rows = [ndarray::arr0(1.0).into_dyn(), ndarray::arr0(2.0).into_dyn()];
        assert_eq!(stack_rows(&rows), arr1(&[1.0, 2.0]).into_dyn());
    }

    #[test]
    fn stacking_nothing_gives_an_empty_vector() {
        assert_eq!(stack_rows(&[]).shape(), &[0]);
    }

    #[test]
    #[should_panic(expected = "has shape [3], but example 0 has [2]")]
    fn stacking_rejects_a_ragged_batch() {
        let _ = stack_rows(&[
            arr1(&[1.0, 2.0]).into_dyn(),
            arr1(&[1.0, 2.0, 3.0]).into_dyn(),
        ]);
    }
}
