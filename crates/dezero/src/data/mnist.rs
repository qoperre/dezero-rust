//! [`Mnist`]: the handwritten-digit dataset (step 51).
//!
//! Port of `class MNIST(Dataset)` in
//! `vendor/dezero-python/dezero/datasets.py`, minus the download.
//!
//! # What this does, and what it deliberately does not
//!
//! Python's `prepare()` calls `utils.get_file(url)`, which fetches the four
//! `.gz` archives over HTTP into `~/.dezero` the first time and reads the cache
//! afterwards. This port implements **everything except the fetch**:
//!
//! * [`Mnist::from_bytes`] — the IDX/gzip decoding and the
//!   `Compose([Flatten(), ToFloat(), Normalize(0., 255.)])` transform;
//! * [`Mnist::from_files`] — the same from two paths;
//! * [`Mnist::from_cache_dir`] and [`Mnist::from_cache`] — the cache lookup,
//!   including Python's `~/.dezero` location and its four filenames;
//! * [`MnistError::Missing`] — what a cache miss says, naming the exact file
//!   and the exact URL to put there.
//!
//! There is **no HTTP client**, because adding one is a dependency decision
//! nobody asked for: `flate2` is here for gzip and that is the whole of the new
//! dependency budget (`docs/DIVERGENCES.md`). A cache miss therefore ends in an
//! error that tells the user what to download rather than downloading it.
//!
//! Consequently the tests cover the decode path and the cache path, over files
//! they build themselves; **nothing in this crate's test suite touches the
//! network**, and nothing needs the real 11 MB archives to be present.
//!
//! # The transform
//!
//! ```python
//! transform=Compose([Flatten(), ToFloat(), Normalize(0., 255.)])
//! ```
//!
//! An example is stored as `[1, 28, 28]` bytes and handed out as a `[784]`
//! vector of `f64` in `[0, 1]`. That is baked in rather than optional: an MNIST
//! that handed out raw bytes would silently train a network on inputs two
//! orders of magnitude too large. [`Mnist::raw_image`] is there for the caller
//! who genuinely wants the pixels, and the step-56 convolutional path.

use std::error::Error;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use ndarray::{ArrayD, IxDyn};

use crate::data::Dataset;
use crate::data::idx::{IdxArray, IdxError, read_idx};

/// Where Python's `utils.get_file` caches downloads, relative to `$HOME`.
const CACHE_DIRECTORY: &str = ".dezero";

/// The mirror `datasets.py` points at, the original `yann.lecun.com` host
/// having stopped serving the files.
const MNIST_URL_BASE: &str = "https://ossci-datasets.s3.amazonaws.com/mnist/";

/// `train_files` / `test_files` in Python's `MNIST.prepare`.
const TRAIN_FILES: (&str, &str) = ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz");
const TEST_FILES: (&str, &str) = ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz");

/// The largest value a pixel byte can take — Python's `Normalize(0., 255.)`.
const PIXEL_MAX: f64 = 255.0;

/// Why MNIST could not be loaded.
#[derive(Debug)]
pub enum MnistError {
    /// A file was there but could not be read.
    Io {
        /// The file being read.
        path: PathBuf,
        /// What the operating system said.
        source: io::Error,
    },
    /// A file was read but is not the IDX archive it should be.
    Idx {
        /// The file being decoded.
        path: PathBuf,
        /// What the decoder found wrong.
        source: IdxError,
    },
    /// The cache does not hold this file, and this port does not download.
    ///
    /// The message names both halves of the fix: where the file goes, and
    /// where to get it.
    Missing {
        /// Where the file was looked for.
        path: PathBuf,
        /// Where the file can be downloaded from.
        url: String,
    },
    /// `$HOME` (or `%USERPROFILE%`) is not set, so there is no `~/.dezero`.
    NoHomeDirectory,
    /// The image archive is not a stack of two-dimensional images.
    ImageShape {
        /// The shape the archive's header declared.
        shape: Vec<usize>,
    },
    /// The label archive is not a flat list of labels.
    LabelShape {
        /// The shape the archive's header declared.
        shape: Vec<usize>,
    },
    /// The two archives describe different numbers of examples.
    CountMismatch {
        /// Examples in the image archive.
        images: usize,
        /// Examples in the label archive.
        labels: usize,
    },
}

impl fmt::Display for MnistError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(f, "MNIST: cannot read {}: {source}", path.display())
            }
            Self::Idx { path, source } => {
                write!(f, "MNIST: {} is unreadable: {source}", path.display())
            }
            Self::Missing { path, url } => write!(
                f,
                "MNIST: {} is not in the cache. This port does not download; \
                 fetch {url} and save it there.",
                path.display()
            ),
            Self::NoHomeDirectory => write!(
                f,
                "MNIST: neither HOME nor USERPROFILE is set, so the ~/.dezero cache \
                 has no location; pass a directory to Mnist::from_cache_dir instead"
            ),
            Self::ImageShape { shape } => write!(
                f,
                "MNIST: the image archive should be [count, rows, columns] but is {shape:?}"
            ),
            Self::LabelShape { shape } => write!(
                f,
                "MNIST: the label archive should be [count] but is {shape:?}"
            ),
            Self::CountMismatch { images, labels } => {
                write!(f, "MNIST: {images} images but {labels} labels")
            }
        }
    }
}

impl Error for MnistError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Idx { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// The MNIST handwritten digits — Python's `datasets.MNIST`.
///
/// Pixels are kept as bytes and normalised on the way out, which is both what
/// Python does (the transform runs inside `__getitem__`) and what keeps the
/// training set to 47 MB rather than 376 MB.
///
/// # Examples
///
/// Building one from IDX bytes — two 2×2 "images" with their labels:
///
/// ```
/// use dezero::{Dataset, Mnist};
///
/// let images = [0, 0, 8, 3, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 51, 102, 255, 0, 0, 0, 0];
/// let labels = [0, 0, 8, 1, 0, 0, 0, 2, 7, 3];
///
/// let set = Mnist::from_bytes(&images, &labels).expect("well-formed archives");
///
/// assert_eq!(set.len(), 2);
/// assert_eq!(set.rows(), 2);
/// assert_eq!(set.columns(), 2);
/// assert_eq!(set.label(0), Some(7));
///
/// // Flattened to [rows * columns] and scaled into [0, 1].
/// let x = set.input(0);
/// assert_eq!(x.shape(), &[4]);
/// assert!((x[[3]] - 1.0).abs() < 1e-12, "255 becomes 1.0");
/// assert!((x[[1]] - 51.0 / 255.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mnist {
    /// `count * rows * columns` pixel bytes, in C order.
    images: Vec<u8>,
    labels: Vec<u8>,
    count: usize,
    rows: usize,
    columns: usize,
}

impl Mnist {
    /// Builds the dataset from two decoded IDX archives.
    ///
    /// # Errors
    ///
    /// Returns [`MnistError::ImageShape`], [`MnistError::LabelShape`] or
    /// [`MnistError::CountMismatch`] if the two archives do not describe the
    /// same number of equally-sized images.
    pub fn from_idx(images: &IdxArray, labels: &IdxArray) -> Result<Self, MnistError> {
        let [count, rows, columns] = *images.shape() else {
            return Err(MnistError::ImageShape {
                shape: images.shape().to_vec(),
            });
        };
        let [label_count] = *labels.shape() else {
            return Err(MnistError::LabelShape {
                shape: labels.shape().to_vec(),
            });
        };
        if count != label_count {
            return Err(MnistError::CountMismatch {
                images: count,
                labels: label_count,
            });
        }

        Ok(Self {
            images: images.data().to_vec(),
            labels: labels.data().to_vec(),
            count,
            rows,
            columns,
        })
    }

    /// Builds the dataset from two raw archives, gzipped or not.
    ///
    /// # Errors
    ///
    /// Returns [`MnistError::Idx`] if either archive is malformed, or one of
    /// the shape errors from [`from_idx`](Self::from_idx).
    pub fn from_bytes(images: &[u8], labels: &[u8]) -> Result<Self, MnistError> {
        let decode = |bytes: &[u8], what: &str| {
            read_idx(bytes).map_err(|source| MnistError::Idx {
                path: PathBuf::from(what),
                source,
            })
        };
        Self::from_idx(
            &decode(images, "<image bytes>")?,
            &decode(labels, "<label bytes>")?,
        )
    }

    /// Builds the dataset from two files, gzipped or not.
    ///
    /// # Errors
    ///
    /// Returns [`MnistError::Missing`] if a path does not exist,
    /// [`MnistError::Io`] if it cannot be read, or [`MnistError::Idx`] if its
    /// contents are not a valid archive.
    pub fn from_files(images: &Path, labels: &Path) -> Result<Self, MnistError> {
        let image_bytes = read_file(images)?;
        let label_bytes = read_file(labels)?;
        Self::from_idx(
            &decode_file(images, &image_bytes)?,
            &decode_file(labels, &label_bytes)?,
        )
    }

    /// Loads the training or test split from a cache directory — the second
    /// half of Python's `get_file`, without the first.
    ///
    /// Looks for the four names Python uses. The `.gz` archive is preferred;
    /// the same name without the extension is accepted too, so a caller who
    /// has already decompressed does not have to re-compress.
    ///
    /// # Errors
    ///
    /// Returns [`MnistError::Missing`], naming the file and its download URL,
    /// when neither form is present; otherwise the errors of
    /// [`from_files`](Self::from_files).
    pub fn from_cache_dir(directory: &Path, train: bool) -> Result<Self, MnistError> {
        let (image_name, label_name) = if train { TRAIN_FILES } else { TEST_FILES };
        let images = locate(directory, image_name)?;
        let labels = locate(directory, label_name)?;
        Self::from_files(&images, &labels)
    }

    /// Loads a split from `~/.dezero` — Python's `cache_dir`.
    ///
    /// # Errors
    ///
    /// Returns [`MnistError::NoHomeDirectory`] if the home directory cannot be
    /// determined, otherwise the errors of
    /// [`from_cache_dir`](Self::from_cache_dir) — in particular
    /// [`MnistError::Missing`], which is what an unprimed cache gives, because
    /// this port does not download.
    pub fn from_cache(train: bool) -> Result<Self, MnistError> {
        let directory = cache_dir().ok_or(MnistError::NoHomeDirectory)?;
        Self::from_cache_dir(&directory, train)
    }

    /// The number of examples.
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Pixel rows per image — 28 for real MNIST.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Pixel columns per image — 28 for real MNIST.
    #[must_use]
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// Pixels per image, and so the width of the flattened input vector.
    #[must_use]
    pub fn pixels(&self) -> usize {
        self.rows * self.columns
    }

    /// One image as `[1, rows, columns]` with its original `0..=255` values —
    /// Python's untransformed `self.data[index]`.
    ///
    /// The leading axis is the (single) colour channel, which is what a
    /// convolution expects.
    ///
    /// # Panics
    ///
    /// Panics if `index` is not below [`count`](Self::count).
    #[must_use]
    pub fn raw_image(&self, index: usize) -> ArrayD<f64> {
        let pixels = self.pixel_slice(index);
        ArrayD::from_shape_vec(
            IxDyn(&[1, self.rows, self.columns]),
            pixels.iter().map(|&byte| f64::from(byte)).collect(),
        )
        .expect("the slice holds exactly rows * columns pixels")
    }

    /// The URL an archive of this dataset is downloaded from, for a caller that
    /// wants to prime the cache itself.
    ///
    /// # Examples
    ///
    /// ```
    /// use dezero::Mnist;
    ///
    /// assert!(Mnist::urls(true).0.ends_with("train-images-idx3-ubyte.gz"));
    /// assert!(Mnist::urls(false).1.ends_with("t10k-labels-idx1-ubyte.gz"));
    /// ```
    #[must_use]
    pub fn urls(train: bool) -> (String, String) {
        let (images, labels) = if train { TRAIN_FILES } else { TEST_FILES };
        (
            format!("{MNIST_URL_BASE}{images}"),
            format!("{MNIST_URL_BASE}{labels}"),
        )
    }

    /// The bytes of one image.
    ///
    /// # Panics
    ///
    /// Panics if `index` is not below [`count`](Self::count).
    fn pixel_slice(&self, index: usize) -> &[u8] {
        assert!(
            index < self.count,
            "dezero: MNIST holds {} images, so index {index} is out of range",
            self.count
        );
        let start = index * self.pixels();
        &self.images[start..start + self.pixels()]
    }
}

impl Dataset for Mnist {
    fn len(&self) -> usize {
        self.count
    }

    /// The image flattened to `[rows * columns]` and scaled into `[0, 1]` —
    /// Python's `Compose([Flatten(), ToFloat(), Normalize(0., 255.)])`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is not below [`len`](Dataset::len).
    fn input(&self, index: usize) -> ArrayD<f64> {
        let pixels = self.pixel_slice(index);
        ArrayD::from_shape_vec(
            IxDyn(&[self.pixels()]),
            pixels
                .iter()
                .map(|&byte| f64::from(byte) / PIXEL_MAX)
                .collect(),
        )
        .expect("the slice holds exactly rows * columns pixels")
    }

    /// The digit, `0..=9`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is not below [`len`](Dataset::len).
    fn label(&self, index: usize) -> Option<usize> {
        let label = self.labels.get(index).copied().unwrap_or_else(|| {
            panic!(
                "dezero: MNIST holds {} labels, so index {index} is out of range",
                self.labels.len()
            )
        });
        Some(usize::from(label))
    }
}

/// Python's `cache_dir = os.path.join(os.path.expanduser('~'), '.dezero')`.
///
/// `HOME` first, then `USERPROFILE` for a Windows shell that does not set it.
/// Reading the environment rather than calling `std::env::home_dir` keeps this
/// free of that function's long deprecation history.
///
/// # Examples
///
/// ```
/// // Present on any normal system; `None` only where neither variable is set.
/// if let Some(directory) = dezero::mnist_cache_dir() {
///     assert!(directory.ends_with(".dezero"));
/// }
/// ```
#[must_use]
pub fn cache_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .filter(|home| !home.is_empty())
        .map(|home| PathBuf::from(home).join(CACHE_DIRECTORY))
}

/// Finds `name` in `directory`, accepting the archive with or without its
/// `.gz` extension.
fn locate(directory: &Path, name: &str) -> Result<PathBuf, MnistError> {
    let compressed = directory.join(name);
    if compressed.exists() {
        return Ok(compressed);
    }

    let plain = directory.join(name.trim_end_matches(".gz"));
    if plain.exists() {
        return Ok(plain);
    }

    Err(MnistError::Missing {
        path: compressed,
        url: format!("{MNIST_URL_BASE}{name}"),
    })
}

fn read_file(path: &Path) -> Result<Vec<u8>, MnistError> {
    fs::read(path).map_err(|source| {
        if source.kind() == io::ErrorKind::NotFound {
            let name = path
                .file_name()
                .map_or_else(String::new, |name| name.to_string_lossy().into_owned());
            MnistError::Missing {
                path: path.to_path_buf(),
                url: format!("{MNIST_URL_BASE}{name}"),
            }
        } else {
            MnistError::Io {
                path: path.to_path_buf(),
                source,
            }
        }
    })
}

fn decode_file(path: &Path, bytes: &[u8]) -> Result<IdxArray, MnistError> {
    read_idx(bytes).map_err(|source| MnistError::Idx {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn idx_file(shape: &[usize], payload: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0x00, 0x00, 0x08, u8::try_from(shape.len()).expect("rank")];
        for &size in shape {
            bytes.extend_from_slice(&u32::try_from(size).expect("dimension").to_be_bytes());
        }
        bytes.extend_from_slice(payload);
        bytes
    }

    fn gzipped(bytes: &[u8]) -> Vec<u8> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(bytes).expect("in-memory write");
        encoder.finish().expect("in-memory finish")
    }

    /// Three 2x2 "digits", with pixel values that identify their position.
    fn image_archive() -> Vec<u8> {
        idx_file(&[3, 2, 2], &[0, 51, 102, 255, 10, 20, 30, 40, 1, 2, 3, 4])
    }

    fn label_archive() -> Vec<u8> {
        idx_file(&[3], &[7, 0, 9])
    }

    fn tiny() -> Mnist {
        Mnist::from_bytes(&image_archive(), &label_archive()).expect("well-formed archives")
    }

    /// A private directory under the system temp dir, unique per call so that
    /// parallel tests cannot collide.
    fn scratch_directory(tag: &str) -> PathBuf {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "dezero-mnist-{tag}-{}-{unique}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir_all(&directory).expect("a writable temp directory");
        directory
    }

    // -- decoding ----------------------------------------------------------

    #[test]
    fn the_archives_decode_into_examples() {
        let set = tiny();
        assert_eq!(set.len(), 3);
        assert_eq!(set.count(), 3);
        assert_eq!(set.rows(), 2);
        assert_eq!(set.columns(), 2);
        assert_eq!(set.pixels(), 4);
        assert!(!set.is_empty());
    }

    #[test]
    fn an_input_is_flattened_and_normalised() {
        let x = tiny().input(0);
        assert_eq!(x.shape(), &[4]);
        for (actual, expected) in x.iter().zip([0.0, 51.0 / 255.0, 102.0 / 255.0, 1.0]) {
            assert!((actual - expected).abs() < 1e-15, "{actual} vs {expected}");
        }
        assert!(
            x.iter().all(|v| (0.0..=1.0).contains(v)),
            "normalisation must land in [0, 1]"
        );
    }

    #[test]
    fn every_example_gets_its_own_pixels() {
        let set = tiny();
        let scaled = |values: [f64; 4]| values.map(|v| v / PIXEL_MAX);
        for (index, expected) in [
            scaled([0.0, 51.0, 102.0, 255.0]),
            scaled([10.0, 20.0, 30.0, 40.0]),
            scaled([1.0, 2.0, 3.0, 4.0]),
        ]
        .into_iter()
        .enumerate()
        {
            let actual = set.input(index);
            for (a, e) in actual.iter().zip(expected) {
                assert!((a - e).abs() < 1e-15, "example {index}: {a} vs {e}");
            }
        }
    }

    #[test]
    fn labels_come_through_as_class_indices() {
        let set = tiny();
        assert_eq!(set.label(0), Some(7));
        assert_eq!(set.label(1), Some(0));
        assert_eq!(set.label(2), Some(9));
    }

    #[test]
    fn a_raw_image_keeps_its_channel_and_its_byte_values() {
        let image = tiny().raw_image(0);
        assert_eq!(image.shape(), &[1, 2, 2], "channel, rows, columns");
        assert_eq!(image[[0, 0, 1]], 51.0);
        assert_eq!(image[[0, 1, 1]], 255.0);
    }

    #[test]
    fn a_gzipped_archive_gives_the_same_dataset() {
        let compressed = Mnist::from_bytes(&gzipped(&image_archive()), &gzipped(&label_archive()))
            .expect("gzipped archives");
        assert_eq!(compressed, tiny());
    }

    #[test]
    fn one_gzipped_and_one_plain_archive_is_fine_too() {
        let mixed = Mnist::from_bytes(&gzipped(&image_archive()), &label_archive())
            .expect("mixed archives");
        assert_eq!(mixed, tiny());
    }

    #[test]
    #[should_panic(expected = "index 3 is out of range")]
    fn reading_past_the_last_image_panics() {
        let _ = tiny().input(3);
    }

    #[test]
    #[should_panic(expected = "index 3 is out of range")]
    fn reading_past_the_last_label_panics() {
        let _ = tiny().label(3);
    }

    // -- rejections --------------------------------------------------------

    #[test]
    fn a_two_dimensional_image_archive_is_rejected() {
        let flat = idx_file(&[3, 4], &[0; 12]);
        assert!(matches!(
            Mnist::from_bytes(&flat, &label_archive()),
            Err(MnistError::ImageShape { .. })
        ));
    }

    #[test]
    fn a_two_dimensional_label_archive_is_rejected() {
        let nested = idx_file(&[3, 1], &[7, 0, 9]);
        assert!(matches!(
            Mnist::from_bytes(&image_archive(), &nested),
            Err(MnistError::LabelShape { .. })
        ));
    }

    #[test]
    fn mismatched_counts_are_rejected() {
        let short = idx_file(&[2], &[7, 0]);
        match Mnist::from_bytes(&image_archive(), &short) {
            Err(MnistError::CountMismatch { images, labels }) => {
                assert_eq!((images, labels), (3, 2));
            }
            other => panic!("expected a count mismatch, got {other:?}"),
        }
    }

    #[test]
    fn a_malformed_archive_reports_the_idx_error() {
        match Mnist::from_bytes(b"not an archive", &label_archive()) {
            Err(MnistError::Idx { source, .. }) => {
                assert!(matches!(source, IdxError::NotIdx { .. }));
            }
            other => panic!("expected an IDX error, got {other:?}"),
        }
    }

    // -- files and the cache ----------------------------------------------

    #[test]
    fn a_dataset_loads_from_two_files() {
        let directory = scratch_directory("files");
        let images = directory.join("images.idx");
        let labels = directory.join("labels.idx");
        fs::write(&images, image_archive()).expect("write");
        fs::write(&labels, label_archive()).expect("write");

        assert_eq!(Mnist::from_files(&images, &labels).expect("loads"), tiny());
        fs::remove_dir_all(&directory).expect("cleanup");
    }

    #[test]
    fn the_cache_path_finds_the_gzipped_archives() {
        let directory = scratch_directory("cache-gz");
        fs::write(directory.join(TRAIN_FILES.0), gzipped(&image_archive())).expect("write");
        fs::write(directory.join(TRAIN_FILES.1), gzipped(&label_archive())).expect("write");

        assert_eq!(
            Mnist::from_cache_dir(&directory, true).expect("loads"),
            tiny()
        );
        assert!(
            matches!(
                Mnist::from_cache_dir(&directory, false),
                Err(MnistError::Missing { .. })
            ),
            "the test split was not primed"
        );
        fs::remove_dir_all(&directory).expect("cleanup");
    }

    #[test]
    fn the_cache_path_also_accepts_decompressed_archives() {
        let directory = scratch_directory("cache-plain");
        fs::write(
            directory.join(TEST_FILES.0.trim_end_matches(".gz")),
            image_archive(),
        )
        .expect("write");
        fs::write(
            directory.join(TEST_FILES.1.trim_end_matches(".gz")),
            label_archive(),
        )
        .expect("write");

        assert_eq!(
            Mnist::from_cache_dir(&directory, false).expect("loads"),
            tiny()
        );
        fs::remove_dir_all(&directory).expect("cleanup");
    }

    #[test]
    fn a_cache_miss_names_the_file_and_the_url() {
        let directory = scratch_directory("cache-miss");
        let Err(error) = Mnist::from_cache_dir(&directory, true) else {
            panic!("an empty cache cannot produce a dataset");
        };

        let MnistError::Missing { path, url } = &error else {
            panic!("expected a cache miss, got {error:?}");
        };
        assert!(path.ends_with(TRAIN_FILES.0), "{}", path.display());
        assert_eq!(*url, Mnist::urls(true).0);

        let rendered = error.to_string();
        assert!(rendered.contains("does not download"), "{rendered}");
        assert!(rendered.contains("ossci-datasets"), "{rendered}");
        fs::remove_dir_all(&directory).expect("cleanup");
    }

    #[test]
    fn a_corrupt_cached_archive_reports_where_the_problem_is() {
        let directory = scratch_directory("cache-corrupt");
        fs::write(directory.join(TRAIN_FILES.0), b"neither gzip nor idx").expect("write");
        fs::write(directory.join(TRAIN_FILES.1), gzipped(&label_archive())).expect("write");

        match Mnist::from_cache_dir(&directory, true) {
            Err(MnistError::Idx { path, source }) => {
                assert!(path.ends_with(TRAIN_FILES.0), "{}", path.display());
                assert!(matches!(source, IdxError::NotIdx { .. }));
            }
            other => panic!("expected an IDX error, got {other:?}"),
        }
        fs::remove_dir_all(&directory).expect("cleanup");
    }

    // -- the download side, which exists only as a description ------------

    #[test]
    fn the_urls_are_the_reference_mirrors_four_files() {
        let (train_images, train_labels) = Mnist::urls(true);
        let (test_images, test_labels) = Mnist::urls(false);
        for url in [&train_images, &train_labels, &test_images, &test_labels] {
            assert!(
                url.starts_with("https://ossci-datasets.s3.amazonaws.com/mnist/"),
                "{url}"
            );
        }
        assert!(train_images.ends_with("train-images-idx3-ubyte.gz"));
        assert!(train_labels.ends_with("train-labels-idx1-ubyte.gz"));
        assert!(test_images.ends_with("t10k-images-idx3-ubyte.gz"));
        assert!(test_labels.ends_with("t10k-labels-idx1-ubyte.gz"));
    }

    #[test]
    fn the_cache_directory_is_dot_dezero_under_the_home_directory() {
        match cache_dir() {
            Some(directory) => {
                assert!(
                    directory.ends_with(CACHE_DIRECTORY),
                    "{}",
                    directory.display()
                );
                assert!(directory.parent().is_some());
            }
            None => {
                // Only on a system with neither HOME nor USERPROFILE, where
                // `from_cache` correctly refuses to guess.
                assert!(matches!(
                    Mnist::from_cache(true),
                    Err(MnistError::NoHomeDirectory)
                ));
            }
        }
    }

    #[test]
    fn every_error_renders_something_a_reader_can_act_on() {
        let errors = [
            MnistError::Io {
                path: PathBuf::from("a"),
                source: io::Error::from(io::ErrorKind::PermissionDenied),
            },
            MnistError::Idx {
                path: PathBuf::from("b"),
                source: IdxError::ShapeOverflow,
            },
            MnistError::Missing {
                path: PathBuf::from("c"),
                url: "https://example.invalid/c".to_owned(),
            },
            MnistError::NoHomeDirectory,
            MnistError::ImageShape { shape: vec![1, 2] },
            MnistError::LabelShape { shape: vec![1, 2] },
            MnistError::CountMismatch {
                images: 3,
                labels: 2,
            },
        ];

        for error in &errors {
            assert!(error.to_string().starts_with("MNIST: "), "{error}");
        }
        assert!(errors[0].source().is_some(), "an io error keeps its cause");
        assert!(errors[1].source().is_some(), "an IDX error keeps its cause");
        assert!(errors[3].source().is_none());
    }
}
