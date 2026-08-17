//! The IDX file format, and the gzip wrapper MNIST ships it in (step 51).
//!
//! Python gets this for free — `gzip.open` plus one line of numpy:
//!
//! ```python
//! with gzip.open(filepath, 'rb') as f:
//!     data = np.frombuffer(f.read(), np.uint8, offset=16)
//! data = data.reshape(-1, 1, 28, 28)
//! ```
//!
//! which is not a reader so much as a hard-coded seek: the 16 is
//! `4 + 4 * 3` for a rank-3 file and the shape in the header is thrown away and
//! replaced by a literal `28`. That works exactly once, for exactly MNIST.
//! The port parses the header instead, so a wrong file is an error rather than
//! a plausibly-shaped array of garbage.
//!
//! # The format
//!
//! ```text
//! byte 0     0x00
//! byte 1     0x00
//! byte 2     element type   (0x08 = unsigned byte; MNIST uses only this)
//! byte 3     number of dimensions
//! bytes 4..  one big-endian u32 per dimension
//! then       the elements, in C (row-major) order
//! ```
//!
//! Big-endian, in a format from 1998 — `u32::from_be_bytes` and no
//! configuration.
//!
//! # gzip
//!
//! [`read_idx`] sniffs the two-byte gzip magic and decompresses when it is
//! there, so a caller never has to care whether its file is `.gz`. The DEFLATE
//! decoder itself is `flate2`, this crate's only dependency besides `ndarray`
//! (`docs/DIVERGENCES.md`): hand-writing one is a large, subtle unit of work
//! that teaches nothing about automatic differentiation.

use std::error::Error;
use std::fmt;
use std::io::Read;

use flate2::read::GzDecoder;

/// The magic bytes every IDX file starts with.
const IDX_MAGIC: [u8; 2] = [0x00, 0x00];

/// The one element type MNIST uses: `unsigned byte`.
const ELEMENT_TYPE_U8: u8 = 0x08;

/// gzip's magic number, RFC 1952 §2.3.1.
const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];

/// The header is at least the magic, the type, and the rank.
const HEADER_PREFIX: usize = 4;

/// A decoded IDX file: a shape and its bytes in C order.
///
/// # Examples
///
/// ```
/// use dezero::IdxArray;
///
/// // A 2x3 array of unsigned bytes.
/// let bytes = [0, 0, 8, 2, 0, 0, 0, 2, 0, 0, 0, 3, 1, 2, 3, 4, 5, 6];
/// let array = dezero::read_idx(&bytes).expect("a well-formed IDX file");
///
/// assert_eq!(array.shape(), &[2, 3]);
/// assert_eq!(array.data(), &[1, 2, 3, 4, 5, 6]);
/// assert_eq!(array.len(), 6);
/// # let _: IdxArray = array;
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdxArray {
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl IdxArray {
    /// The dimensions, outermost first.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The elements, in C (row-major) order.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// The total number of elements — the product of [`shape`](Self::shape).
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the array holds no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The number of dimensions.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Consumes the array and returns its bytes.
    #[must_use]
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }
}

/// Why an IDX file could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IdxError {
    /// The file ended before the header or the payload did.
    Truncated {
        /// Bytes the header says are needed.
        needed: usize,
        /// Bytes actually present.
        found: usize,
    },
    /// The first two bytes were not `00 00`, so this is not an IDX file at all.
    NotIdx {
        /// The two bytes that were there instead.
        magic: [u8; 2],
    },
    /// The element type byte named a type this reader does not decode.
    ///
    /// The format defines signed bytes, shorts, ints, floats and doubles as
    /// well; MNIST uses none of them, and shipping four untested branches to
    /// say so would be worse than saying it here.
    UnsupportedElementType {
        /// The type code from byte 2 of the header.
        code: u8,
    },
    /// The payload length disagreed with the product of the dimensions.
    PayloadLength {
        /// Elements the header's shape calls for.
        expected: usize,
        /// Bytes left after the header.
        found: usize,
    },
    /// The dimensions multiply to more than a `usize` can hold.
    ShapeOverflow,
    /// The gzip container could not be decompressed.
    Gzip(String),
}

impl fmt::Display for IdxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated { needed, found } => write!(
                f,
                "IDX: the file needs at least {needed} bytes but has {found}"
            ),
            Self::NotIdx { magic } => write!(
                f,
                "IDX: expected the magic bytes 00 00 but found {:02x} {:02x}",
                magic[0], magic[1]
            ),
            Self::UnsupportedElementType { code } => write!(
                f,
                "IDX: element type 0x{code:02x} is not supported; this reader decodes \
                 0x08 (unsigned byte), which is what MNIST uses"
            ),
            Self::PayloadLength { expected, found } => write!(
                f,
                "IDX: the header describes {expected} elements but the file holds {found} bytes"
            ),
            Self::ShapeOverflow => {
                write!(f, "IDX: the dimensions multiply to more than a usize holds")
            }
            Self::Gzip(message) => write!(f, "IDX: the gzip container is unreadable: {message}"),
        }
    }
}

impl Error for IdxError {}

/// Reads an IDX file, transparently decompressing a gzip container.
///
/// This is the entry point; [`decode_idx`] is the same thing without the gzip
/// sniff, for a caller that has already decompressed.
///
/// # Errors
///
/// Returns [`IdxError`] if the gzip stream is corrupt or the IDX header does
/// not describe the bytes that follow it.
///
/// # Examples
///
/// ```
/// // A rank-1 file of three bytes.
/// let plain = [0, 0, 8, 1, 0, 0, 0, 3, 7, 8, 9];
/// assert_eq!(dezero::read_idx(&plain).expect("valid").data(), &[7, 8, 9]);
///
/// // The same content, and it does not matter that it is not gzipped.
/// assert!(dezero::read_idx(&[0x1f, 0x8b, 0]).is_err(), "a truncated gzip stream is an error");
/// ```
pub fn read_idx(bytes: &[u8]) -> Result<IdxArray, IdxError> {
    if bytes.starts_with(&GZIP_MAGIC) {
        let mut decompressed = Vec::new();
        GzDecoder::new(bytes)
            .read_to_end(&mut decompressed)
            .map_err(|e| IdxError::Gzip(e.to_string()))?;
        decode_idx(&decompressed)
    } else {
        decode_idx(bytes)
    }
}

/// Decodes an uncompressed IDX file.
///
/// # Errors
///
/// Returns [`IdxError`] if the magic bytes, the element type, the rank or the
/// payload length are wrong.
pub fn decode_idx(bytes: &[u8]) -> Result<IdxArray, IdxError> {
    if bytes.len() < HEADER_PREFIX {
        return Err(IdxError::Truncated {
            needed: HEADER_PREFIX,
            found: bytes.len(),
        });
    }
    if bytes[..2] != IDX_MAGIC {
        return Err(IdxError::NotIdx {
            magic: [bytes[0], bytes[1]],
        });
    }
    if bytes[2] != ELEMENT_TYPE_U8 {
        return Err(IdxError::UnsupportedElementType { code: bytes[2] });
    }

    let ndim = usize::from(bytes[3]);
    let header_len = HEADER_PREFIX + 4 * ndim;
    if bytes.len() < header_len {
        return Err(IdxError::Truncated {
            needed: header_len,
            found: bytes.len(),
        });
    }

    let mut shape = Vec::with_capacity(ndim);
    let mut expected = 1_usize;
    for dimension in 0..ndim {
        let start = HEADER_PREFIX + 4 * dimension;
        let mut field = [0_u8; 4];
        field.copy_from_slice(&bytes[start..start + 4]);
        let size =
            usize::try_from(u32::from_be_bytes(field)).map_err(|_| IdxError::ShapeOverflow)?;
        expected = expected.checked_mul(size).ok_or(IdxError::ShapeOverflow)?;
        shape.push(size);
    }

    let payload = &bytes[header_len..];
    if payload.len() != expected {
        return Err(IdxError::PayloadLength {
            expected,
            found: payload.len(),
        });
    }

    Ok(IdxArray {
        shape,
        data: payload.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::io::Write;

    /// Builds an IDX file in memory: the header this module's docs describe,
    /// then the payload.
    fn idx_file(element_type: u8, shape: &[usize], payload: &[u8]) -> Vec<u8> {
        let mut bytes = vec![
            0x00,
            0x00,
            element_type,
            u8::try_from(shape.len()).expect("rank"),
        ];
        for &size in shape {
            bytes.extend_from_slice(&u32::try_from(size).expect("dimension").to_be_bytes());
        }
        bytes.extend_from_slice(payload);
        bytes
    }

    fn gzipped(bytes: &[u8]) -> Vec<u8> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(bytes).expect("in-memory write");
        encoder.finish().expect("in-memory finish")
    }

    #[test]
    fn a_rank_one_file_round_trips() {
        let file = idx_file(0x08, &[4], &[10, 20, 30, 40]);
        let array = decode_idx(&file).expect("well formed");
        assert_eq!(array.shape(), &[4]);
        assert_eq!(array.data(), &[10, 20, 30, 40]);
        assert_eq!(array.ndim(), 1);
        assert_eq!(array.len(), 4);
        assert!(!array.is_empty());
    }

    #[test]
    fn a_rank_three_file_keeps_its_dimensions_and_c_order() {
        // The MNIST image shape, in miniature: 2 images of 3x2 pixels.
        let payload: Vec<u8> = (1..=12).collect();
        let file = idx_file(0x08, &[2, 3, 2], &payload);
        let array = decode_idx(&file).expect("well formed");
        assert_eq!(array.shape(), &[2, 3, 2]);
        assert_eq!(array.data()[..6], [1, 2, 3, 4, 5, 6]);
        assert_eq!(array.data()[6..], [7, 8, 9, 10, 11, 12]);
    }

    #[test]
    fn dimensions_are_read_big_endian() {
        // 0x00000102 is 258 big-endian and 33_554_432 little-endian, so a
        // reader with the bytes the wrong way round cannot pass this.
        let file = idx_file(0x08, &[258], &[7_u8; 258]);
        assert_eq!(file[4..8], [0x00, 0x00, 0x01, 0x02]);
        assert_eq!(decode_idx(&file).expect("well formed").shape(), &[258]);
    }

    #[test]
    fn a_rank_zero_file_is_a_single_element() {
        let file = idx_file(0x08, &[], &[42]);
        let array = decode_idx(&file).expect("well formed");
        assert_eq!(array.shape(), &[] as &[usize]);
        assert_eq!(array.data(), &[42]);
    }

    #[test]
    fn a_zero_length_dimension_gives_an_empty_array() {
        let array = decode_idx(&idx_file(0x08, &[0, 28], &[])).expect("well formed");
        assert_eq!(array.shape(), &[0, 28]);
        assert!(array.is_empty());
    }

    #[test]
    fn into_data_hands_over_the_buffer() {
        let array = decode_idx(&idx_file(0x08, &[3], &[1, 2, 3])).expect("well formed");
        assert_eq!(array.into_data(), vec![1, 2, 3]);
    }

    // -- gzip --------------------------------------------------------------

    #[test]
    fn a_gzipped_file_decodes_to_the_same_array() {
        let payload: Vec<u8> = (0..=255).collect();
        let plain = idx_file(0x08, &[16, 16], &payload);
        let compressed = gzipped(&plain);

        assert_eq!(
            compressed[..2],
            GZIP_MAGIC,
            "the sniff has something to see"
        );
        assert_ne!(compressed, plain);
        assert_eq!(
            read_idx(&compressed).expect("gzipped"),
            read_idx(&plain).expect("plain"),
            "the container must not change the array"
        );
    }

    #[test]
    fn read_idx_accepts_an_uncompressed_file_too() {
        let plain = idx_file(0x08, &[2], &[1, 2]);
        assert_eq!(read_idx(&plain).expect("plain").data(), &[1, 2]);
    }

    #[test]
    fn a_corrupt_gzip_stream_is_reported_as_such() {
        let mut broken = gzipped(&idx_file(0x08, &[4], &[1, 2, 3, 4]));
        let last = broken.len() - 1;
        broken[last] ^= 0xFF; // clobber the trailing CRC/length

        match read_idx(&broken) {
            Err(IdxError::Gzip(message)) => assert!(!message.is_empty()),
            other => panic!("expected a gzip error, got {other:?}"),
        }
    }

    #[test]
    fn a_gzipped_file_that_is_not_idx_reports_the_inner_problem() {
        let compressed = gzipped(b"not an idx file at all");
        assert!(matches!(
            read_idx(&compressed),
            Err(IdxError::NotIdx { .. })
        ));
    }

    // -- rejections --------------------------------------------------------

    #[test]
    fn a_short_file_is_truncated_not_misread() {
        assert_eq!(
            decode_idx(&[0, 0, 8]),
            Err(IdxError::Truncated {
                needed: 4,
                found: 3
            })
        );
        // The rank says three dimensions, so twelve more bytes are needed.
        assert_eq!(
            decode_idx(&[0, 0, 8, 3, 0, 0]),
            Err(IdxError::Truncated {
                needed: 16,
                found: 6
            })
        );
    }

    #[test]
    fn a_wrong_magic_is_rejected() {
        assert_eq!(
            decode_idx(&[0x50, 0x4b, 8, 1, 0, 0, 0, 0]),
            Err(IdxError::NotIdx {
                magic: [0x50, 0x4b]
            }),
            "a zip file must not be mistaken for a rank-1 IDX"
        );
    }

    #[test]
    fn an_unsupported_element_type_is_named() {
        // 0x0D is `float`, a legal IDX type this reader does not decode.
        assert_eq!(
            decode_idx(&idx_file(0x0D, &[1], &[0, 0, 0, 0])),
            Err(IdxError::UnsupportedElementType { code: 0x0D })
        );
    }

    #[test]
    fn a_payload_that_does_not_match_the_shape_is_rejected() {
        assert_eq!(
            decode_idx(&idx_file(0x08, &[5], &[1, 2, 3])),
            Err(IdxError::PayloadLength {
                expected: 5,
                found: 3
            })
        );
        assert_eq!(
            decode_idx(&idx_file(0x08, &[2], &[1, 2, 3])),
            Err(IdxError::PayloadLength {
                expected: 2,
                found: 3
            }),
            "trailing bytes are as wrong as missing ones"
        );
    }

    #[test]
    fn a_shape_that_overflows_is_rejected_rather_than_wrapping() {
        // Four dimensions of 2^32 - 1 multiply past 2^64.
        let mut bytes = vec![0x00, 0x00, 0x08, 4];
        for _ in 0..4 {
            bytes.extend_from_slice(&u32::MAX.to_be_bytes());
        }
        assert_eq!(decode_idx(&bytes), Err(IdxError::ShapeOverflow));
    }

    #[test]
    fn every_error_renders_something_a_reader_can_act_on() {
        let errors = [
            IdxError::Truncated {
                needed: 16,
                found: 4,
            },
            IdxError::NotIdx {
                magic: [0x1f, 0x8b],
            },
            IdxError::UnsupportedElementType { code: 0x0D },
            IdxError::PayloadLength {
                expected: 10,
                found: 9,
            },
            IdxError::ShapeOverflow,
            IdxError::Gzip("invalid deflate stream".to_owned()),
        ];
        for error in &errors {
            let rendered = error.to_string();
            assert!(rendered.starts_with("IDX: "), "{rendered}");
            let _: &dyn Error = error;
        }
    }
}
