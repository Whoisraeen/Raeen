//! The decryption boundary.
//!
//! XPS5X ships **no keys** and no key-extraction tooling. All decryption is
//! driven by a user-supplied [`KeyProvider`]; the default [`NoKeysProvider`]
//! returns nothing and decryption fails cleanly with
//! [`FirmwareError::MissingKey`]. This module consumes keys — it never
//! derives, guesses, brute-forces, or extracts them.

use xps5x_core::error::FirmwareError;

/// Identifies which key a SELF/module segment needs, read from its metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyRequest {
    /// Key type from the SELF/segment header.
    pub key_type: u32,
    /// Key id / seed identifying the specific key.
    pub key_id: u64,
}

/// A content key + IV supplied by a user [`KeyProvider`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentKey {
    /// 128-bit content key.
    pub key: [u8; 16],
    /// 128-bit initialization vector.
    pub iv: [u8; 16],
}

/// User-supplied source of decryption keys.
///
/// Implementors load keys the user obtained from hardware they own. The
/// default [`NoKeysProvider`] supplies none.
pub trait KeyProvider: Send + Sync {
    /// Return the key for `req`, or `None` if unavailable.
    fn segment_key(&self, req: &KeyRequest) -> Option<SegmentKey>;
}

/// Default provider that holds no keys. Decryption through it always fails
/// cleanly with [`FirmwareError::MissingKey`].
pub struct NoKeysProvider;

impl KeyProvider for NoKeysProvider {
    fn segment_key(&self, _req: &KeyRequest) -> Option<SegmentKey> {
        None
    }
}

/// Resolve a key or produce the canonical [`FirmwareError::MissingKey`].
///
/// Callers should treat the error as a normal, expected condition (log at
/// `info`, not `error`).
pub fn require_key(
    provider: &dyn KeyProvider,
    req: &KeyRequest,
) -> Result<SegmentKey, FirmwareError> {
    provider
        .segment_key(req)
        .ok_or(FirmwareError::MissingKey { key_id: req.key_id })
}

#[cfg(test)]
mod tests {
    use super::*;
    use xps5x_core::error::FirmwareError;

    #[test]
    fn no_keys_provider_returns_none() {
        let req = KeyRequest { key_type: 1, key_id: 0xABCD };
        assert_eq!(NoKeysProvider.segment_key(&req), None);
    }

    #[test]
    fn require_key_maps_missing_to_error() {
        let req = KeyRequest { key_type: 1, key_id: 0xABCD };
        let err = require_key(&NoKeysProvider, &req).unwrap_err();
        assert!(matches!(err, FirmwareError::MissingKey { key_id: 0xABCD }));
    }

    #[test]
    fn require_key_returns_supplied_key() {
        struct FixedProvider;
        impl KeyProvider for FixedProvider {
            fn segment_key(&self, _req: &KeyRequest) -> Option<SegmentKey> {
                Some(SegmentKey { key: [1u8; 16], iv: [2u8; 16] })
            }
        }
        let req = KeyRequest { key_type: 0, key_id: 7 };
        let key = require_key(&FixedProvider, &req).unwrap();
        assert_eq!(key.key, [1u8; 16]);
        assert_eq!(key.iv, [2u8; 16]);
    }
}
