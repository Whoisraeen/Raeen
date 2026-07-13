//! Port of Kyty's `include/Kyty/Core/MagicEnum.h`.
//!
//! The original header is a thin wrapper around the third-party
//! `magic_enum` C++ library, which uses compiler-specific `__PRETTY_FUNCTION__`
//! / `__FUNCSIG__` tricks to reflect enum variant names at compile time
//! without any manual per-enum boilerplate. Stable Rust has no equivalent
//! compile-time reflection over arbitrary `enum` types (that requires a
//! proc-macro `derive`, which is out of scope for this crate).
//!
//! Std/idiomatic mapping used here: a small `MagicEnum` trait that an enum
//! implements (typically via a short hand-written `match`, playing the role
//! that `magic_enum`'s compiler intrinsics played in C++) exposing:
//!   - `name(&self) -> &'static str`      (mirrors `magic_enum::enum_name`)
//!   - `from_name(&str) -> Option<Self>`  (mirrors `magic_enum::enum_cast`)
//!
//! On top of that trait we provide free functions with the same names/shape
//! as Kyty's API:
//!   - `enum_name(v)`  / `enum_name8(v)`  -> `EnumName` / `EnumName8`
//!     (Kyty distinguished UTF-16 `String` vs UTF-8 `String8`; Rust `String`
//!     is always UTF-8, so both collapse to the same `String` return type).
//!   - `enum_value(s, default)`           -> `EnumValue`
//!
//! `KYTY_ENUM_RANGE`, the macro used in C++ to widen `magic_enum`'s default
//! scan range for enums with values outside `[-128, 128)`, has no Rust
//! equivalent and is not needed: a hand-written `MagicEnum` impl already
//! covers whatever values the enum actually has, so there is no scan range
//! to configure.

/// Trait implemented by enums that support Kyty-style name/value reflection.
///
/// Implementations are expected to be a straightforward `match` over the
/// enum's variants, standing in for what `magic_enum`'s compile-time
/// reflection produced automatically in C++.
pub trait MagicEnum: Sized + Copy {
    /// Returns the variant's name, mirroring `magic_enum::enum_name`.
    fn name(&self) -> &'static str;

    /// Looks up a variant by name, mirroring `magic_enum::enum_cast`.
    fn from_name(name: &str) -> Option<Self>;
}

/// Port of `Kyty::Core::EnumName`.
pub fn enum_name<E: MagicEnum>(v: E) -> String {
    v.name().to_string()
}

/// Port of `Kyty::Core::EnumName8`.
///
/// Kyty kept a separate UTF-8 `String8` overload; Rust's `String` is
/// always UTF-8, so this is identical to [`enum_name`].
pub fn enum_name8<E: MagicEnum>(v: E) -> String {
    v.name().to_string()
}

/// Port of `Kyty::Core::EnumValue`.
pub fn enum_value<E: MagicEnum>(str: &str, default_value: E) -> E {
    E::from_name(str).unwrap_or(default_value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Color {
        Red,
        Green,
        Blue,
    }

    impl MagicEnum for Color {
        fn name(&self) -> &'static str {
            match self {
                Color::Red => "Red",
                Color::Green => "Green",
                Color::Blue => "Blue",
            }
        }

        fn from_name(name: &str) -> Option<Self> {
            match name {
                "Red" => Some(Color::Red),
                "Green" => Some(Color::Green),
                "Blue" => Some(Color::Blue),
                _ => None,
            }
        }
    }

    #[test]
    fn test_enum_name() {
        assert_eq!(enum_name(Color::Red), "Red");
        assert_eq!(enum_name(Color::Blue), "Blue");
    }

    #[test]
    fn test_enum_name8_matches_enum_name() {
        assert_eq!(enum_name8(Color::Green), enum_name(Color::Green));
    }

    #[test]
    fn test_enum_value_found() {
        assert_eq!(enum_value("Green", Color::Red), Color::Green);
    }

    #[test]
    fn test_enum_value_default_on_miss() {
        assert_eq!(enum_value("Purple", Color::Blue), Color::Blue);
    }

    #[test]
    fn test_enum_value_default_on_empty() {
        assert_eq!(enum_value("", Color::Red), Color::Red);
    }
}
