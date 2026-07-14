//! Port of Kyty's `Math::vec2/vec3/vec4` and `mat2/mat3/mat4`
//! (`reference/kyty/source/include/Kyty/Math/VectorAndMatrix.h`).
//!
//! Kyty's vector/matrix classes are ordinary float column-vectors / matrices
//! with the usual arithmetic, dot/cross, length/normalize, and matrix
//! multiply. Rather than transliterate ~3k lines of C++ templates, this
//! module aliases them to [`glam`]'s SIMD-accelerated equivalents (the
//! workspace-crate convention) and re-exposes the handful of Kyty
//! construction spellings whose names differ.
//!
//! | Kyty | Rust (glam) |
//! |------|-------------|
//! | `vec2`/`vec3`/`vec4` | [`Vec2`]/[`Vec3`]/[`Vec4`] |
//! | `mat2`/`mat3`/`mat4` | [`Mat2`]/[`Mat3`]/[`Mat4`] |
//! | `vec4(s)` (scalar splat) | [`splat4`] → `Vec4::splat(s)` |
//! | `vec4(vec3, w)` | [`vec3_w`] → `(v, w).into()` |
//! | component access `.x/.y/.z/.w`, `.r/.g/.b/.a` | `.x/.y/.z/.w` (glam) |
//! | `Dot`/`Cross`/`Length`/`Normalize` | `.dot()`/`.cross()`/`.length()`/`.normalize()` |
//! | `mat4 * mat4`, `mat4 * vec4` | `*` (glam operators) |
//!
//! glam's element order is column-major, matching the GLSL/HLSL/GNM
//! convention Kyty's `mat4` targets — the natural fit for the later Graphics
//! port that will consume this.

pub use glam::{Mat2, Mat3, Mat4, Vec2, Vec3, Vec4};

/// Kyty `vec2`.
pub type Vec2f = Vec2;
/// Kyty `vec3`.
pub type Vec3f = Vec3;
/// Kyty `vec4`.
pub type Vec4f = Vec4;
/// Kyty `mat2`.
pub type Mat2f = Mat2;
/// Kyty `mat3`.
pub type Mat3f = Mat3;
/// Kyty `mat4`.
pub type Mat4f = Mat4;

/// Kyty `explicit vec4(float s)` — splat a scalar across all four lanes.
#[must_use]
pub fn splat4(s: f32) -> Vec4 {
    Vec4::splat(s)
}

/// Kyty `explicit vec3(float s)`.
#[must_use]
pub fn splat3(s: f32) -> Vec3 {
    Vec3::splat(s)
}

/// Kyty `explicit vec4(const vec3& v, float s4)` — extend a `vec3` with a w.
#[must_use]
pub fn vec3_w(v: Vec3, w: f32) -> Vec4 {
    (v, w).into()
}

/// The `mat4` identity — Kyty's default-constructed identity matrix.
#[must_use]
pub fn mat4_identity() -> Mat4 {
    Mat4::IDENTITY
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_construction_and_dot_cross() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(a.dot(b), 0.0);
        assert_eq!(a.cross(b), Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(splat3(2.0), Vec3::new(2.0, 2.0, 2.0));
    }

    #[test]
    fn vec4_splat_and_extend() {
        assert_eq!(splat4(1.5), Vec4::new(1.5, 1.5, 1.5, 1.5));
        assert_eq!(
            vec3_w(Vec3::new(1.0, 2.0, 3.0), 4.0),
            Vec4::new(1.0, 2.0, 3.0, 4.0)
        );
    }

    #[test]
    fn vec_length_and_normalize() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert_eq!(v.length(), 5.0);
        assert!((v.normalize().length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mat4_identity_and_multiply() {
        let id = mat4_identity();
        let v = Vec4::new(1.0, 2.0, 3.0, 1.0);
        // Identity leaves a vector unchanged.
        assert_eq!(id * v, v);
        // A translation matrix moves a point.
        let t = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0));
        assert_eq!((t * v).x, 11.0);
    }
}
