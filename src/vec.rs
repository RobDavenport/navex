//! Vector types (Vec2, Vec3) and the Vec trait for dimension-agnostic steering.

use crate::float::Float;

/// Trait for vector types used in steering calculations.
pub trait Vec: Copy + Clone + Default + core::fmt::Debug {
    type Scalar: Float;

    fn zero() -> Self;
    fn splat(s: Self::Scalar) -> Self;
    fn dot(self, other: Self) -> Self::Scalar;
    fn length_sq(self) -> Self::Scalar;
    fn scale(self, s: Self::Scalar) -> Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn component_mul(self, other: Self) -> Self;
    fn neg(self) -> Self;

    /// Returns the Euclidean length of this vector.
    #[inline]
    fn length(self) -> Self::Scalar {
        self.length_sq().sqrt()
    }

    /// Normalizes this vector to unit length, or returns zero if the length
    /// is below epsilon.
    #[inline]
    fn normalize_or_zero(self) -> Self {
        let len = self.length();
        if len > Self::Scalar::epsilon() {
            self.scale(len.recip())
        } else {
            Self::zero()
        }
    }

    /// Normalizes this vector to unit length. Panics if the vector is zero.
    #[inline]
    fn normalize(self) -> Self {
        let len = self.length();
        self.scale(len.recip())
    }

    /// Truncates this vector to at most `max_length`.
    #[inline]
    fn truncate(self, max_length: Self::Scalar) -> Self {
        let len_sq = self.length_sq();
        if len_sq > max_length * max_length {
            self.normalize_or_zero().scale(max_length)
        } else {
            self
        }
    }

    /// Returns the Euclidean distance between this vector and another.
    #[inline]
    fn distance(self, other: Self) -> Self::Scalar {
        self.sub(other).length()
    }

    /// Returns the squared Euclidean distance between this vector and another.
    #[inline]
    fn distance_sq(self, other: Self) -> Self::Scalar {
        self.sub(other).length_sq()
    }

    /// Linearly interpolates between this vector and another by parameter `t`.
    #[inline]
    fn lerp(self, other: Self, t: Self::Scalar) -> Self {
        self.add(other.sub(self).scale(t))
    }
}

// ---------------------------------------------------------------------------
// Vec2
// ---------------------------------------------------------------------------

/// 2D vector generic over float type.
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct Vec2<F: Float> {
    pub x: F,
    pub y: F,
}

impl<F: Float> Vec2<F> {
    /// Creates a new 2D vector.
    #[inline]
    pub fn new(x: F, y: F) -> Self {
        Self { x, y }
    }

    /// Returns the perpendicular vector (90 degrees counter-clockwise).
    #[inline]
    pub fn perp(self) -> Self {
        Self {
            x: -self.y,
            y: self.x,
        }
    }

    /// Returns the angle of this vector in radians (atan2(y, x)).
    #[inline]
    pub fn angle(self) -> F {
        self.y.atan2(self.x)
    }

    /// Creates a unit vector from an angle in radians.
    #[inline]
    pub fn from_angle(radians: F) -> Self {
        Self {
            x: radians.cos(),
            y: radians.sin(),
        }
    }
}

impl<F: Float> Vec for Vec2<F> {
    type Scalar = F;

    #[inline]
    fn zero() -> Self {
        Self {
            x: F::zero(),
            y: F::zero(),
        }
    }

    #[inline]
    fn splat(s: F) -> Self {
        Self { x: s, y: s }
    }

    #[inline]
    fn dot(self, other: Self) -> F {
        self.x * other.x + self.y * other.y
    }

    #[inline]
    fn length_sq(self) -> F {
        self.dot(self)
    }

    #[inline]
    fn scale(self, s: F) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
        }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }

    #[inline]
    fn component_mul(self, other: Self) -> Self {
        Self {
            x: self.x * other.x,
            y: self.y * other.y,
        }
    }

    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

// ---------------------------------------------------------------------------
// Vec3
// ---------------------------------------------------------------------------

/// 3D vector generic over float type.
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct Vec3<F: Float> {
    pub x: F,
    pub y: F,
    pub z: F,
}

impl<F: Float> Vec3<F> {
    /// Creates a new 3D vector.
    #[inline]
    pub fn new(x: F, y: F, z: F) -> Self {
        Self { x, y, z }
    }

    /// Returns the cross product of this vector and another.
    #[inline]
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

impl<F: Float> Vec for Vec3<F> {
    type Scalar = F;

    #[inline]
    fn zero() -> Self {
        Self {
            x: F::zero(),
            y: F::zero(),
            z: F::zero(),
        }
    }

    #[inline]
    fn splat(s: F) -> Self {
        Self { x: s, y: s, z: s }
    }

    #[inline]
    fn dot(self, other: Self) -> F {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    fn length_sq(self) -> F {
        self.dot(self)
    }

    #[inline]
    fn scale(self, s: F) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    #[inline]
    fn component_mul(self, other: Self) -> Self {
        Self {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }

    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar (1D wrapper)
// ---------------------------------------------------------------------------

/// 1D scalar wrapper that implements the `Vec` trait.
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct Scalar<F: Float>(pub F);

impl<F: Float> Vec for Scalar<F> {
    type Scalar = F;

    #[inline]
    fn zero() -> Self {
        Scalar(F::zero())
    }

    #[inline]
    fn splat(s: F) -> Self {
        Scalar(s)
    }

    #[inline]
    fn dot(self, other: Self) -> F {
        self.0 * other.0
    }

    #[inline]
    fn length_sq(self) -> F {
        self.0 * self.0
    }

    #[inline]
    fn scale(self, s: F) -> Self {
        Scalar(self.0 * s)
    }

    #[inline]
    fn add(self, other: Self) -> Self {
        Scalar(self.0 + other.0)
    }

    #[inline]
    fn sub(self, other: Self) -> Self {
        Scalar(self.0 - other.0)
    }

    #[inline]
    fn component_mul(self, other: Self) -> Self {
        Scalar(self.0 * other.0)
    }

    #[inline]
    fn neg(self) -> Self {
        Scalar(-self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Vec2 tests
    // -----------------------------------------------------------------------

    #[test]
    fn vec2_new() {
        let v = Vec2::new(1.0f32, 2.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
    }

    #[test]
    fn vec2_zero() {
        let v = Vec2::<f32>::zero();
        assert_eq!(v.x, 0.0);
        assert_eq!(v.y, 0.0);
    }

    #[test]
    fn vec2_splat() {
        let v = Vec2::<f32>::splat(3.0);
        assert_eq!(v.x, 3.0);
        assert_eq!(v.y, 3.0);
    }

    #[test]
    fn vec2_dot() {
        let a = Vec2::new(1.0f32, 2.0);
        let b = Vec2::new(3.0, 4.0);
        assert!((Vec::dot(a, b) - 11.0f32).abs() < 1e-6);
    }

    #[test]
    fn vec2_length() {
        let v = Vec2::new(3.0f32, 4.0);
        assert!((Vec::length(v) - 5.0f32).abs() < 1e-6);
    }

    #[test]
    fn vec2_normalize() {
        let v = Vec2::new(3.0f32, 4.0);
        let n = Vec::normalize(v);
        assert!((Vec::length(n) - 1.0f32).abs() < 1e-5);
        assert!((n.x - 0.6f32).abs() < 1e-5);
        assert!((n.y - 0.8f32).abs() < 1e-5);
    }

    #[test]
    fn vec2_normalize_or_zero_nonzero() {
        let v = Vec2::new(0.0f32, 5.0);
        let n = Vec::normalize_or_zero(v);
        assert!((n.x).abs() < 1e-6);
        assert!((n.y - 1.0f32).abs() < 1e-5);
    }

    #[test]
    fn vec2_normalize_or_zero_zero() {
        let v = Vec2::<f32>::zero();
        let n = Vec::normalize_or_zero(v);
        assert_eq!(n.x, 0.0);
        assert_eq!(n.y, 0.0);
    }

    #[test]
    fn vec2_truncate_within() {
        let v = Vec2::new(1.0f32, 0.0);
        let t = Vec::truncate(v, 5.0);
        assert!((t.x - 1.0f32).abs() < 1e-6);
        assert!((t.y).abs() < 1e-6);
    }

    #[test]
    fn vec2_truncate_exceeds() {
        let v = Vec2::new(6.0f32, 8.0);
        let t = Vec::truncate(v, 5.0);
        assert!((Vec::length(t) - 5.0f32).abs() < 1e-4);
    }

    #[test]
    fn vec2_distance() {
        let a = Vec2::new(1.0f32, 0.0);
        let b = Vec2::new(4.0, 0.0);
        assert!((Vec::distance(a, b) - 3.0f32).abs() < 1e-6);
    }

    #[test]
    fn vec2_distance_sq() {
        let a = Vec2::new(1.0f32, 0.0);
        let b = Vec2::new(4.0, 0.0);
        assert!((Vec::distance_sq(a, b) - 9.0f32).abs() < 1e-6);
    }

    #[test]
    fn vec2_lerp() {
        let a = Vec2::new(0.0f32, 0.0);
        let b = Vec2::new(10.0, 10.0);
        let mid = Vec::lerp(a, b, 0.5);
        assert!((mid.x - 5.0f32).abs() < 1e-6);
        assert!((mid.y - 5.0f32).abs() < 1e-6);
    }

    #[test]
    fn vec2_perp() {
        let v = Vec2::new(1.0f32, 0.0);
        let p = v.perp();
        assert!((p.x).abs() < 1e-6);
        assert!((p.y - 1.0f32).abs() < 1e-6);
    }

    #[test]
    fn vec2_angle() {
        let v = Vec2::new(1.0f32, 0.0);
        assert!((v.angle()).abs() < 1e-6);

        let v2 = Vec2::new(0.0f32, 1.0);
        assert!((v2.angle() - core::f32::consts::FRAC_PI_2).abs() < 1e-6);
    }

    #[test]
    fn vec2_from_angle() {
        let v = Vec2::<f32>::from_angle(0.0);
        assert!((v.x - 1.0f32).abs() < 1e-6);
        assert!((v.y).abs() < 1e-6);

        let v2 = Vec2::<f32>::from_angle(core::f32::consts::FRAC_PI_2);
        assert!((v2.x).abs() < 1e-5);
        assert!((v2.y - 1.0f32).abs() < 1e-5);
    }

    #[test]
    fn vec2_neg() {
        let v = Vec2::new(1.0f32, -2.0);
        let n = Vec::neg(v);
        assert_eq!(n.x, -1.0);
        assert_eq!(n.y, 2.0);
    }

    #[test]
    fn vec2_add_sub() {
        let a = Vec2::new(1.0f32, 2.0);
        let b = Vec2::new(3.0, 4.0);
        let sum = Vec::add(a, b);
        assert_eq!(sum.x, 4.0);
        assert_eq!(sum.y, 6.0);
        let diff = Vec::sub(a, b);
        assert_eq!(diff.x, -2.0);
        assert_eq!(diff.y, -2.0);
    }

    #[test]
    fn vec2_scale() {
        let v = Vec2::new(2.0f32, 3.0);
        let s = Vec::scale(v, 2.0);
        assert_eq!(s.x, 4.0);
        assert_eq!(s.y, 6.0);
    }

    #[test]
    fn vec2_component_mul() {
        let a = Vec2::new(2.0f32, 3.0);
        let b = Vec2::new(4.0, 5.0);
        let r = Vec::component_mul(a, b);
        assert_eq!(r.x, 8.0);
        assert_eq!(r.y, 15.0);
    }

    // -----------------------------------------------------------------------
    // Vec3 tests
    // -----------------------------------------------------------------------

    #[test]
    fn vec3_new() {
        let v = Vec3::new(1.0f32, 2.0, 3.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
    }

    #[test]
    fn vec3_zero() {
        let v = Vec3::<f32>::zero();
        assert_eq!(v.x, 0.0);
        assert_eq!(v.y, 0.0);
        assert_eq!(v.z, 0.0);
    }

    #[test]
    fn vec3_dot() {
        let a = Vec3::new(1.0f32, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert!((Vec::dot(a, b) - 32.0f32).abs() < 1e-6);
    }

    #[test]
    fn vec3_cross() {
        let x = Vec3::new(1.0f32, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        let z = x.cross(y);
        assert!((z.x).abs() < 1e-6);
        assert!((z.y).abs() < 1e-6);
        assert!((z.z - 1.0f32).abs() < 1e-6);
    }

    #[test]
    fn vec3_cross_anticommutative() {
        let a = Vec3::new(1.0f32, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let ab = a.cross(b);
        let ba = b.cross(a);
        assert!((ab.x + ba.x).abs() < 1e-6);
        assert!((ab.y + ba.y).abs() < 1e-6);
        assert!((ab.z + ba.z).abs() < 1e-6);
    }

    #[test]
    fn vec3_length() {
        let v = Vec3::new(1.0f32, 2.0, 2.0);
        assert!((Vec::length(v) - 3.0f32).abs() < 1e-5);
    }

    #[test]
    fn vec3_normalize() {
        let v = Vec3::new(0.0f32, 0.0, 5.0);
        let n = Vec::normalize(v);
        assert!((n.x).abs() < 1e-6);
        assert!((n.y).abs() < 1e-6);
        assert!((n.z - 1.0f32).abs() < 1e-5);
    }

    #[test]
    fn vec3_add_sub() {
        let a = Vec3::new(1.0f32, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let sum = Vec::add(a, b);
        assert_eq!(sum.x, 5.0);
        assert_eq!(sum.y, 7.0);
        assert_eq!(sum.z, 9.0);
    }

    #[test]
    fn vec3_neg() {
        let v = Vec3::new(1.0f32, -2.0, 3.0);
        let n = Vec::neg(v);
        assert_eq!(n.x, -1.0);
        assert_eq!(n.y, 2.0);
        assert_eq!(n.z, -3.0);
    }

    // -----------------------------------------------------------------------
    // Scalar tests
    // -----------------------------------------------------------------------

    #[test]
    fn scalar_basic() {
        let a = Scalar(3.0f32);
        let b = Scalar(4.0f32);
        let sum = Vec::add(a, b);
        assert_eq!(sum.0, 7.0);
        let diff = Vec::sub(a, b);
        assert_eq!(diff.0, -1.0);
    }

    #[test]
    fn scalar_dot() {
        let a = Scalar(3.0f32);
        let b = Scalar(4.0f32);
        assert!((Vec::dot(a, b) - 12.0f32).abs() < 1e-6);
    }

    #[test]
    fn scalar_length() {
        let v = Scalar(5.0f32);
        assert!((Vec::length(v) - 5.0f32).abs() < 1e-6);
    }

    #[test]
    fn scalar_length_negative() {
        let v = Scalar(-5.0f32);
        assert!((Vec::length(v) - 5.0f32).abs() < 1e-6);
    }

    #[test]
    fn scalar_normalize() {
        let v = Scalar(5.0f32);
        let n = Vec::normalize(v);
        assert!((n.0 - 1.0f32).abs() < 1e-6);
    }

    #[test]
    fn scalar_neg() {
        let v = Scalar(3.0f32);
        let n = Vec::neg(v);
        assert_eq!(n.0, -3.0);
    }

    #[test]
    fn scalar_zero() {
        let v = Scalar::<f32>::zero();
        assert_eq!(v.0, 0.0);
    }

    #[test]
    fn scalar_splat() {
        let v = Scalar::<f32>::splat(7.0);
        assert_eq!(v.0, 7.0);
    }

    #[test]
    fn scalar_scale() {
        let v = Scalar(2.0f32);
        let s = Vec::scale(v, 3.0);
        assert_eq!(s.0, 6.0);
    }

    #[test]
    fn scalar_lerp() {
        let a = Scalar(0.0f32);
        let b = Scalar(10.0f32);
        let mid = Vec::lerp(a, b, 0.5);
        assert!((mid.0 - 5.0f32).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // f64 variants
    // -----------------------------------------------------------------------

    #[test]
    fn vec2_f64() {
        let v = Vec2::new(3.0f64, 4.0);
        assert!((Vec::length(v) - 5.0f64).abs() < 1e-10);
    }

    #[test]
    fn vec3_f64() {
        let v = Vec3::new(1.0f64, 2.0, 2.0);
        assert!((Vec::length(v) - 3.0f64).abs() < 1e-10);
    }

    #[test]
    fn scalar_f64() {
        let v = Scalar(5.0f64);
        assert!((Vec::length(v) - 5.0f64).abs() < 1e-10);
    }
}
