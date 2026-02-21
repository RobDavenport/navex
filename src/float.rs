//! Float trait abstracting f32/f64 for generic numeric computation.

use core::ops::{Add, Div, Mul, Neg, Sub};

/// Trait abstracting floating-point arithmetic for f32/f64 generics.
pub trait Float:
    Copy
    + Clone
    + PartialEq
    + PartialOrd
    + Default
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + core::fmt::Debug
{
    fn zero() -> Self;
    fn one() -> Self;
    fn half() -> Self;
    fn two() -> Self;
    fn pi() -> Self;
    fn epsilon() -> Self;
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn atan2(self, other: Self) -> Self;
    fn abs(self) -> Self;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn from_f32(v: f32) -> Self;
    fn to_f32(self) -> f32;
    fn recip(self) -> Self;
    fn clamp(self, min: Self, max: Self) -> Self;
}

impl Float for f32 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn one() -> Self {
        1.0
    }
    #[inline]
    fn half() -> Self {
        0.5
    }
    #[inline]
    fn two() -> Self {
        2.0
    }
    #[inline]
    fn pi() -> Self {
        core::f32::consts::PI
    }
    #[inline]
    fn epsilon() -> Self {
        f32::EPSILON
    }
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrtf(self)
    }
    #[inline]
    fn sin(self) -> Self {
        libm::sinf(self)
    }
    #[inline]
    fn cos(self) -> Self {
        libm::cosf(self)
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        libm::atan2f(self, other)
    }
    #[inline]
    fn abs(self) -> Self {
        libm::fabsf(self)
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        if self <= other { self } else { other }
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        if self >= other { self } else { other }
    }
    #[inline]
    fn from_f32(v: f32) -> Self {
        v
    }
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline]
    fn recip(self) -> Self {
        1.0 / self
    }
    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        Float::max(Float::min(self, max), min)
    }
}

impl Float for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn one() -> Self {
        1.0
    }
    #[inline]
    fn half() -> Self {
        0.5
    }
    #[inline]
    fn two() -> Self {
        2.0
    }
    #[inline]
    fn pi() -> Self {
        core::f64::consts::PI
    }
    #[inline]
    fn epsilon() -> Self {
        f64::EPSILON
    }
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
    #[inline]
    fn sin(self) -> Self {
        libm::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        libm::cos(self)
    }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        libm::atan2(self, other)
    }
    #[inline]
    fn abs(self) -> Self {
        libm::fabs(self)
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        if self <= other { self } else { other }
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        if self >= other { self } else { other }
    }
    #[inline]
    fn from_f32(v: f32) -> Self {
        v as f64
    }
    #[inline]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline]
    fn recip(self) -> Self {
        1.0 / self
    }
    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        Float::max(Float::min(self, max), min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_zero_one() {
        assert_eq!(f32::zero(), 0.0f32);
        assert_eq!(f32::one(), 1.0f32);
        assert_eq!(f32::half(), 0.5f32);
        assert_eq!(f32::two(), 2.0f32);
    }

    #[test]
    fn f64_zero_one() {
        assert_eq!(f64::zero(), 0.0f64);
        assert_eq!(f64::one(), 1.0f64);
        assert_eq!(f64::half(), 0.5f64);
        assert_eq!(f64::two(), 2.0f64);
    }

    #[test]
    fn f32_sqrt() {
        let v: f32 = Float::from_f32(4.0);
        assert!((Float::sqrt(v) - 2.0f32).abs() < f32::epsilon());
    }

    #[test]
    fn f64_sqrt() {
        let v: f64 = Float::from_f32(9.0);
        assert!((Float::sqrt(v) - 3.0f64).abs() < f64::epsilon());
    }

    #[test]
    fn f32_sin_cos() {
        let zero: f32 = Float::zero();
        assert!((Float::sin(zero)).abs() < 1e-6);
        assert!((Float::cos(zero) - 1.0f32).abs() < 1e-6);
    }

    #[test]
    fn f64_sin_cos() {
        let zero: f64 = Float::zero();
        assert!((Float::sin(zero)).abs() < 1e-12);
        assert!((Float::cos(zero) - 1.0f64).abs() < 1e-12);
    }

    #[test]
    fn f32_epsilon() {
        assert!(f32::epsilon() > 0.0f32);
        assert!(f32::epsilon() < 0.001f32);
    }

    #[test]
    fn f32_from_f32() {
        let v: f32 = Float::from_f32(3.14);
        assert!((v - 3.14f32).abs() < 1e-6);
    }

    #[test]
    fn f64_from_f32() {
        let v: f64 = Float::from_f32(3.14);
        assert!((v - 3.14f64).abs() < 0.001);
    }

    #[test]
    fn f32_min_max() {
        assert_eq!(Float::min(1.0f32, 2.0f32), 1.0f32);
        assert_eq!(Float::max(1.0f32, 2.0f32), 2.0f32);
    }

    #[test]
    fn f32_clamp() {
        assert_eq!(Float::clamp(0.5f32, 0.0f32, 1.0f32), 0.5f32);
        assert_eq!(Float::clamp(-1.0f32, 0.0f32, 1.0f32), 0.0f32);
        assert_eq!(Float::clamp(2.0f32, 0.0f32, 1.0f32), 1.0f32);
    }

    #[test]
    fn f32_recip() {
        assert!((Float::recip(2.0f32) - 0.5f32).abs() < 1e-6);
    }

    #[test]
    fn f32_abs() {
        assert_eq!(Float::abs(-3.0f32), 3.0f32);
        assert_eq!(Float::abs(3.0f32), 3.0f32);
    }

    #[test]
    fn f32_atan2() {
        let y: f32 = Float::one();
        let x: f32 = Float::zero();
        let result = Float::atan2(y, x);
        assert!((result - core::f32::consts::FRAC_PI_2).abs() < 1e-6);
    }

    #[test]
    fn f32_pi() {
        assert!((f32::pi() - 3.14159f32).abs() < 0.001);
    }

    #[test]
    fn f32_to_f32() {
        assert_eq!(Float::to_f32(2.5f32), 2.5f32);
    }

    #[test]
    fn f64_to_f32() {
        assert!((Float::to_f32(2.5f64) - 2.5f32).abs() < 1e-6);
    }
}
