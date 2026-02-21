//! Agent representation — position, velocity, mass, and movement limits.

use crate::vec::Vec;

/// An autonomous agent with position, velocity, and movement constraints.
#[derive(Copy, Clone, Debug)]
pub struct Agent<V: Vec> {
    pub position: V,
    pub velocity: V,
    pub mass: V::Scalar,
    pub max_speed: V::Scalar,
    pub max_force: V::Scalar,
}

impl<V: Vec> Agent<V> {
    /// Creates a new agent with the given parameters.
    pub fn new(
        position: V,
        velocity: V,
        mass: V::Scalar,
        max_speed: V::Scalar,
        max_force: V::Scalar,
    ) -> Self {
        Self {
            position,
            velocity,
            mass,
            max_speed,
            max_force,
        }
    }

    /// Unit vector in the direction of travel. Returns zero if stationary.
    pub fn heading(&self) -> V {
        self.velocity.normalize_or_zero()
    }

    /// Current speed (magnitude of velocity).
    pub fn speed(&self) -> V::Scalar {
        self.velocity.length()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec::Vec2;

    #[test]
    fn heading_returns_normalized_velocity() {
        let agent = Agent::new(
            Vec2::new(0.0f32, 0.0),
            Vec2::new(3.0, 4.0),
            1.0,
            10.0,
            5.0,
        );
        let h = agent.heading();
        // Should be unit length
        let len = Vec::length(h);
        assert!((len - 1.0f32).abs() < 1e-5);
        // Direction should be (0.6, 0.8)
        assert!((h.x - 0.6f32).abs() < 1e-5);
        assert!((h.y - 0.8f32).abs() < 1e-5);
    }

    #[test]
    fn heading_returns_zero_when_stationary() {
        let agent = Agent::new(
            Vec2::new(5.0f32, 5.0),
            Vec2::zero(),
            1.0,
            10.0,
            5.0,
        );
        let h = agent.heading();
        assert!((h.x).abs() < 1e-6);
        assert!((h.y).abs() < 1e-6);
    }

    #[test]
    fn speed_returns_correct_magnitude() {
        let agent = Agent::new(
            Vec2::new(0.0f32, 0.0),
            Vec2::new(3.0, 4.0),
            1.0,
            10.0,
            5.0,
        );
        assert!((agent.speed() - 5.0f32).abs() < 1e-5);
    }

    #[test]
    fn speed_zero_when_stationary() {
        let agent = Agent::new(
            Vec2::new(0.0f32, 0.0),
            Vec2::zero(),
            1.0,
            10.0,
            5.0,
        );
        assert!((agent.speed()).abs() < 1e-6);
    }

    #[test]
    fn new_sets_all_fields() {
        let pos = Vec2::new(1.0f32, 2.0);
        let vel = Vec2::new(3.0f32, 4.0);
        let agent = Agent::new(pos, vel, 2.0, 15.0, 8.0);
        assert_eq!(agent.position.x, 1.0);
        assert_eq!(agent.position.y, 2.0);
        assert_eq!(agent.velocity.x, 3.0);
        assert_eq!(agent.velocity.y, 4.0);
        assert_eq!(agent.mass, 2.0);
        assert_eq!(agent.max_speed, 15.0);
        assert_eq!(agent.max_force, 8.0);
    }
}
