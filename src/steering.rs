//! SteeringOutput and force application.

use crate::agent::Agent;
use crate::float::Float;
use crate::vec::Vec;

/// The output of a steering behavior: a linear force to apply.
#[derive(Copy, Clone, Debug)]
pub struct SteeringOutput<V: Vec> {
    pub linear: V,
}

impl<V: Vec> SteeringOutput<V> {
    /// Creates a zero steering output (no force).
    pub fn zero() -> Self {
        Self { linear: V::zero() }
    }

    /// Creates a steering output with the given linear force.
    pub fn new(linear: V) -> Self {
        Self { linear }
    }

    /// Truncates the linear force to at most `max` magnitude.
    pub fn truncate(self, max: V::Scalar) -> Self {
        Self {
            linear: self.linear.truncate(max),
        }
    }

    /// Scales the linear force by `s`.
    pub fn scale(self, s: V::Scalar) -> Self {
        Self {
            linear: self.linear.scale(s),
        }
    }

    /// Adds another steering output to this one.
    pub fn add(self, other: Self) -> Self {
        Self {
            linear: self.linear.add(other.linear),
        }
    }

    /// Returns true if the force magnitude is below `eps`.
    pub fn is_zero(self, eps: V::Scalar) -> bool {
        self.linear.length_sq() < eps * eps
    }
}

/// Apply a steering force to an agent, returning the updated agent.
///
/// The integration follows: force/mass -> acceleration -> velocity -> position.
/// Velocity is clamped to `agent.max_speed` and force is clamped to `agent.max_force`.
pub fn apply_steering<V: Vec>(
    agent: &Agent<V>,
    steering: &SteeringOutput<V>,
    dt: V::Scalar,
) -> Agent<V> {
    let force = steering.linear.truncate(agent.max_force);
    let acceleration = force.scale(agent.mass.recip());
    let new_velocity = agent.velocity.add(acceleration.scale(dt));
    let new_velocity = new_velocity.truncate(agent.max_speed);
    let new_position = agent.position.add(new_velocity.scale(dt));

    Agent {
        position: new_position,
        velocity: new_velocity,
        mass: agent.mass,
        max_speed: agent.max_speed,
        max_force: agent.max_force,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec::Vec2;

    #[test]
    fn steering_output_zero() {
        let s = SteeringOutput::<Vec2<f32>>::zero();
        assert_eq!(s.linear.x, 0.0);
        assert_eq!(s.linear.y, 0.0);
    }

    #[test]
    fn steering_output_new() {
        let s = SteeringOutput::new(Vec2::new(3.0f32, 4.0));
        assert_eq!(s.linear.x, 3.0);
        assert_eq!(s.linear.y, 4.0);
    }

    #[test]
    fn steering_output_truncate() {
        let s = SteeringOutput::new(Vec2::new(6.0f32, 8.0));
        let t = s.truncate(5.0);
        let len = Vec::length(t.linear);
        assert!((len - 5.0f32).abs() < 1e-4);
    }

    #[test]
    fn steering_output_truncate_within() {
        let s = SteeringOutput::new(Vec2::new(1.0f32, 0.0));
        let t = s.truncate(5.0);
        assert!((t.linear.x - 1.0f32).abs() < 1e-6);
        assert!((t.linear.y).abs() < 1e-6);
    }

    #[test]
    fn steering_output_scale() {
        let s = SteeringOutput::new(Vec2::new(2.0f32, 3.0));
        let scaled = s.scale(2.0);
        assert!((scaled.linear.x - 4.0f32).abs() < 1e-6);
        assert!((scaled.linear.y - 6.0f32).abs() < 1e-6);
    }

    #[test]
    fn steering_output_add() {
        let a = SteeringOutput::new(Vec2::new(1.0f32, 2.0));
        let b = SteeringOutput::new(Vec2::new(3.0f32, 4.0));
        let sum = a.add(b);
        assert!((sum.linear.x - 4.0f32).abs() < 1e-6);
        assert!((sum.linear.y - 6.0f32).abs() < 1e-6);
    }

    #[test]
    fn steering_output_is_zero_true() {
        let s = SteeringOutput::<Vec2<f32>>::zero();
        assert!(s.is_zero(0.001));
    }

    #[test]
    fn steering_output_is_zero_false() {
        let s = SteeringOutput::new(Vec2::new(1.0f32, 0.0));
        assert!(!s.is_zero(0.001));
    }

    #[test]
    fn apply_steering_basic() {
        // Agent at origin, stationary, mass=1, max_speed=10, max_force=5
        let agent = Agent::new(
            Vec2::new(0.0f32, 0.0),
            Vec2::new(0.0, 0.0),
            1.0,
            10.0,
            5.0,
        );
        // Apply force (1, 0) for 1 second
        let steering = SteeringOutput::new(Vec2::new(1.0f32, 0.0));
        let result = apply_steering(&agent, &steering, 1.0);

        // acceleration = (1, 0) / 1 = (1, 0)
        // new_velocity = (0, 0) + (1, 0) * 1 = (1, 0)
        // new_position = (0, 0) + (1, 0) * 1 = (1, 0)
        assert!((result.velocity.x - 1.0f32).abs() < 1e-5);
        assert!((result.velocity.y).abs() < 1e-5);
        assert!((result.position.x - 1.0f32).abs() < 1e-5);
        assert!((result.position.y).abs() < 1e-5);
    }

    #[test]
    fn apply_steering_velocity_clamped_to_max_speed() {
        // Agent with max_speed=2
        let agent = Agent::new(
            Vec2::new(0.0f32, 0.0),
            Vec2::new(0.0, 0.0),
            1.0,
            2.0,
            100.0,
        );
        // Apply large force (100, 0) for 1 second
        let steering = SteeringOutput::new(Vec2::new(100.0f32, 0.0));
        let result = apply_steering(&agent, &steering, 1.0);

        // Velocity should be clamped to max_speed = 2
        let speed = Vec::length(result.velocity);
        assert!((speed - 2.0f32).abs() < 1e-4);
    }

    #[test]
    fn apply_steering_force_clamped_to_max_force() {
        // Agent with max_force=1
        let agent = Agent::new(
            Vec2::new(0.0f32, 0.0),
            Vec2::new(0.0, 0.0),
            1.0,
            100.0,
            1.0,
        );
        // Apply force (10, 0) — should be truncated to (1, 0)
        let steering = SteeringOutput::new(Vec2::new(10.0f32, 0.0));
        let result = apply_steering(&agent, &steering, 1.0);

        // acceleration = (1, 0) / 1 = (1, 0), velocity = (1, 0)
        assert!((result.velocity.x - 1.0f32).abs() < 1e-5);
        assert!((result.velocity.y).abs() < 1e-5);
    }

    #[test]
    fn apply_steering_with_mass() {
        // Agent with mass=2
        let agent = Agent::new(
            Vec2::new(0.0f32, 0.0),
            Vec2::new(0.0, 0.0),
            2.0,
            10.0,
            10.0,
        );
        // Apply force (4, 0) for 1 second
        let steering = SteeringOutput::new(Vec2::new(4.0f32, 0.0));
        let result = apply_steering(&agent, &steering, 1.0);

        // acceleration = (4, 0) / 2 = (2, 0)
        // velocity = (2, 0)
        assert!((result.velocity.x - 2.0f32).abs() < 1e-5);
        assert!((result.velocity.y).abs() < 1e-5);
    }

    #[test]
    fn apply_steering_preserves_mass_and_limits() {
        let agent = Agent::new(
            Vec2::new(0.0f32, 0.0),
            Vec2::new(0.0, 0.0),
            3.0,
            7.0,
            5.0,
        );
        let steering = SteeringOutput::new(Vec2::new(1.0f32, 0.0));
        let result = apply_steering(&agent, &steering, 0.1);

        assert_eq!(result.mass, 3.0);
        assert_eq!(result.max_speed, 7.0);
        assert_eq!(result.max_force, 5.0);
    }
}
