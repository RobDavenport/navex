//! Seek, flee, and arrive behaviors.

use crate::agent::Agent;
use crate::float::Float;
use crate::steering::SteeringOutput;
use crate::vec::Vec;

/// Produces a steering force that moves the agent toward the target at max speed.
///
/// The returned force is the difference between the desired velocity (toward the
/// target at `max_speed`) and the agent's current velocity.
pub fn seek<V: Vec>(agent: &Agent<V>, target: V) -> SteeringOutput<V> {
    let desired = target.sub(agent.position).normalize_or_zero().scale(agent.max_speed);
    SteeringOutput::new(desired.sub(agent.velocity))
}

/// Produces a steering force that moves the agent away from the target at max speed.
///
/// The inverse of [`seek`]: the desired velocity points away from the target.
pub fn flee<V: Vec>(agent: &Agent<V>, target: V) -> SteeringOutput<V> {
    let desired = agent.position.sub(target).normalize_or_zero().scale(agent.max_speed);
    SteeringOutput::new(desired.sub(agent.velocity))
}

/// Produces a steering force that moves the agent toward the target, slowing
/// down as it enters the `slowing_radius`.
///
/// Outside the slowing radius this behaves like [`seek`]. Inside, the desired
/// speed is linearly reduced to zero. When the agent is already at the target,
/// a braking force (opposite current velocity) is returned.
pub fn arrive<V: Vec>(agent: &Agent<V>, target: V, slowing_radius: V::Scalar) -> SteeringOutput<V> {
    let offset = target.sub(agent.position);
    let distance = offset.length();

    if distance < V::Scalar::epsilon() {
        // Already at target -- brake
        return SteeringOutput::new(agent.velocity.neg());
    }

    let speed = if distance < slowing_radius {
        agent.max_speed * (distance / slowing_radius)
    } else {
        agent.max_speed
    };

    let desired = offset.normalize_or_zero().scale(speed);
    SteeringOutput::new(desired.sub(agent.velocity))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec::Vec2;

    fn test_agent() -> Agent<Vec2<f32>> {
        Agent::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), 1.0, 10.0, 20.0)
    }

    #[test]
    fn seek_moves_toward_target() {
        let agent = test_agent();
        let target = Vec2::new(10.0, 0.0);
        let result = seek(&agent, target);
        assert!(result.linear.x > 0.0);
    }

    #[test]
    fn flee_moves_away_from_target() {
        let agent = test_agent();
        let target = Vec2::new(10.0, 0.0);
        let result = flee(&agent, target);
        assert!(result.linear.x < 0.0);
    }

    #[test]
    fn arrive_decelerates_near_target() {
        let mut agent = test_agent();
        agent.position = Vec2::new(8.0, 0.0);
        let target = Vec2::new(10.0, 0.0);
        let slowing_radius = 5.0;

        let far_result = seek(&test_agent(), target);
        let near_result = arrive(&agent, target, slowing_radius);
        assert!(near_result.linear.length() < far_result.linear.length());
    }

    #[test]
    fn arrive_at_target_returns_braking_force() {
        let mut agent = test_agent();
        agent.position = Vec2::new(10.0, 0.0);
        let target = Vec2::new(10.0, 0.0);
        let result = arrive(&agent, target, 5.0);
        assert!(result.linear.x < 0.0);
    }
}
