//! Pursue and evade behaviors -- predictive seek/flee.

use crate::agent::Agent;
use crate::float::Float;
use crate::seek;
use crate::steering::SteeringOutput;
use crate::vec::Vec;

/// Produces a steering force that intercepts a moving target.
///
/// Predicts the target's future position based on the closing speed and
/// distance, then seeks toward that predicted point. This gives more natural
/// interception paths than naive [`seek::seek`].
pub fn pursue<V: Vec>(agent: &Agent<V>, target: &Agent<V>) -> SteeringOutput<V> {
    let offset = target.position.sub(agent.position);
    let distance = offset.length();

    let closing_speed = agent.max_speed + target.speed();
    let prediction_time = if closing_speed > V::Scalar::epsilon() {
        distance / closing_speed
    } else {
        distance / (agent.max_speed + V::Scalar::epsilon())
    };

    let future_position = target.position.add(target.velocity.scale(prediction_time));
    seek::seek(agent, future_position)
}

/// Produces a steering force that flees from a moving target's predicted
/// future position.
///
/// The inverse of [`pursue`]: predicts where the target will be and flees
/// from that point.
pub fn evade<V: Vec>(agent: &Agent<V>, target: &Agent<V>) -> SteeringOutput<V> {
    let offset = target.position.sub(agent.position);
    let distance = offset.length();

    let closing_speed = agent.max_speed + target.speed();
    let prediction_time = if closing_speed > V::Scalar::epsilon() {
        distance / closing_speed
    } else {
        distance / (agent.max_speed + V::Scalar::epsilon())
    };

    let future_position = target.position.add(target.velocity.scale(prediction_time));
    seek::flee(agent, future_position)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Agent;
    use crate::vec::Vec2;

    fn test_agent() -> Agent<Vec2<f32>> {
        Agent::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), 1.0, 10.0, 20.0)
    }

    #[test]
    fn pursue_intercepts_moving_target() {
        let agent = test_agent();
        let mut target = test_agent();
        target.position = Vec2::new(20.0, 0.0);
        target.velocity = Vec2::new(0.0, 5.0);
        let result = pursue(&agent, &target);
        assert!(result.linear.y > 0.0);
    }

    #[test]
    fn evade_moves_away_from_predicted_position() {
        let agent = test_agent();
        let mut target = test_agent();
        target.position = Vec2::new(5.0, 0.0);
        target.velocity = Vec2::new(0.0, 5.0);
        let result = evade(&agent, &target);
        assert!(result.linear.x < 0.0 || result.linear.y < 0.0);
    }
}
