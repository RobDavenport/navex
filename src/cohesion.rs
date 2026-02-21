//! Cohesion behavior — steer toward center of mass of neighbors.

use crate::float::Float;
use crate::vec::Vec;
use crate::agent::Agent;
use crate::steering::SteeringOutput;

/// Computes a cohesion steering force toward the center of mass of neighbors.
///
/// Averages neighbor positions and seeks toward that centroid. This is equivalent
/// to calling `seek` on the average neighbor position.
///
/// # Arguments
/// * `agent` - The agent to compute cohesion for.
/// * `neighbor_positions` - Iterator of neighbor position vectors.
pub fn cohesion<V: Vec>(agent: &Agent<V>, neighbor_positions: impl Iterator<Item = V>) -> SteeringOutput<V> {
    let mut center = V::zero();
    let mut count: u32 = 0;

    for pos in neighbor_positions {
        center = center.add(pos);
        count += 1;
    }

    if count == 0 {
        return SteeringOutput::zero();
    }

    let center = center.scale(V::Scalar::one() / V::Scalar::from_f32(count as f32));

    // Inline seek: desired = normalize(target - position) * max_speed, force = desired - velocity
    let to_target = center.sub(agent.position);
    let desired = to_target.normalize_or_zero().scale(agent.max_speed);
    SteeringOutput::new(desired.sub(agent.velocity))
}
