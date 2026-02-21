//! Alignment behavior — match heading with neighbors.

use crate::float::Float;
use crate::vec::Vec;
use crate::agent::Agent;
use crate::steering::SteeringOutput;

/// Computes an alignment steering force that matches the agent's heading with neighbors.
///
/// Averages the velocities of all neighbors and steers toward that average heading
/// at the agent's max speed. Returns a Reynolds-style steering force.
///
/// # Arguments
/// * `agent` - The agent to compute alignment for.
/// * `neighbor_velocities` - Iterator of neighbor velocity vectors.
pub fn alignment<V: Vec>(agent: &Agent<V>, neighbor_velocities: impl Iterator<Item = V>) -> SteeringOutput<V> {
    let mut sum = V::zero();
    let mut count: u32 = 0;

    for vel in neighbor_velocities {
        sum = sum.add(vel);
        count += 1;
    }

    if count == 0 {
        return SteeringOutput::zero();
    }

    let avg = sum.scale(V::Scalar::one() / V::Scalar::from_f32(count as f32));
    let desired = avg.normalize_or_zero().scale(agent.max_speed);
    SteeringOutput::new(desired.sub(agent.velocity))
}
