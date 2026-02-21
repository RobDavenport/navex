//! Separation behavior — push agents apart from neighbors.

use crate::float::Float;
use crate::vec::Vec;
use crate::agent::Agent;
use crate::steering::SteeringOutput;

/// Computes a separation steering force that pushes an agent away from neighbors.
///
/// Each neighbor contributes a repulsion force inversely proportional to its distance.
/// The result is a Reynolds-style steering force (desired velocity minus current velocity).
///
/// # Arguments
/// * `agent` - The agent to compute separation for.
/// * `neighbors` - Iterator of neighbor positions (should NOT include the agent itself).
pub fn separation<V: Vec>(agent: &Agent<V>, neighbors: impl Iterator<Item = V>) -> SteeringOutput<V> {
    let mut force = V::zero();
    let mut count = 0u32;

    for neighbor_pos in neighbors {
        let offset = agent.position.sub(neighbor_pos);
        let dist_sq = offset.length_sq();

        if dist_sq > V::Scalar::epsilon() {
            let dist = dist_sq.sqrt();
            force = force.add(offset.normalize_or_zero().scale(dist.recip()));
            count += 1;
        }
    }

    if count > 0 {
        let desired = force.normalize_or_zero().scale(agent.max_speed);
        SteeringOutput::new(desired.sub(agent.velocity))
    } else {
        SteeringOutput::zero()
    }
}
