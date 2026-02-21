//! Cohesion behavior — steer toward center of mass of neighbors.

use crate::float::Float;
use crate::vec::Vec;
use crate::agent::Agent;
use crate::steering::SteeringOutput;
use crate::seek;

/// Cohesion: steer toward the center of mass of nearby neighbors.
///
/// # Arguments
/// - `neighbor_positions` — iterator of neighbor world positions
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
    seek::seek(agent, center)
}
