//! Flocking — combined separation + alignment + cohesion with configurable weights.

use crate::float::Float;
use crate::vec::Vec;
use crate::agent::Agent;
use crate::steering::SteeringOutput;
use crate::separation;
use crate::alignment;
use crate::cohesion;

/// Weight configuration for the three flocking sub-behaviors.
#[derive(Copy, Clone, Debug)]
pub struct FlockWeights<F: Float> {
    pub separation: F,
    pub alignment: F,
    pub cohesion: F,
}

impl<F: Float> FlockWeights<F> {
    /// Creates new flock weights with explicit values.
    pub fn new(separation: F, alignment: F, cohesion: F) -> Self {
        Self { separation, alignment, cohesion }
    }

    /// Returns the classic Reynolds weighting: separation=1.5, alignment=1.0, cohesion=1.0.
    pub fn default_reynolds() -> Self {
        Self {
            separation: F::one() + F::half(),
            alignment: F::one(),
            cohesion: F::one(),
        }
    }
}

/// Computes a combined flocking steering force from separation, alignment, and cohesion.
///
/// Each sub-behavior is computed independently and then weighted according to the
/// provided `FlockWeights`. The result is a single combined steering output.
///
/// # Arguments
/// * `agent` - The agent to compute flocking for.
/// * `positions` - Slice of neighbor positions (should NOT include the agent itself).
/// * `velocities` - Slice of neighbor velocities (matching indices with `positions`).
/// * `weights` - The blend weights for each sub-behavior.
pub fn flock<V: Vec>(
    agent: &Agent<V>,
    positions: &[V],
    velocities: &[V],
    weights: &FlockWeights<V::Scalar>,
) -> SteeringOutput<V> {
    let sep = separation::separation(agent, positions.iter().copied());
    let ali = alignment::alignment(agent, velocities.iter().copied());
    let coh = cohesion::cohesion(agent, positions.iter().copied());

    let combined = sep.scale(weights.separation)
        .add(ali.scale(weights.alignment))
        .add(coh.scale(weights.cohesion));

    combined
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec::Vec2;
    use crate::agent::Agent;
    use crate::steering::apply_steering;
    use crate::separation::separation;
    use crate::cohesion::cohesion;
    use crate::alignment::alignment;
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec as AllocVec;

    fn test_agent() -> Agent<Vec2<f32>> {
        Agent::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), 1.0, 10.0, 20.0)
    }

    #[test]
    fn separation_pushes_agents_apart() {
        let agent = test_agent();
        let neighbors = vec![Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)];
        let result = separation(&agent, neighbors.into_iter());
        assert!(result.linear.length() > 0.0);
    }

    #[test]
    fn cohesion_pulls_toward_center() {
        let agent = test_agent();
        let neighbors = vec![Vec2::new(10.0, 0.0), Vec2::new(10.0, 10.0)];
        let result = cohesion(&agent, neighbors.into_iter());
        assert!(result.linear.x > 0.0);
    }

    #[test]
    fn alignment_matches_heading() {
        let agent = test_agent();
        let neighbor_vels = vec![Vec2::new(0.0, 5.0), Vec2::new(0.0, 3.0)];
        let result = alignment(&agent, neighbor_vels.into_iter());
        assert!(result.linear.y > 0.0);
    }

    #[test]
    fn flock_stable_over_1000_steps() {
        let weights = FlockWeights::default_reynolds();
        let mut agents: AllocVec<Agent<Vec2<f32>>> = (0..10)
            .map(|i| Agent::new(
                Vec2::new(i as f32 * 2.0, 0.0),
                Vec2::new(1.0, 0.0),
                1.0, 5.0, 10.0,
            ))
            .collect();

        for _ in 0..1000 {
            let positions: AllocVec<_> = agents.iter().map(|a| a.position).collect();
            let velocities: AllocVec<_> = agents.iter().map(|a| a.velocity).collect();

            let mut new_agents = AllocVec::new();
            for (idx, agent) in agents.iter().enumerate() {
                let neighbor_pos: AllocVec<_> = positions.iter().enumerate()
                    .filter(|(i, _)| *i != idx).map(|(_, p)| *p).collect();
                let neighbor_vel: AllocVec<_> = velocities.iter().enumerate()
                    .filter(|(i, _)| *i != idx).map(|(_, v)| *v).collect();

                let steering = flock(agent, &neighbor_pos, &neighbor_vel, &weights);
                new_agents.push(apply_steering(agent, &steering, 0.016));
            }
            agents = new_agents;
        }

        for agent in &agents {
            assert!(agent.position.length() < 10000.0, "Agent diverged: {:?}", agent.position);
        }
    }
}
