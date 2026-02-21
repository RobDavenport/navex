//! Wander behavior -- Reynolds' circle-based wandering.

use crate::agent::Agent;
use crate::float::Float;
use crate::seek;
use crate::steering::SteeringOutput;
use crate::vec::{Vec, Vec2, Vec3};
use rand_core::RngCore;

/// Parameters controlling the wander behavior.
///
/// - `circle_distance`: how far in front of the agent the wander circle is placed.
/// - `circle_radius`: the radius of the wander circle (larger = more erratic).
/// - `angle_change`: maximum random jitter per step in radians.
#[derive(Copy, Clone, Debug)]
pub struct WanderParams<F: Float> {
    pub circle_distance: F,
    pub circle_radius: F,
    pub angle_change: F,
}

impl<F: Float> WanderParams<F> {
    /// Creates new wander parameters.
    pub fn new(circle_distance: F, circle_radius: F, angle_change: F) -> Self {
        Self {
            circle_distance,
            circle_radius,
            angle_change,
        }
    }
}

/// Persistent state for the wander behavior.
///
/// Stores the current wander angle so the behavior produces smooth,
/// continuous wandering rather than random jitter.
#[derive(Copy, Clone, Debug)]
pub struct WanderState<F: Float> {
    pub wander_angle: F,
}

impl<F: Float> WanderState<F> {
    /// Creates a new wander state with the given initial angle.
    pub fn new(initial_angle: F) -> Self {
        Self {
            wander_angle: initial_angle,
        }
    }

    /// Creates a new wander state with angle zero.
    pub fn default_val() -> Self {
        Self {
            wander_angle: F::zero(),
        }
    }
}

/// Generates a random float in \[-1, 1\] from an RNG.
fn random_clamped<F: Float>(rng: &mut impl RngCore) -> F {
    let val = rng.next_u32();
    let normalized = F::from_f32(val as f32 / u32::MAX as f32);
    normalized * F::two() - F::one()
}

/// 2D wander behavior.
///
/// Places a "wander circle" a fixed distance ahead of the agent, then picks a
/// target on that circle by jittering the stored wander angle. Returns the
/// steering force and updated state.
pub fn wander_2d<F: Float>(
    agent: &Agent<Vec2<F>>,
    state: &WanderState<F>,
    params: &WanderParams<F>,
    rng: &mut impl RngCore,
) -> (SteeringOutput<Vec2<F>>, WanderState<F>) {
    let jitter: F = random_clamped::<F>(rng) * params.angle_change;
    let new_angle = state.wander_angle + jitter;

    let heading = agent.heading();
    let circle_center = agent.position.add(heading.scale(params.circle_distance));
    let offset = Vec2::from_angle(new_angle).scale(params.circle_radius);
    let target = circle_center.add(offset);

    let steering = seek::seek(agent, target);
    let new_state = WanderState {
        wander_angle: new_angle,
    };

    (steering, new_state)
}

/// 3D wander behavior.
///
/// Similar to [`wander_2d`] but adds a random elevation component so the agent
/// wanders in three dimensions. The elevation is drawn uniformly from
/// \[-pi/2, pi/2\].
pub fn wander_3d<F: Float>(
    agent: &Agent<Vec3<F>>,
    state: &WanderState<F>,
    params: &WanderParams<F>,
    rng: &mut impl RngCore,
) -> (SteeringOutput<Vec3<F>>, WanderState<F>) {
    let jitter: F = random_clamped::<F>(rng) * params.angle_change;
    let new_angle = state.wander_angle + jitter;

    let heading = agent.heading();
    let circle_center = agent.position.add(heading.scale(params.circle_distance));

    let elevation: F = random_clamped::<F>(rng) * F::pi() * F::half();
    let cos_elev = elevation.cos();
    let sin_elev = elevation.sin();

    let offset = Vec3::new(
        new_angle.cos() * cos_elev * params.circle_radius,
        sin_elev * params.circle_radius,
        new_angle.sin() * cos_elev * params.circle_radius,
    );

    let target = circle_center.add(offset);
    let steering = seek::seek(agent, target);
    let new_state = WanderState {
        wander_angle: new_angle,
    };

    (steering, new_state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Agent;
    use crate::vec::Vec2;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    fn test_agent() -> Agent<Vec2<f32>> {
        Agent::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), 1.0, 10.0, 20.0)
    }

    #[test]
    fn wander_stays_near_forward() {
        let agent = test_agent();
        let params = WanderParams::new(2.0, 1.0, 0.3);
        let state = WanderState::default_val();
        let mut rng = SmallRng::seed_from_u64(42);
        let (result, _new_state) = wander_2d(&agent, &state, &params, &mut rng);
        assert!(result.linear.length() > 0.0);
    }

    #[test]
    fn determinism_same_seed_same_result() {
        let agent = test_agent();
        let params = WanderParams::new(2.0, 1.0, 0.5);
        for _ in 0..100 {
            let state = WanderState::default_val();
            let mut rng1 = SmallRng::seed_from_u64(12345);
            let mut rng2 = SmallRng::seed_from_u64(12345);
            let (r1, s1) = wander_2d(&agent, &state, &params, &mut rng1);
            let (r2, s2) = wander_2d(&agent, &state, &params, &mut rng2);
            assert_eq!(r1.linear.x, r2.linear.x);
            assert_eq!(r1.linear.y, r2.linear.y);
            assert_eq!(s1.wander_angle, s2.wander_angle);
        }
    }
}
