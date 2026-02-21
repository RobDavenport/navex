//! Obstacle avoidance — circles, AABBs, and walls.

use crate::agent::Agent;
use crate::float::Float;
use crate::steering::SteeringOutput;
use crate::vec::{Vec, Vec2};

/// A circular obstacle with center and radius.
#[derive(Copy, Clone, Debug)]
pub struct Circle<V: Vec> {
    pub center: V,
    pub radius: V::Scalar,
}

/// An axis-aligned bounding box obstacle defined by min/max corners.
#[derive(Copy, Clone, Debug)]
pub struct Aabb<V: Vec> {
    pub min: V,
    pub max: V,
}

/// An infinite wall defined by a point on the wall and an outward-facing normal.
#[derive(Copy, Clone, Debug)]
pub struct Wall<V: Vec> {
    pub point: V,
    /// Must be unit length.
    pub normal: V,
}

/// Produces a steering force to avoid circular obstacles.
///
/// Casts a ray along the agent's heading up to `detection_length`. If the ray
/// passes through any circle, a lateral force pushes the agent away from the
/// nearest penetrated obstacle.
pub fn avoid_circles<V: Vec>(
    agent: &Agent<V>,
    obstacles: &[Circle<V>],
    detection_length: V::Scalar,
) -> SteeringOutput<V> {
    let heading = agent.heading();
    let mut force = V::zero();
    let mut closest_dist = detection_length;

    for obs in obstacles {
        let to_center = obs.center.sub(agent.position);
        let proj = to_center.dot(heading);

        // Obstacle is behind the agent.
        if proj < V::Scalar::zero() {
            continue;
        }

        let proj_clamped = proj.min(detection_length);
        let closest_point = agent.position.add(heading.scale(proj_clamped));
        let dist_to_center = closest_point.distance(obs.center);

        if dist_to_center < obs.radius && proj_clamped < closest_dist {
            closest_dist = proj_clamped;
            let lateral = closest_point.sub(obs.center);
            // When the ray passes exactly through the center, the lateral
            // offset is zero. Fall back to the agent-to-center direction
            // (flee from center) so we always produce a nonzero force.
            let avoidance = if lateral.length_sq() < V::Scalar::epsilon() * V::Scalar::epsilon() {
                agent.position.sub(obs.center).normalize_or_zero()
            } else {
                lateral.normalize_or_zero()
            };
            let urgency = (obs.radius - dist_to_center) / obs.radius;
            force = avoidance.scale(agent.max_force * urgency);
        }
    }

    SteeringOutput::new(force)
}

/// Produces a steering force to avoid axis-aligned bounding box obstacles.
///
/// Projects the agent's heading onto each AABB. If the projection falls inside
/// an AABB, a force pushes the agent away from the box center.
pub fn avoid_aabbs<V: Vec>(
    agent: &Agent<V>,
    obstacles: &[Aabb<V>],
    detection_length: V::Scalar,
) -> SteeringOutput<V> {
    let heading = agent.heading();
    let mut force = V::zero();
    let _best_t = detection_length;

    for obs in obstacles {
        let aabb_center = obs.min.add(obs.max).scale(V::Scalar::half());
        let to_center = aabb_center.sub(agent.position);
        let proj = to_center.dot(heading);

        if proj > V::Scalar::zero() && proj < detection_length {
            let closest_on_ray = agent.position.add(heading.scale(proj));
            let half_extents = obs.max.sub(obs.min).scale(V::Scalar::half());
            let local = closest_on_ray.sub(aabb_center);

            let dist_to_surface = local.length() - half_extents.length();
            if dist_to_surface < V::Scalar::zero() {
                let avoidance = agent.position.sub(aabb_center).normalize_or_zero();
                let urgency = (-dist_to_surface) / half_extents.length();
                force = avoidance.scale(agent.max_force * urgency);
            }
        }
    }

    SteeringOutput::new(force)
}

/// Produces a steering force to avoid infinite walls.
///
/// Extends a single feeler ray along the agent's heading by `feeler_length`.
/// If the feeler tip crosses behind any wall (negative side of the normal),
/// a force proportional to the penetration depth pushes the agent back.
pub fn avoid_walls<V: Vec>(
    agent: &Agent<V>,
    walls: &[Wall<V>],
    feeler_length: V::Scalar,
) -> SteeringOutput<V> {
    let heading = agent.heading();
    let feeler_tip = agent.position.add(heading.scale(feeler_length));

    let mut force = V::zero();

    for wall in walls {
        let to_tip = feeler_tip.sub(wall.point);
        let signed_dist = to_tip.dot(wall.normal);

        if signed_dist < V::Scalar::zero() {
            force = force.add(wall.normal.scale(-signed_dist));
        }
    }

    SteeringOutput::new(force)
}

/// Produces a steering force to avoid infinite walls using three 2D feelers.
///
/// In addition to the main forward feeler, two shorter side feelers are cast
/// at +/- 30 degrees from the heading. This provides earlier detection when
/// the agent approaches a wall at an angle.
pub fn avoid_walls_2d<F: Float>(
    agent: &Agent<Vec2<F>>,
    walls: &[Wall<Vec2<F>>],
    feeler_length: F,
) -> SteeringOutput<Vec2<F>> {
    let heading = agent.heading();
    let short_length = feeler_length * F::from_f32(0.7);

    let angle_offset = F::from_f32(core::f32::consts::FRAC_PI_6); // 30 degrees
    let heading_angle = heading.angle();

    let feelers = [
        agent.position.add(heading.scale(feeler_length)),
        agent
            .position
            .add(Vec2::from_angle(heading_angle + angle_offset).scale(short_length)),
        agent
            .position
            .add(Vec2::from_angle(heading_angle - angle_offset).scale(short_length)),
    ];

    let mut force = Vec2::zero();

    for feeler_tip in &feelers {
        for wall in walls {
            let to_tip = feeler_tip.sub(wall.point);
            let signed_dist = to_tip.dot(wall.normal);

            if signed_dist < F::zero() {
                force = force.add(wall.normal.scale(-signed_dist));
            }
        }
    }

    SteeringOutput::new(force)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Agent;
    use crate::vec::Vec2;

    #[test]
    fn avoid_steers_around_circle() {
        let agent = Agent::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            1.0,
            10.0,
            20.0,
        );
        let obstacle = Circle {
            center: Vec2::new(5.0, 0.0),
            radius: 2.0,
        };
        let result = avoid_circles(&agent, &[obstacle], 10.0);
        assert!(result.linear.length() > 0.0);
    }

    #[test]
    fn avoid_no_force_when_no_obstacles() {
        let agent = Agent::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            1.0,
            10.0,
            20.0,
        );
        let result = avoid_circles::<Vec2<f32>>(&agent, &[], 10.0);
        assert!(result.linear.length() < 0.001);
    }

    #[test]
    fn avoid_wall_pushes_back() {
        let agent = Agent::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            1.0,
            10.0,
            20.0,
        );
        let wall = Wall {
            point: Vec2::new(5.0, 0.0),
            normal: Vec2::new(-1.0, 0.0), // facing left
        };
        let result = avoid_walls(&agent, &[wall], 10.0);
        // Feeler tip at (10, 0), wall point at (5, 0), wall normal (-1, 0)
        // to_tip = (10, 0) - (5, 0) = (5, 0)
        // signed_dist = (5, 0) dot (-1, 0) = -5
        // Force should push in normal direction (-x)
        assert!(result.linear.x < 0.0);
    }
}
