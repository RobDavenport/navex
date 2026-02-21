//! Path following — waypoint sequences and path-seek behavior.

use alloc::vec::Vec as AllocVec;
use crate::float::Float;
use crate::vec::Vec;
use crate::agent::Agent;
use crate::steering::SteeringOutput;
use crate::seek;

/// A polyline path defined by a sequence of waypoints with precomputed cumulative lengths.
pub struct Path<V: Vec> {
    /// Waypoints defining the path.
    pub points: AllocVec<V>,
    /// Cumulative arc-length at each waypoint (starts at 0).
    pub cumulative: AllocVec<V::Scalar>,
    /// Total arc-length of the path.
    pub total_length: V::Scalar,
}

impl<V: Vec> Path<V> {
    /// Creates a new path from a list of waypoints, precomputing cumulative lengths.
    pub fn new(points: AllocVec<V>) -> Self {
        let mut cumulative = AllocVec::with_capacity(points.len());
        let mut total = V::Scalar::zero();
        cumulative.push(V::Scalar::zero());

        for i in 1..points.len() {
            let seg_len = points[i - 1].distance(points[i]);
            total = total + seg_len;
            cumulative.push(total);
        }

        Self {
            points,
            cumulative,
            total_length: total,
        }
    }

    /// Finds the closest point on the path to `pos`, returning the point and its
    /// arc-length parameter along the path.
    pub fn closest_point(&self, pos: V) -> (V, V::Scalar) {
        if self.points.len() < 2 {
            if self.points.is_empty() {
                return (V::zero(), V::Scalar::zero());
            }
            return (self.points[0], V::Scalar::zero());
        }

        let mut best_point = self.points[0];
        let mut best_param = V::Scalar::zero();
        let mut best_dist_sq = pos.distance_sq(self.points[0]);

        for i in 0..(self.points.len() - 1) {
            let a = self.points[i];
            let b = self.points[i + 1];
            let ab = b.sub(a);
            let ap = pos.sub(a);
            let ab_len_sq = ab.length_sq();

            if ab_len_sq < V::Scalar::epsilon() {
                continue;
            }

            let t = ap.dot(ab) / ab_len_sq;
            let t = t.max(V::Scalar::zero()).min(V::Scalar::one());

            let closest = a.add(ab.scale(t));
            let dist_sq = pos.distance_sq(closest);

            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_point = closest;
                let seg_len = ab_len_sq.sqrt();
                best_param = self.cumulative[i] + seg_len * t;
            }
        }

        (best_point, best_param)
    }

    /// Returns the point on the path at the given arc-length parameter.
    /// The parameter is clamped to `[0, total_length]`.
    pub fn point_at(&self, param: V::Scalar) -> V {
        if self.points.len() < 2 {
            return if self.points.is_empty() {
                V::zero()
            } else {
                self.points[0]
            };
        }

        let param = param.max(V::Scalar::zero()).min(self.total_length);

        for i in 0..(self.points.len() - 1) {
            if param <= self.cumulative[i + 1] || i == self.points.len() - 2 {
                let seg_start = self.cumulative[i];
                let seg_len = self.cumulative[i + 1] - seg_start;
                let t = if seg_len > V::Scalar::epsilon() {
                    (param - seg_start) / seg_len
                } else {
                    V::Scalar::zero()
                };
                return self.points[i].lerp(self.points[i + 1], t);
            }
        }

        *self.points.last().unwrap()
    }
}

/// Produces a steering force that follows a path.
///
/// Projects the agent onto the path, then seeks a point `ahead_distance` further
/// along the path. This creates smooth path-following behavior.
pub fn path_follow<V: Vec>(
    agent: &Agent<V>,
    path: &Path<V>,
    ahead_distance: V::Scalar,
) -> SteeringOutput<V> {
    let (_closest, param) = path.closest_point(agent.position);
    let target_param = param + ahead_distance;
    let target = path.point_at(target_param);
    seek::seek(agent, target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec::Vec2;
    extern crate alloc;
    use alloc::vec;

    #[test]
    fn path_total_length() {
        let path = Path::new(vec![
            Vec2::new(0.0f32, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
        ]);
        assert!((path.total_length - 20.0).abs() < 0.01);
    }

    #[test]
    fn path_closest_point() {
        let path = Path::new(vec![
            Vec2::new(0.0f32, 0.0),
            Vec2::new(10.0, 0.0),
        ]);
        let (closest, param) = path.closest_point(Vec2::new(5.0, 3.0));
        assert!((closest.x - 5.0).abs() < 0.01);
        assert!((closest.y - 0.0).abs() < 0.01);
        assert!((param - 5.0).abs() < 0.01);
    }

    #[test]
    fn path_point_at() {
        let path = Path::new(vec![
            Vec2::new(0.0f32, 0.0),
            Vec2::new(10.0, 0.0),
        ]);
        let pt = path.point_at(5.0);
        assert!((pt.x - 5.0).abs() < 0.01);
    }

    #[test]
    fn path_empty() {
        let path = Path::<Vec2<f32>>::new(vec![]);
        let (closest, param) = path.closest_point(Vec2::new(1.0, 1.0));
        assert_eq!(closest.x, 0.0);
        assert_eq!(closest.y, 0.0);
        assert_eq!(param, 0.0);
    }

    #[test]
    fn path_single_point() {
        let path = Path::new(vec![Vec2::new(5.0f32, 5.0)]);
        let (closest, param) = path.closest_point(Vec2::new(1.0, 1.0));
        assert!((closest.x - 5.0).abs() < 0.01);
        assert_eq!(param, 0.0);
    }

    #[test]
    fn path_point_at_clamped() {
        let path = Path::new(vec![
            Vec2::new(0.0f32, 0.0),
            Vec2::new(10.0, 0.0),
        ]);
        // Beyond end
        let pt = path.point_at(15.0);
        assert!((pt.x - 10.0).abs() < 0.01);
        // Before start
        let pt = path.point_at(-5.0);
        assert!((pt.x - 0.0).abs() < 0.01);
    }

    #[test]
    fn path_follow_seeks_ahead() {
        let path = Path::new(vec![
            Vec2::new(0.0f32, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
        ]);
        let agent = Agent::new(
            Vec2::new(5.0, 0.0),
            Vec2::new(1.0, 0.0),
            1.0,
            10.0,
            20.0,
        );
        let result = path_follow(&agent, &path, 3.0);
        // Should seek toward a point ahead on the path (x > 5)
        assert!(result.linear.x > 0.0);
    }

    #[test]
    fn path_multi_segment_closest() {
        let path = Path::new(vec![
            Vec2::new(0.0f32, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 10.0),
        ]);
        // Point closest to the second segment
        let (closest, _param) = path.closest_point(Vec2::new(12.0, 5.0));
        assert!((closest.x - 10.0).abs() < 0.01);
        assert!((closest.y - 5.0).abs() < 0.01);
    }
}
