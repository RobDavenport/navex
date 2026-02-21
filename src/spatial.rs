//! Spatial query — neighbor lookups for group behaviors.

use alloc::vec::Vec as AllocVec;
use crate::float::Float;
use crate::vec::Vec;

/// Information about a neighbor returned from spatial queries.
#[derive(Copy, Clone, Debug)]
pub struct NeighborInfo<V: Vec> {
    /// The neighbor's position.
    pub position: V,
    /// The neighbor's velocity.
    pub velocity: V,
    /// The distance from the query center to this neighbor.
    pub distance: V::Scalar,
}

/// Trait for spatial query structures that find neighbors within a radius.
pub trait SpatialQuery<V: Vec> {
    /// Returns all neighbors within `radius` of `center`, sorted nearest-first.
    fn query_radius(&self, center: V, radius: V::Scalar) -> AllocVec<NeighborInfo<V>>;
}

/// Brute-force O(n) spatial query that checks every inserted entity.
///
/// Simple and correct for small populations. For large populations,
/// consider a grid or tree-based approach.
pub struct BruteForceQuery<V: Vec> {
    positions: AllocVec<V>,
    velocities: AllocVec<V>,
}

impl<V: Vec> BruteForceQuery<V> {
    /// Creates an empty brute-force query structure.
    pub fn new() -> Self {
        Self {
            positions: AllocVec::new(),
            velocities: AllocVec::new(),
        }
    }

    /// Removes all inserted entities.
    pub fn clear(&mut self) {
        self.positions.clear();
        self.velocities.clear();
    }

    /// Inserts an entity with the given position and velocity.
    pub fn insert(&mut self, position: V, velocity: V) {
        self.positions.push(position);
        self.velocities.push(velocity);
    }

    /// Returns all neighbors within `radius` of `center`, sorted nearest-first.
    ///
    /// Entities whose distance from `center` is below `epsilon` are excluded
    /// (this filters out the querying entity itself when it is in the structure).
    pub fn query_radius(&self, center: V, radius: V::Scalar) -> AllocVec<NeighborInfo<V>> {
        let radius_sq = radius * radius;
        let mut results = AllocVec::new();

        for i in 0..self.positions.len() {
            let dist_sq = center.distance_sq(self.positions[i]);
            if dist_sq < radius_sq && dist_sq > V::Scalar::epsilon() {
                results.push(NeighborInfo {
                    position: self.positions[i],
                    velocity: self.velocities[i],
                    distance: dist_sq.sqrt(),
                });
            }
        }

        // Sort by distance (nearest first)
        results.sort_by(|a, b| {
            a.distance
                .to_f32()
                .partial_cmp(&b.distance.to_f32())
                .unwrap()
        });
        results
    }
}

impl<V: Vec> SpatialQuery<V> for BruteForceQuery<V> {
    fn query_radius(&self, center: V, radius: V::Scalar) -> AllocVec<NeighborInfo<V>> {
        self.query_radius(center, radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec::Vec2;

    #[test]
    fn brute_force_finds_neighbors() {
        let mut q = BruteForceQuery::new();
        q.insert(Vec2::new(1.0f32, 0.0), Vec2::new(1.0, 0.0));
        q.insert(Vec2::new(5.0, 0.0), Vec2::new(0.0, 1.0));
        q.insert(Vec2::new(100.0, 0.0), Vec2::new(-1.0, 0.0));

        let results = q.query_radius(Vec2::new(0.0, 0.0), 10.0);
        assert_eq!(results.len(), 2);
        // Nearest first
        assert!((results[0].distance - 1.0).abs() < 0.01);
        assert!((results[1].distance - 5.0).abs() < 0.01);
    }

    #[test]
    fn brute_force_excludes_self() {
        let mut q = BruteForceQuery::new();
        q.insert(Vec2::new(0.0f32, 0.0), Vec2::new(0.0, 0.0));
        q.insert(Vec2::new(1.0, 0.0), Vec2::new(1.0, 0.0));

        let results = q.query_radius(Vec2::new(0.0, 0.0), 10.0);
        // Self (distance ~0) should be excluded
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn brute_force_sorted_by_distance() {
        let mut q = BruteForceQuery::new();
        q.insert(Vec2::new(10.0f32, 0.0), Vec2::new(0.0, 0.0));
        q.insert(Vec2::new(2.0, 0.0), Vec2::new(0.0, 0.0));
        q.insert(Vec2::new(5.0, 0.0), Vec2::new(0.0, 0.0));

        let results = q.query_radius(Vec2::new(0.0, 0.0), 20.0);
        assert_eq!(results.len(), 3);
        assert!(results[0].distance <= results[1].distance);
        assert!(results[1].distance <= results[2].distance);
    }

    #[test]
    fn spatial_query_trait() {
        let mut q = BruteForceQuery::new();
        q.insert(Vec2::new(3.0f32, 0.0), Vec2::new(1.0, 0.0));

        let spatial: &dyn SpatialQuery<Vec2<f32>> = &q;
        let results = spatial.query_radius(Vec2::new(0.0, 0.0), 5.0);
        assert_eq!(results.len(), 1);
    }
}
