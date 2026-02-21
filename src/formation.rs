//! Formation patterns — circle, V, grid, column, leader-follow.

use alloc::vec::Vec as AllocVec;
use crate::float::Float;
use crate::vec::{Vec, Vec2};
use crate::agent::Agent;
use crate::steering::SteeringOutput;
use crate::seek;

/// A single slot in a formation, defined by its offset from the formation center.
#[derive(Copy, Clone, Debug)]
pub struct FormationSlot<V: Vec> {
    pub offset: V,
}

/// Trait for types that generate formation slot layouts.
pub trait FormationPattern<V: Vec> {
    /// Returns `count` slots arranged according to this pattern.
    fn slots(&self, count: usize) -> AllocVec<FormationSlot<V>>;
}

/// Arranges slots evenly around a circle of the given radius.
#[derive(Copy, Clone, Debug)]
pub struct CircleFormation<F: Float> {
    pub radius: F,
}

impl<F: Float> FormationPattern<Vec2<F>> for CircleFormation<F> {
    fn slots(&self, count: usize) -> AllocVec<FormationSlot<Vec2<F>>> {
        let mut result = AllocVec::with_capacity(count);
        if count == 0 { return result; }

        let step = F::two() * F::pi() / F::from_f32(count as f32);
        for i in 0..count {
            let angle = step * F::from_f32(i as f32);
            result.push(FormationSlot {
                offset: Vec2::new(angle.cos() * self.radius, angle.sin() * self.radius),
            });
        }
        result
    }
}

/// Arranges slots in a V (chevron) shape. The leader sits at the origin,
/// with subsequent slots alternating left and right along the V arms.
#[derive(Copy, Clone, Debug)]
pub struct VFormation<F: Float> {
    /// Distance between successive ranks along each arm.
    pub spacing: F,
    /// Half-angle of the V opening (radians).
    pub angle: F,
}

impl<F: Float> FormationPattern<Vec2<F>> for VFormation<F> {
    fn slots(&self, count: usize) -> AllocVec<FormationSlot<Vec2<F>>> {
        let mut result = AllocVec::with_capacity(count);
        if count == 0 { return result; }

        // Leader at origin
        result.push(FormationSlot { offset: Vec2::zero() });

        let left_dir = Vec2::new(-self.angle.sin(), -self.angle.cos());
        let right_dir = Vec2::new(self.angle.sin(), -self.angle.cos());

        for i in 1..count {
            let rank = ((i + 1) / 2) as f32;
            let dist = self.spacing * F::from_f32(rank);
            let dir = if i % 2 == 1 { left_dir } else { right_dir };
            result.push(FormationSlot { offset: dir.scale(dist) });
        }

        result
    }
}

/// Arranges slots in a rectangular grid with the given number of columns.
#[derive(Copy, Clone, Debug)]
pub struct GridFormation<F: Float> {
    /// Number of columns in the grid.
    pub cols: usize,
    /// Distance between adjacent slots.
    pub spacing: F,
}

impl<F: Float> FormationPattern<Vec2<F>> for GridFormation<F> {
    fn slots(&self, count: usize) -> AllocVec<FormationSlot<Vec2<F>>> {
        let mut result = AllocVec::with_capacity(count);
        if count == 0 || self.cols == 0 { return result; }

        let rows = (count + self.cols - 1) / self.cols;
        let center_col = F::from_f32((self.cols - 1) as f32) * F::half();
        let center_row = F::from_f32((rows - 1) as f32) * F::half();

        for i in 0..count {
            let col = i % self.cols;
            let row = i / self.cols;
            let x = (F::from_f32(col as f32) - center_col) * self.spacing;
            let y = (F::from_f32(row as f32) - center_row) * self.spacing;
            result.push(FormationSlot { offset: Vec2::new(x, y) });
        }

        result
    }
}

/// Arranges slots in a single column (single file) along the negative Y axis.
#[derive(Copy, Clone, Debug)]
pub struct ColumnFormation<F: Float> {
    /// Distance between successive slots.
    pub spacing: F,
}

impl<F: Float> FormationPattern<Vec2<F>> for ColumnFormation<F> {
    fn slots(&self, count: usize) -> AllocVec<FormationSlot<Vec2<F>>> {
        let mut result = AllocVec::with_capacity(count);
        for i in 0..count {
            result.push(FormationSlot {
                offset: Vec2::new(F::zero(), -F::from_f32(i as f32) * self.spacing),
            });
        }
        result
    }
}

/// Arranges slots in a staggered follow pattern behind a leader.
#[derive(Copy, Clone, Debug)]
pub struct LeaderFollow<F: Float> {
    /// `x` is the lateral offset, `y` is the longitudinal spacing per rank.
    pub offset: Vec2<F>,
}

impl<F: Float> FormationPattern<Vec2<F>> for LeaderFollow<F> {
    fn slots(&self, count: usize) -> AllocVec<FormationSlot<Vec2<F>>> {
        let mut result = AllocVec::with_capacity(count);
        for i in 0..count {
            let stagger = F::from_f32(i as f32) * self.offset.y;
            result.push(FormationSlot {
                offset: Vec2::new(self.offset.x, stagger),
            });
        }
        result
    }
}

/// A concrete formation instance: a center position, heading, and computed slots.
pub struct Formation<V: Vec> {
    /// World-space center of the formation.
    pub center: V,
    /// Normalized heading direction.
    pub heading: V,
    /// Slots in local space relative to the center.
    pub slots: AllocVec<FormationSlot<V>>,
}

impl<F: Float> Formation<Vec2<F>> {
    /// Creates a formation from a pattern, placing `count` slots around `center`
    /// oriented along `heading`.
    pub fn from_pattern(
        center: Vec2<F>,
        heading: Vec2<F>,
        count: usize,
        pattern: &dyn FormationPattern<Vec2<F>>,
    ) -> Self {
        Self {
            center,
            heading: heading.normalize_or_zero(),
            slots: pattern.slots(count),
        }
    }

    /// Returns the world-space position of the slot at `slot_idx`.
    /// If the index is out of bounds, returns the formation center.
    pub fn world_slot_position(&self, slot_idx: usize) -> Vec2<F> {
        if slot_idx >= self.slots.len() {
            return self.center;
        }

        let offset = self.slots[slot_idx].offset;
        let angle = self.heading.angle();
        let cos_a = angle.cos();
        let sin_a = angle.sin();

        let rotated = Vec2::new(
            offset.x * cos_a - offset.y * sin_a,
            offset.x * sin_a + offset.y * cos_a,
        );

        self.center.add(rotated)
    }
}

/// Steers an agent toward its assigned formation slot using arrive behavior.
pub fn steer_to_slot<F: Float>(
    agent: &Agent<Vec2<F>>,
    formation: &Formation<Vec2<F>>,
    slot_idx: usize,
    slowing_radius: F,
) -> SteeringOutput<Vec2<F>> {
    let target = formation.world_slot_position(slot_idx);
    seek::arrive(agent, target, slowing_radius)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec::Vec2;
    use crate::vec::Vec;

    #[test]
    fn circle_formation_slots_equidistant() {
        let pattern = CircleFormation { radius: 10.0f32 };
        let slots = pattern.slots(4);
        assert_eq!(slots.len(), 4);
        for slot in &slots {
            let dist = slot.offset.length();
            assert!((dist - 10.0).abs() < 0.01);
        }
    }

    #[test]
    fn grid_formation_correct_count() {
        let pattern = GridFormation { cols: 3, spacing: 2.0f32 };
        let slots = pattern.slots(7);
        assert_eq!(slots.len(), 7);
    }

    #[test]
    fn v_formation_leader_at_origin() {
        let pattern = VFormation { spacing: 2.0f32, angle: core::f32::consts::FRAC_PI_6 };
        let slots = pattern.slots(5);
        assert_eq!(slots.len(), 5);
        assert!((slots[0].offset.x).abs() < 0.001);
        assert!((slots[0].offset.y).abs() < 0.001);
    }

    #[test]
    fn column_formation_single_file() {
        let pattern = ColumnFormation { spacing: 3.0f32 };
        let slots = pattern.slots(4);
        assert_eq!(slots.len(), 4);
        for slot in &slots {
            assert!((slot.offset.x).abs() < 0.001);
        }
    }

    #[test]
    fn formation_world_position() {
        let pattern = CircleFormation { radius: 5.0f32 };
        let formation = Formation::from_pattern(
            Vec2::new(10.0, 10.0),
            Vec2::new(1.0, 0.0),
            4,
            &pattern,
        );
        // First slot should be at (radius, 0) rotated by heading, translated
        let pos = formation.world_slot_position(0);
        assert!((pos.x - 15.0).abs() < 0.1);
        assert!((pos.y - 10.0).abs() < 0.1);
    }
}
