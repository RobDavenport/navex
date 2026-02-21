//! Flow fields — BFS-generated direction grids for navigation.

use alloc::vec::Vec as AllocVec;
use alloc::collections::VecDeque;
use crate::float::Float;
use crate::vec::{Vec, Vec2};
use crate::agent::Agent;
use crate::steering::SteeringOutput;

/// A 2D grid of direction vectors for flow-field navigation.
///
/// Each cell stores a unit direction vector that agents can sample to determine
/// their desired movement direction. The grid maps world-space positions to cells
/// via `cell_size`.
pub struct FlowField<F: Float> {
    /// Number of columns in the grid.
    pub width: usize,
    /// Number of rows in the grid.
    pub height: usize,
    /// World-space size of each cell.
    pub cell_size: F,
    /// Direction vectors stored in row-major order (y * width + x).
    pub directions: AllocVec<Vec2<F>>,
}

impl<F: Float> FlowField<F> {
    /// Creates a new flow field with all directions set to zero.
    pub fn new(width: usize, height: usize, cell_size: F) -> Self {
        Self {
            width,
            height,
            cell_size,
            directions: AllocVec::from_iter(
                core::iter::repeat(Vec2::zero()).take(width * height),
            ),
        }
    }

    /// Sets the direction vector for the cell at grid coordinates `(x, y)`.
    pub fn set(&mut self, x: usize, y: usize, direction: Vec2<F>) {
        if x < self.width && y < self.height {
            self.directions[y * self.width + x] = direction;
        }
    }

    /// Returns the direction vector for the cell at grid coordinates `(x, y)`.
    /// Returns zero if out of bounds.
    pub fn get(&self, x: usize, y: usize) -> Vec2<F> {
        if x < self.width && y < self.height {
            self.directions[y * self.width + x]
        } else {
            Vec2::zero()
        }
    }

    /// Samples the flow field at a world-space position using nearest-cell lookup.
    pub fn sample(&self, world_pos: Vec2<F>) -> Vec2<F> {
        let gx = (world_pos.x / self.cell_size).to_f32() as isize;
        let gy = (world_pos.y / self.cell_size).to_f32() as isize;

        if gx >= 0
            && gy >= 0
            && (gx as usize) < self.width
            && (gy as usize) < self.height
        {
            self.get(gx as usize, gy as usize)
        } else {
            Vec2::zero()
        }
    }

    /// Samples the flow field at a world-space position using bilinear interpolation
    /// of the four nearest cells. The result is normalized to produce a unit direction.
    pub fn sample_bilinear(&self, world_pos: Vec2<F>) -> Vec2<F> {
        let fx = world_pos.x / self.cell_size - F::half();
        let fy = world_pos.y / self.cell_size - F::half();

        let x0 = fx.to_f32() as isize;
        let y0 = fy.to_f32() as isize;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        let tx = fx - F::from_f32(x0 as f32);
        let ty = fy - F::from_f32(y0 as f32);

        let get_safe = |x: isize, y: isize| -> Vec2<F> {
            if x >= 0
                && y >= 0
                && (x as usize) < self.width
                && (y as usize) < self.height
            {
                self.get(x as usize, y as usize)
            } else {
                Vec2::zero()
            }
        };

        let d00 = get_safe(x0, y0);
        let d10 = get_safe(x1, y0);
        let d01 = get_safe(x0, y1);
        let d11 = get_safe(x1, y1);

        let one_minus_tx = F::one() - tx;
        let one_minus_ty = F::one() - ty;

        let top = d00.scale(one_minus_tx).add(d10.scale(tx));
        let bottom = d01.scale(one_minus_tx).add(d11.scale(tx));
        let result = top.scale(one_minus_ty).add(bottom.scale(ty));

        result.normalize_or_zero()
    }
}

/// Generates a flow field that directs agents toward a target position using BFS.
///
/// Each cell's direction vector points toward the shortest unblocked path to the
/// target. The `blocked` callback returns `true` for impassable cells.
pub fn generate_toward<F: Float>(
    width: usize,
    height: usize,
    cell_size: F,
    target: Vec2<F>,
    blocked: &dyn Fn(usize, usize) -> bool,
) -> FlowField<F> {
    let mut field = FlowField::new(width, height, cell_size);

    let tx = (target.x / cell_size).to_f32() as isize;
    let ty = (target.y / cell_size).to_f32() as isize;

    if tx < 0 || ty < 0 || tx as usize >= width || ty as usize >= height {
        return field;
    }

    let tx = tx as usize;
    let ty = ty as usize;

    let mut dist = AllocVec::from_iter(core::iter::repeat(u32::MAX).take(width * height));

    let idx = |x: usize, y: usize| -> usize { y * width + x };

    let mut queue = VecDeque::new();
    dist[idx(tx, ty)] = 0;
    queue.push_back((tx, ty));

    let dirs: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    while let Some((cx, cy)) = queue.pop_front() {
        let current_dist = dist[idx(cx, cy)];

        for &(dx, dy) in &dirs {
            let nx = cx as isize + dx;
            let ny = cy as isize + dy;

            if nx < 0 || ny < 0 || nx as usize >= width || ny as usize >= height {
                continue;
            }

            let nx = nx as usize;
            let ny = ny as usize;

            if blocked(nx, ny) {
                continue;
            }

            let new_dist = current_dist + 1;
            if new_dist < dist[idx(nx, ny)] {
                dist[idx(nx, ny)] = new_dist;

                let direction = Vec2::new(
                    F::from_f32(cx as f32 - nx as f32),
                    F::from_f32(cy as f32 - ny as f32),
                )
                .normalize_or_zero();

                field.set(nx, ny, direction);
                queue.push_back((nx, ny));
            }
        }
    }

    field
}

/// Produces a steering force that follows a flow field.
///
/// Samples the flow field at the agent's position using bilinear interpolation,
/// then computes a steering force to match the desired direction at max speed.
pub fn flow_follow<F: Float>(
    agent: &Agent<Vec2<F>>,
    field: &FlowField<F>,
) -> SteeringOutput<Vec2<F>> {
    let direction = field.sample_bilinear(agent.position);

    if direction.length_sq() < F::epsilon() {
        return SteeringOutput::zero();
    }

    let desired = direction.scale(agent.max_speed);
    SteeringOutput::new(desired.sub(agent.velocity))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec::Vec2;

    #[test]
    fn flow_field_bfs_points_toward_target() {
        let field = generate_toward(10, 10, 1.0, Vec2::new(5.0, 5.0), &|_, _| false);
        let dir = field.get(0, 5);
        assert!(dir.x > 0.0);
        let dir = field.get(9, 5);
        assert!(dir.x < 0.0);
    }

    #[test]
    fn flow_field_blocked_cells() {
        // Partial wall at x==2 blocking only rows 1..=3, leaving rows 0 and 4 open
        let field = generate_toward(5, 5, 1.0, Vec2::new(4.0, 2.0), &|x, y| {
            x == 2 && y >= 1 && y <= 3
        });
        // Cell (3, 2) should still point toward target
        let dir = field.get(3, 2);
        assert!(dir.x > 0.0);
        // Cell (1, 2) should be blocked from direct path but reachable going around
        let dir = field.get(1, 2);
        assert!(dir.length_sq() > 0.0);
    }

    #[test]
    fn flow_field_new_all_zero() {
        let field = FlowField::<f32>::new(5, 5, 1.0);
        for y in 0..5 {
            for x in 0..5 {
                let d = field.get(x, y);
                assert_eq!(d.x, 0.0);
                assert_eq!(d.y, 0.0);
            }
        }
    }

    #[test]
    fn flow_field_set_get() {
        let mut field = FlowField::<f32>::new(5, 5, 1.0);
        field.set(2, 3, Vec2::new(1.0, 0.0));
        let d = field.get(2, 3);
        assert!((d.x - 1.0).abs() < 1e-6);
        assert!((d.y).abs() < 1e-6);
    }

    #[test]
    fn flow_field_get_out_of_bounds() {
        let field = FlowField::<f32>::new(5, 5, 1.0);
        let d = field.get(10, 10);
        assert_eq!(d.x, 0.0);
        assert_eq!(d.y, 0.0);
    }

    #[test]
    fn flow_field_sample_nearest() {
        let mut field = FlowField::<f32>::new(5, 5, 2.0);
        field.set(1, 1, Vec2::new(0.0, 1.0));
        let d = field.sample(Vec2::new(2.5, 2.5));
        assert!((d.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn flow_field_sample_out_of_bounds() {
        let field = FlowField::<f32>::new(5, 5, 1.0);
        let d = field.sample(Vec2::new(-1.0, -1.0));
        assert_eq!(d.x, 0.0);
        assert_eq!(d.y, 0.0);
    }

    #[test]
    fn flow_follow_produces_steering() {
        let mut field = FlowField::<f32>::new(10, 10, 1.0);
        // Set all cells to point right
        for y in 0..10 {
            for x in 0..10 {
                field.set(x, y, Vec2::new(1.0, 0.0));
            }
        }
        let agent = Agent::new(
            Vec2::new(5.0, 5.0),
            Vec2::new(0.0, 0.0),
            1.0,
            10.0,
            20.0,
        );
        let result = flow_follow(&agent, &field);
        // Should want to go right
        assert!(result.linear.x > 0.0);
    }

    #[test]
    fn generate_toward_out_of_bounds_target() {
        let field = generate_toward(5, 5, 1.0, Vec2::new(-10.0f32, -10.0), &|_, _| false);
        // All directions should be zero since target is out of bounds
        for y in 0..5 {
            for x in 0..5 {
                let d = field.get(x, y);
                assert_eq!(d.x, 0.0);
                assert_eq!(d.y, 0.0);
            }
        }
    }

    #[test]
    fn generate_toward_target_cell_has_zero_direction() {
        let field = generate_toward(5, 5, 1.0, Vec2::new(2.0, 2.0), &|_, _| false);
        // The target cell itself should have zero direction (distance 0, never enqueued as neighbor)
        let d = field.get(2, 2);
        assert_eq!(d.x, 0.0);
        assert_eq!(d.y, 0.0);
    }
}
