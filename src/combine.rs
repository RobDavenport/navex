//! Behavior composition — weighted blending and priority selection.

extern crate alloc;
use alloc::vec::Vec as AllocVec;
use crate::vec::Vec;
use crate::steering::SteeringOutput;

/// A steering force paired with a blend weight.
#[derive(Copy, Clone, Debug)]
pub struct WeightedBehavior<V: Vec> {
    pub force: SteeringOutput<V>,
    pub weight: V::Scalar,
}

/// Blends multiple weighted steering forces, truncating the result to `max_force`.
///
/// Each behavior's force is scaled by its weight before summing. The final
/// combined force is clamped to `max_force` magnitude.
pub fn weighted_blend<V: Vec>(
    behaviors: &[WeightedBehavior<V>],
    max_force: V::Scalar,
) -> SteeringOutput<V> {
    let mut result = SteeringOutput::zero();

    for b in behaviors {
        result = result.add(b.force.scale(b.weight));
    }

    result.truncate(max_force)
}

/// Returns the first behavior whose magnitude exceeds `threshold`.
///
/// Behaviors are evaluated in order; the first one with magnitude above the
/// threshold is returned immediately, ignoring all subsequent behaviors.
/// Returns zero if no behavior exceeds the threshold.
pub fn priority_select<V: Vec>(
    behaviors: &[SteeringOutput<V>],
    threshold: V::Scalar,
) -> SteeringOutput<V> {
    for b in behaviors {
        if !b.is_zero(threshold) {
            return *b;
        }
    }
    SteeringOutput::zero()
}

/// A dynamic pipeline for composing steering behaviors at runtime.
///
/// Behaviors are added with weights, then resolved via [`blend`](Self::blend)
/// (weighted sum) or [`select`](Self::select) (priority selection).
/// Requires `alloc`.
pub struct BehaviorPipeline<V: Vec> {
    behaviors: AllocVec<WeightedBehavior<V>>,
    max_force: V::Scalar,
}

impl<V: Vec> BehaviorPipeline<V> {
    /// Creates a new empty pipeline with the given maximum force magnitude.
    pub fn new(max_force: V::Scalar) -> Self {
        Self {
            behaviors: AllocVec::new(),
            max_force,
        }
    }

    /// Adds a steering force with its blend weight to the pipeline.
    pub fn add(&mut self, force: SteeringOutput<V>, weight: V::Scalar) -> &mut Self {
        self.behaviors.push(WeightedBehavior { force, weight });
        self
    }

    /// Resolves the pipeline via weighted blending, capped to `max_force`.
    pub fn blend(&self) -> SteeringOutput<V> {
        weighted_blend(&self.behaviors, self.max_force)
    }

    /// Resolves the pipeline via priority selection.
    ///
    /// Each behavior is scaled by its weight, then the first one exceeding
    /// `threshold` magnitude is returned.
    pub fn select(&self, threshold: V::Scalar) -> SteeringOutput<V> {
        let forces: AllocVec<SteeringOutput<V>> = self
            .behaviors
            .iter()
            .map(|b| b.force.scale(b.weight))
            .collect();
        priority_select(&forces, threshold)
    }

    /// Removes all behaviors from the pipeline.
    pub fn clear(&mut self) {
        self.behaviors.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec::Vec2;
    use crate::steering::SteeringOutput;

    #[test]
    fn weighted_blend_respects_weights() {
        let a = SteeringOutput::new(Vec2::new(10.0f32, 0.0));
        let b = SteeringOutput::new(Vec2::new(0.0, 10.0));

        let result = weighted_blend(
            &[
                WeightedBehavior {
                    force: a,
                    weight: 1.0,
                },
                WeightedBehavior {
                    force: b,
                    weight: 0.0,
                },
            ],
            100.0,
        );

        assert!(result.linear.x > 0.0);
        assert!(result.linear.y.abs() < 0.001);
    }

    #[test]
    fn priority_select_returns_first_significant() {
        let zero = SteeringOutput::<Vec2<f32>>::zero();
        let force = SteeringOutput::new(Vec2::new(5.0, 0.0));

        let result = priority_select(&[zero, force], 0.1);
        assert!(result.linear.x > 0.0);
    }

    #[test]
    fn behavior_pipeline_blend() {
        let mut pipe = BehaviorPipeline::new(100.0f32);
        pipe.add(SteeringOutput::new(Vec2::new(10.0, 0.0)), 2.0);
        pipe.add(SteeringOutput::new(Vec2::new(0.0, 10.0)), 1.0);
        let result = pipe.blend();
        assert!((result.linear.x - 20.0).abs() < 0.01);
        assert!((result.linear.y - 10.0).abs() < 0.01);
    }

    #[test]
    fn behavior_pipeline_clear() {
        let mut pipe = BehaviorPipeline::new(100.0f32);
        pipe.add(SteeringOutput::new(Vec2::new(10.0, 0.0)), 1.0);
        pipe.clear();
        let result = pipe.blend();
        assert!(result.linear.length() < 0.001);
    }
}
