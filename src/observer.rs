//! SteeringObserver trait for monitoring steering decisions.

use crate::vec::Vec;
use crate::agent::Agent;
use crate::steering::SteeringOutput;

/// Trait for observing steering decisions as they happen.
///
/// All methods have default no-op implementations, so consumers only need
/// to override the events they care about. Useful for debugging, logging,
/// or feeding telemetry without coupling the steering logic to any I/O.
pub trait SteeringObserver<V: Vec> {
    /// Called after a seek computation.
    fn on_seek(&mut self, _agent: &Agent<V>, _target: V, _result: &SteeringOutput<V>) {}

    /// Called after a flock computation.
    fn on_flock(&mut self, _agent: &Agent<V>, _neighbor_count: usize, _result: &SteeringOutput<V>) {}

    /// Called after an obstacle avoidance computation.
    fn on_avoid(&mut self, _agent: &Agent<V>, _obstacle_count: usize, _result: &SteeringOutput<V>) {}

    /// Called after a weighted blend resolves.
    fn on_blend(&mut self, _input_count: usize, _result: &SteeringOutput<V>) {}
}

/// A no-op observer that discards all events. Zero cost.
pub struct NoOpSteeringObserver;

impl<V: Vec> SteeringObserver<V> for NoOpSteeringObserver {}
