//! Error types for steering operations.

use core::fmt;

/// Errors that can occur during steering operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SteeringError {
    /// The agent has zero velocity where a heading is required.
    ZeroVelocity,
    /// No neighbors were provided for a group behavior.
    NoNeighbors,
    /// An invalid weight value was supplied.
    InvalidWeight,
    /// The path has no waypoints.
    PathEmpty,
    /// The queried position is outside the flow field bounds.
    FlowFieldOutOfBounds,
}

impl fmt::Display for SteeringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SteeringError::ZeroVelocity => write!(f, "agent has zero velocity"),
            SteeringError::NoNeighbors => write!(f, "no neighbors provided for group behavior"),
            SteeringError::InvalidWeight => write!(f, "invalid weight value"),
            SteeringError::PathEmpty => write!(f, "path has no waypoints"),
            SteeringError::FlowFieldOutOfBounds => {
                write!(f, "position outside flow field bounds")
            }
        }
    }
}
