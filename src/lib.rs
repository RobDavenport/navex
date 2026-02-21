//! Composable steering behaviors, flocking, flow fields, and formations for games.
//!
//! `navex` provides Craig Reynolds-style autonomous agent movement as pure functions.
//! Each behavior takes an agent and context, returning a steering force. Compose
//! behaviors via weighted blending or priority selection.
//!
//! # Features
//!
//! - **Individual behaviors**: Seek, flee, arrive, pursue, evade, wander
//! - **Group behaviors**: Separation, alignment, cohesion, flocking
//! - **Obstacle avoidance**: Circles, AABBs, walls
//! - **Flow fields**: BFS-generated direction grids
//! - **Formations**: Leader-follow, circle, V, grid, column patterns
//! - **Composable**: Weighted blend or priority-select multiple behaviors
//! - **Spatial decoupled**: Users provide neighbor data via trait or iterators
//! - **Observable**: Monitor steering decisions via the `SteeringObserver` trait
//! - **`no_std` compatible**: Works in embedded and WASM environments

#![no_std]

extern crate alloc;

pub mod float;
pub mod vec;
pub mod agent;
pub mod steering;
pub mod seek;
pub mod pursue;
pub mod wander;
pub mod separation;
pub mod alignment;
pub mod cohesion;
pub mod flock;
pub mod avoid;
pub mod path;
pub mod flow;
pub mod formation;
pub mod combine;
pub mod spatial;
pub mod config;
pub mod observer;
pub mod error;

// Re-export primary API
pub use float::Float;
pub use vec::{Vec, Scalar, Vec2, Vec3};
pub use agent::Agent;
pub use steering::{SteeringOutput, apply_steering};
pub use seek::{seek, flee, arrive};
pub use pursue::{pursue, evade};
pub use wander::{WanderParams, WanderState, wander_2d, wander_3d};
pub use separation::separation;
pub use alignment::alignment;
pub use cohesion::cohesion;
pub use flock::{FlockWeights, flock};
pub use avoid::{Circle, Aabb, Wall, avoid_circles, avoid_aabbs, avoid_walls};
pub use path::{Path, path_follow};
pub use flow::{FlowField, generate_toward, flow_follow};
pub use formation::{
    FormationSlot, FormationPattern, Formation,
    CircleFormation, VFormation, GridFormation, ColumnFormation, LeaderFollow,
    steer_to_slot,
};
pub use combine::{WeightedBehavior, weighted_blend, priority_select, BehaviorPipeline};
pub use spatial::{NeighborInfo, SpatialQuery, BruteForceQuery};
pub use config::FlockConfig;
pub use observer::{SteeringObserver, NoOpSteeringObserver};
pub use error::SteeringError;
