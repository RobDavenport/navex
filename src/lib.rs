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
