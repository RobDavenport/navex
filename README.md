# navex

Composable steering behaviors, flocking, flow fields, and formations for games.

`no_std` compatible | MIT / Apache-2.0

**[Live Demo](https://robdavenport.github.io/navex/)**

## Overview

`navex` provides Craig Reynolds-style autonomous agent movement as pure functions. Each behavior takes an agent and context, returning a steering force. Compose behaviors via weighted blending or priority selection.

- Pure functional API: behaviors are stateless functions returning `SteeringOutput`
- Compose any combination of behaviors with weighted blend or priority select
- `no_std` + no allocations in core behaviors (users provide neighbor data)
- Works in embedded, WASM, and native environments

## Quick Start

```rust
use navex::*;

// Create an agent
let agent = Agent::new(
    Vec2::new(0.0, 0.0),   // position
    Vec2::new(1.0, 0.0),   // velocity
    1.0,                     // mass
    100.0,                   // max_speed
    200.0,                   // max_force
);

// Seek a target
let steering = seek(&agent, Vec2::new(100.0, 50.0));
let updated = apply_steering(&agent, &steering, 0.016);

// Flock with neighbors
let weights = FlockWeights::default_reynolds();
let flock_steering = flock(&agent, &neighbor_positions, &neighbor_velocities, &weights);

// Blend multiple behaviors
let blended = weighted_blend(&[
    WeightedBehavior::new(seek_force, 0.6),
    WeightedBehavior::new(avoid_force, 2.0),
]);
```

## Features

- **Individual behaviors**: Seek, flee, arrive, pursue, evade, wander (2D/3D)
- **Group behaviors**: Separation, alignment, cohesion, Reynolds flocking
- **Obstacle avoidance**: Circles, AABBs, walls (single-feeler and 3-feeler 2D)
- **Flow fields**: BFS-generated direction grids with blocked cell support
- **Formations**: Circle, V (chevron), grid, column, leader-follow patterns
- **Composition**: Weighted blend, priority select, behavior pipeline
- **Spatial queries**: `SpatialQuery` trait with brute-force implementation
- **Configuration**: `FlockConfig` for tunable flock parameters
- **Observable**: `SteeringObserver` trait to monitor steering decisions

## API Overview

| Type | Role |
|------|------|
| `Agent<V>` | Position, velocity, mass, and movement limits |
| `SteeringOutput<V>` | Linear force returned by all behaviors |
| `FlockWeights<F>` | Separation, alignment, cohesion weights |
| `FlockConfig<F>` | Full flock configuration (weights + radii) |
| `Circle<V>` / `Aabb<V>` / `Wall<V>` | Obstacle types for avoidance |
| `FlowField<F>` | Grid of direction vectors |
| `Formation<V>` | Computed slot positions around a leader |
| `FormationPattern<V>` | Trait for circle, V, grid, column patterns |
| `WeightedBehavior<V>` | Behavior + weight for blending |
| `BehaviorPipeline<V>` | Priority-ordered behavior composition |
| `SpatialQuery<V>` | Trait for neighbor lookups |
| `SteeringObserver<V>` | Trait for monitoring steering events |

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
