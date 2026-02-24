# navex Implementation Plan

## What This Is

A composable steering behaviors library for autonomous game agents. Implements Craig Reynolds' classic steering behaviors as pure functions: each behavior takes an agent and context, returns a steering force vector. Compose behaviors via weighted blending or priority selection. Zero-alloc for core behaviors, `no_std` compatible, works in embedded and WASM environments.

Key design: **behaviors are pure functions**. No hidden state, no side effects. An agent struct goes in, a force vector comes out. The caller decides how to apply it.

## Hard Rules

- `#![no_std]` with `extern crate alloc`
- `rand_core` and `libm` are the ONLY dependencies
- All tests: `cargo test --target x86_64-pc-windows-msvc`
- WASM check: `cargo build --target wasm32-unknown-unknown --release`
- Behaviors are PURE FUNCTIONS — no hidden state, no side effects
- Deterministic: same inputs must produce same results (wander uses explicit rng parameter)
- All math is generic over `Float` (f32/f64) — no hardcoded floating point types
- All geometry is generic over `Vec` trait — same behavior works for 2D and 3D

## Reference Implementation

The sibling libraries use identical patterns:
- `../wavfc/` — Wave Function Collapse (original pattern source)
- `../rulebound/` — Constraint propagation solver (same scaffold structure)

Key files to study:
- `../wavfc/src/lib.rs` — Module declarations + re-exports pattern
- `../wavfc/Cargo.toml` — Workspace + dependency pattern
- `../rulebound/demo-wasm/src/lib.rs` — WASM FFI bindings pattern
- `../rulebound/demo-wasm/www/index.html` — Demo page layout

## Implementation Steps

---

### Phase 1: Math Foundations (`float.rs` + `vec.rs`)

These modules provide all the math navex needs, generic over f32/f64 and 2D/3D.

#### `float.rs` — Float Trait

```rust
use core::ops::{Add, Sub, Mul, Div, Neg};

/// Trait abstracting floating-point arithmetic for f32/f64 generics.
pub trait Float:
    Copy
    + Clone
    + PartialEq
    + PartialOrd
    + Default
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + core::fmt::Debug
{
    fn zero() -> Self;
    fn one() -> Self;
    fn half() -> Self;
    fn two() -> Self;
    fn pi() -> Self;
    fn epsilon() -> Self;
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn atan2(self, other: Self) -> Self;
    fn abs(self) -> Self;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn from_f32(v: f32) -> Self;
    fn to_f32(self) -> f32;
    fn recip(self) -> Self;
    fn clamp(self, min: Self, max: Self) -> Self;
}
```

Implement for `f32` and `f64`. Use `core::f32` / `core::f64` intrinsics via `libm` calls. Since we are `no_std`, use the inherent methods on f32/f64 which are available in `core` (sqrt, sin, cos, atan2 are all available as inherent methods on f32/f64 in Rust).

**Important**: For `no_std` compatibility, f32/f64 trig and sqrt require the `libm` feature or linking. In practice, on WASM and most targets these are available as compiler intrinsics. If compilation fails, add `libm = "0.2"` as a dependency and use `libm::sqrtf`, `libm::sinf`, etc. However, try without it first — Rust's inherent float methods delegate to LLVM intrinsics which should work.

#### `vec.rs` — Vector Trait and Concrete Types

```rust
use crate::float::Float;

/// Trait for vector types used in steering calculations.
pub trait Vec: Copy + Clone + Default + core::fmt::Debug {
    type Scalar: Float;

    fn zero() -> Self;
    fn splat(s: Self::Scalar) -> Self;
    fn dot(self, other: Self) -> Self::Scalar;
    fn length_sq(self) -> Self::Scalar;
    fn scale(self, s: Self::Scalar) -> Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn component_mul(self, other: Self) -> Self;
    fn neg(self) -> Self;

    /// Length of the vector.
    fn length(self) -> Self::Scalar {
        self.length_sq().sqrt()
    }

    /// Normalize to unit length. Returns zero vector if length is near zero.
    fn normalize_or_zero(self) -> Self {
        let len = self.length();
        if len > Self::Scalar::epsilon() {
            self.scale(len.recip())
        } else {
            Self::zero()
        }
    }

    /// Normalize to unit length. Panics if zero.
    fn normalize(self) -> Self {
        let len = self.length();
        self.scale(len.recip())
    }

    /// Truncate to max_length if exceeding it.
    fn truncate(self, max_length: Self::Scalar) -> Self {
        let len_sq = self.length_sq();
        if len_sq > max_length * max_length {
            self.normalize_or_zero().scale(max_length)
        } else {
            self
        }
    }

    /// Distance between two points.
    fn distance(self, other: Self) -> Self::Scalar {
        self.sub(other).length()
    }

    /// Distance squared between two points.
    fn distance_sq(self, other: Self) -> Self::Scalar {
        self.sub(other).length_sq()
    }

    /// Linear interpolation: self + (other - self) * t
    fn lerp(self, other: Self, t: Self::Scalar) -> Self {
        self.add(other.sub(self).scale(t))
    }
}
```

##### `Vec2<F: Float>`

```rust
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct Vec2<F: Float> {
    pub x: F,
    pub y: F,
}

impl<F: Float> Vec2<F> {
    pub fn new(x: F, y: F) -> Self { Self { x, y } }

    /// Perpendicular vector (rotate 90 degrees CCW).
    pub fn perp(self) -> Self { Self { x: -self.y, y: self.x } }

    /// Angle of the vector in radians (atan2(y, x)).
    pub fn angle(self) -> F { self.y.atan2(self.x) }

    /// Create unit vector from angle in radians.
    pub fn from_angle(radians: F) -> Self {
        Self { x: radians.cos(), y: radians.sin() }
    }
}

impl<F: Float> Vec for Vec2<F> {
    type Scalar = F;

    fn zero() -> Self { Self { x: F::zero(), y: F::zero() } }
    fn splat(s: F) -> Self { Self { x: s, y: s } }
    fn dot(self, other: Self) -> F { self.x * other.x + self.y * other.y }
    fn length_sq(self) -> F { self.dot(self) }
    fn scale(self, s: F) -> Self { Self { x: self.x * s, y: self.y * s } }
    fn add(self, other: Self) -> Self { Self { x: self.x + other.x, y: self.y + other.y } }
    fn sub(self, other: Self) -> Self { Self { x: self.x - other.x, y: self.y - other.y } }
    fn component_mul(self, other: Self) -> Self { Self { x: self.x * other.x, y: self.y * other.y } }
    fn neg(self) -> Self { Self { x: -self.x, y: -self.y } }
}
```

##### `Vec3<F: Float>`

```rust
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct Vec3<F: Float> {
    pub x: F,
    pub y: F,
    pub z: F,
}

impl<F: Float> Vec3<F> {
    pub fn new(x: F, y: F, z: F) -> Self { Self { x, y, z } }

    /// Cross product.
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

impl<F: Float> Vec for Vec3<F> {
    type Scalar = F;

    fn zero() -> Self { Self { x: F::zero(), y: F::zero(), z: F::zero() } }
    fn splat(s: F) -> Self { Self { x: s, y: s, z: s } }
    fn dot(self, other: Self) -> F { self.x * other.x + self.y * other.y + self.z * other.z }
    fn length_sq(self) -> F { self.dot(self) }
    fn scale(self, s: F) -> Self { Self { x: self.x * s, y: self.y * s, z: self.z * s } }
    fn add(self, other: Self) -> Self { Self { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z } }
    fn sub(self, other: Self) -> Self { Self { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z } }
    fn component_mul(self, other: Self) -> Self { Self { x: self.x * other.x, y: self.y * other.y, z: self.z * other.z } }
    fn neg(self) -> Self { Self { x: -self.x, y: -self.y, z: -self.z } }
}
```

##### `Scalar<F: Float>` (1D wrapper)

```rust
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct Scalar<F: Float>(pub F);

impl<F: Float> Vec for Scalar<F> {
    type Scalar = F;
    fn zero() -> Self { Scalar(F::zero()) }
    fn splat(s: F) -> Self { Scalar(s) }
    fn dot(self, other: Self) -> F { self.0 * other.0 }
    fn length_sq(self) -> F { self.0 * self.0 }
    fn scale(self, s: F) -> Self { Scalar(self.0 * s) }
    fn add(self, other: Self) -> Self { Scalar(self.0 + other.0) }
    fn sub(self, other: Self) -> Self { Scalar(self.0 - other.0) }
    fn component_mul(self, other: Self) -> Self { Scalar(self.0 * other.0) }
    fn neg(self) -> Self { Scalar(-self.0) }
}
```

---

### Phase 2: Core Types (`agent.rs` + `steering.rs`)

#### `agent.rs` — Agent Struct

```rust
use crate::float::Float;
use crate::vec::Vec;

/// An autonomous agent with position, velocity, and movement constraints.
///
/// This is the core input to all steering behaviors. Agents are value types —
/// behaviors return a new agent rather than mutating in place.
#[derive(Copy, Clone, Debug)]
pub struct Agent<V: Vec> {
    /// Current world position.
    pub position: V,
    /// Current velocity vector.
    pub velocity: V,
    /// Agent mass (affects force application). Must be > 0.
    pub mass: V::Scalar,
    /// Maximum speed the agent can travel.
    pub max_speed: V::Scalar,
    /// Maximum steering force that can be applied per frame.
    pub max_force: V::Scalar,
}

impl<V: Vec> Agent<V> {
    /// Create a new agent.
    pub fn new(
        position: V,
        velocity: V,
        mass: V::Scalar,
        max_speed: V::Scalar,
        max_force: V::Scalar,
    ) -> Self {
        Self { position, velocity, mass, max_speed, max_force }
    }

    /// Unit vector in the direction of travel. Returns zero if stationary.
    pub fn heading(&self) -> V {
        self.velocity.normalize_or_zero()
    }

    /// Current speed (magnitude of velocity).
    pub fn speed(&self) -> V::Scalar {
        self.velocity.length()
    }
}
```

#### `steering.rs` — SteeringOutput + Application

```rust
use crate::float::Float;
use crate::vec::Vec;
use crate::agent::Agent;

/// The output of a steering behavior — a force vector to apply to an agent.
///
/// This is intentionally a thin wrapper around a vector. It exists as a
/// distinct type so that the API clearly separates "force to apply" from
/// "position/velocity" vectors.
#[derive(Copy, Clone, Debug)]
pub struct SteeringOutput<V: Vec> {
    /// Linear steering force.
    pub linear: V,
}

impl<V: Vec> SteeringOutput<V> {
    /// Zero steering force (no-op).
    pub fn zero() -> Self {
        Self { linear: V::zero() }
    }

    /// Create from a force vector.
    pub fn new(linear: V) -> Self {
        Self { linear }
    }

    /// Truncate force to maximum magnitude.
    pub fn truncate(self, max: V::Scalar) -> Self {
        Self { linear: self.linear.truncate(max) }
    }

    /// Scale force by a scalar.
    pub fn scale(self, s: V::Scalar) -> Self {
        Self { linear: self.linear.scale(s) }
    }

    /// Add two steering outputs together.
    pub fn add(self, other: Self) -> Self {
        Self { linear: self.linear.add(other.linear) }
    }

    /// Check if force is negligible.
    pub fn is_zero(self, eps: V::Scalar) -> bool {
        self.linear.length_sq() < eps * eps
    }
}

/// Apply a steering force to an agent, returning the updated agent.
///
/// The force is divided by mass (F = ma, so a = F/m), integrated into
/// velocity, clamped to max_speed, then integrated into position.
///
/// # Arguments
/// - `agent` — current agent state
/// - `steering` — force to apply (will be truncated to agent.max_force)
/// - `dt` — time step in seconds
///
/// # Returns
/// Updated agent with new position and velocity.
pub fn apply_steering<V: Vec>(agent: &Agent<V>, steering: &SteeringOutput<V>, dt: V::Scalar) -> Agent<V> {
    // Truncate force to max_force
    let force = steering.linear.truncate(agent.max_force);

    // acceleration = force / mass
    let acceleration = force.scale(agent.mass.recip());

    // velocity += acceleration * dt
    let new_velocity = agent.velocity.add(acceleration.scale(dt));

    // Clamp velocity to max_speed
    let new_velocity = new_velocity.truncate(agent.max_speed);

    // position += velocity * dt
    let new_position = agent.position.add(new_velocity.scale(dt));

    Agent {
        position: new_position,
        velocity: new_velocity,
        mass: agent.mass,
        max_speed: agent.max_speed,
        max_force: agent.max_force,
    }
}
```

---

### Phase 3: Individual Behaviors (`seek.rs` + `pursue.rs` + `wander.rs`)

#### `seek.rs` — Seek, Flee, Arrive

```rust
use crate::vec::Vec;
use crate::agent::Agent;
use crate::steering::SteeringOutput;

/// Seek: steer toward a target position at maximum speed.
///
/// **Algorithm**:
/// ```text
/// desired_velocity = normalize(target - position) * max_speed
/// steering = desired_velocity - velocity
/// ```
///
/// Returns a force that, when applied over time, will turn the agent
/// toward the target. The agent will overshoot and oscillate unless
/// combined with `arrive`.
pub fn seek<V: Vec>(agent: &Agent<V>, target: V) -> SteeringOutput<V> {
    let desired = target.sub(agent.position).normalize_or_zero().scale(agent.max_speed);
    SteeringOutput::new(desired.sub(agent.velocity))
}

/// Flee: steer away from a target position at maximum speed.
///
/// **Algorithm**:
/// ```text
/// desired_velocity = normalize(position - target) * max_speed
/// steering = desired_velocity - velocity
/// ```
///
/// Exact opposite of seek.
pub fn flee<V: Vec>(agent: &Agent<V>, target: V) -> SteeringOutput<V> {
    let desired = agent.position.sub(target).normalize_or_zero().scale(agent.max_speed);
    SteeringOutput::new(desired.sub(agent.velocity))
}

/// Arrive: seek a target but decelerate smoothly within a slowing radius.
///
/// **Algorithm**:
/// ```text
/// offset = target - position
/// distance = length(offset)
/// if distance < epsilon:
///     return zero force (already there)
/// if distance < slowing_radius:
///     speed = max_speed * (distance / slowing_radius)
/// else:
///     speed = max_speed
/// desired_velocity = normalize(offset) * speed
/// steering = desired_velocity - velocity
/// ```
///
/// # Arguments
/// - `slowing_radius` — distance at which the agent begins decelerating.
///   Outside this radius, behaves like `seek`. At the target, speed is zero.
pub fn arrive<V: Vec>(agent: &Agent<V>, target: V, slowing_radius: V::Scalar) -> SteeringOutput<V> {
    let offset = target.sub(agent.position);
    let distance = offset.length();

    if distance < V::Scalar::epsilon() {
        // Already at target — brake
        return SteeringOutput::new(agent.velocity.neg());
    }

    let speed = if distance < slowing_radius {
        agent.max_speed * (distance / slowing_radius)
    } else {
        agent.max_speed
    };

    let desired = offset.normalize_or_zero().scale(speed);
    SteeringOutput::new(desired.sub(agent.velocity))
}
```

#### `pursue.rs` — Pursue and Evade

```rust
use crate::float::Float;
use crate::vec::Vec;
use crate::agent::Agent;
use crate::steering::SteeringOutput;
use crate::seek;

/// Pursue: predict where a moving target will be, then seek that position.
///
/// **Algorithm**:
/// ```text
/// distance = length(target.position - agent.position)
/// closing_speed = agent.max_speed + target.speed()
/// if closing_speed < epsilon:
///     prediction_time = distance / agent.max_speed  // fallback
/// else:
///     prediction_time = distance / closing_speed
/// future_position = target.position + target.velocity * prediction_time
/// return seek(agent, future_position)
/// ```
///
/// The prediction is intentionally simple (linear extrapolation). For more
/// sophisticated prediction, compute the future position yourself and pass
/// it to `seek`.
pub fn pursue<V: Vec>(agent: &Agent<V>, target: &Agent<V>) -> SteeringOutput<V> {
    let offset = target.position.sub(agent.position);
    let distance = offset.length();

    let closing_speed = agent.max_speed + target.speed();
    let prediction_time = if closing_speed > V::Scalar::epsilon() {
        distance / closing_speed
    } else {
        distance / (agent.max_speed + V::Scalar::epsilon())
    };

    let future_position = target.position.add(target.velocity.scale(prediction_time));
    seek::seek(agent, future_position)
}

/// Evade: predict where a moving target will be, then flee from that position.
///
/// **Algorithm**: Same prediction as `pursue`, but `flee` from the predicted position.
pub fn evade<V: Vec>(agent: &Agent<V>, target: &Agent<V>) -> SteeringOutput<V> {
    let offset = target.position.sub(agent.position);
    let distance = offset.length();

    let closing_speed = agent.max_speed + target.speed();
    let prediction_time = if closing_speed > V::Scalar::epsilon() {
        distance / closing_speed
    } else {
        distance / (agent.max_speed + V::Scalar::epsilon())
    };

    let future_position = target.position.add(target.velocity.scale(prediction_time));
    seek::flee(agent, future_position)
}
```

#### `wander.rs` — Wander Behavior

```rust
use crate::float::Float;
use crate::vec::{Vec, Vec2, Vec3};
use crate::agent::Agent;
use crate::steering::SteeringOutput;
use crate::seek;
use rand_core::RngCore;

/// Parameters controlling wander behavior.
#[derive(Copy, Clone, Debug)]
pub struct WanderParams<F: Float> {
    /// Distance of the wander circle/sphere center ahead of the agent.
    pub circle_distance: F,
    /// Radius of the wander circle/sphere.
    pub circle_radius: F,
    /// Maximum angle change per frame (radians). Controls jitter.
    pub angle_change: F,
}

impl<F: Float> WanderParams<F> {
    pub fn new(circle_distance: F, circle_radius: F, angle_change: F) -> Self {
        Self { circle_distance, circle_radius, angle_change }
    }
}

/// Persistent state for wander behavior. Must be kept between frames.
#[derive(Copy, Clone, Debug)]
pub struct WanderState<F: Float> {
    /// Current wander angle (radians). Updated each frame.
    pub wander_angle: F,
}

impl<F: Float> WanderState<F> {
    pub fn new(initial_angle: F) -> Self {
        Self { wander_angle: initial_angle }
    }

    pub fn default_val() -> Self {
        Self { wander_angle: F::zero() }
    }
}

/// Generate a random float in [-1, 1] from an RNG.
fn random_clamped<F: Float>(rng: &mut impl RngCore) -> F {
    // Generate u32, map to [0, 1], then to [-1, 1]
    let val = rng.next_u32();
    let normalized = F::from_f32(val as f32 / u32::MAX as f32);
    normalized * F::two() - F::one()
}

/// 2D wander using Reynolds' circle method.
///
/// **Algorithm**:
/// ```text
/// 1. Place a circle of `circle_radius` at distance `circle_distance`
///    ahead of the agent (along its heading).
/// 2. A point on the circle is determined by `wander_angle`.
/// 3. Each frame, jitter `wander_angle` by a random amount within
///    `[-angle_change, +angle_change]`.
/// 4. The target point is: circle_center + offset_on_circle.
/// 5. Seek that target.
/// ```
///
/// Returns the steering output and the updated wander state (new angle).
pub fn wander_2d<F: Float>(
    agent: &Agent<Vec2<F>>,
    state: &WanderState<F>,
    params: &WanderParams<F>,
    rng: &mut impl RngCore,
) -> (SteeringOutput<Vec2<F>>, WanderState<F>) {
    // Jitter the wander angle
    let jitter: F = random_clamped(rng) * params.angle_change;
    let new_angle = state.wander_angle + jitter;

    // Circle center ahead of the agent
    let heading = agent.heading();
    let circle_center = agent.position.add(heading.scale(params.circle_distance));

    // Displacement on the circle
    let offset = Vec2::from_angle(new_angle).scale(params.circle_radius);

    // Target = circle_center + offset
    let target = circle_center.add(offset);

    let steering = seek::seek(agent, target);
    let new_state = WanderState { wander_angle: new_angle };

    (steering, new_state)
}

/// 3D wander using a sphere instead of a circle.
///
/// Uses two angles (azimuth and elevation) for 3D displacement on a sphere.
/// The `wander_angle` in state is used as the azimuth; a second angle is
/// derived from it for elevation.
///
/// For full 3D wander with independent azimuth/elevation, extend WanderState
/// with a second angle field.
pub fn wander_3d<F: Float>(
    agent: &Agent<Vec3<F>>,
    state: &WanderState<F>,
    params: &WanderParams<F>,
    rng: &mut impl RngCore,
) -> (SteeringOutput<Vec3<F>>, WanderState<F>) {
    let jitter: F = random_clamped(rng) * params.angle_change;
    let new_angle = state.wander_angle + jitter;

    let heading = agent.heading();
    let circle_center = agent.position.add(heading.scale(params.circle_distance));

    // Use angle for azimuth, derive elevation
    let elevation: F = random_clamped(rng) * F::pi() * F::half();
    let cos_elev = elevation.cos();
    let sin_elev = elevation.sin();

    let offset = Vec3::new(
        new_angle.cos() * cos_elev * params.circle_radius,
        sin_elev * params.circle_radius,
        new_angle.sin() * cos_elev * params.circle_radius,
    );

    let target = circle_center.add(offset);
    let steering = seek::seek(agent, target);
    let new_state = WanderState { wander_angle: new_angle };

    (steering, new_state)
}
```

---

### Phase 4: Group Behaviors (`separation.rs` + `alignment.rs` + `cohesion.rs` + `flock.rs`)

#### `separation.rs`

```rust
use crate::float::Float;
use crate::vec::Vec;
use crate::agent::Agent;
use crate::steering::SteeringOutput;

/// Separation: steer away from nearby neighbors to avoid crowding.
///
/// **Algorithm**:
/// ```text
/// force = zero
/// for each neighbor_position in neighbors:
///     offset = agent.position - neighbor_position
///     distance_sq = length_sq(offset)
///     if distance_sq > epsilon:
///         force += normalize(offset) / distance  // closer neighbors push harder
/// return force (as desired velocity change)
/// ```
///
/// The inverse-distance weighting means very close neighbors produce much
/// stronger repulsion than distant ones. This prevents agents from
/// overlapping while allowing loose grouping.
///
/// # Arguments
/// - `neighbors` — iterator of neighbor world positions (NOT including self)
pub fn separation<V: Vec>(agent: &Agent<V>, neighbors: impl Iterator<Item = V>) -> SteeringOutput<V> {
    let mut force = V::zero();
    let mut count = 0u32;

    for neighbor_pos in neighbors {
        let offset = agent.position.sub(neighbor_pos);
        let dist_sq = offset.length_sq();

        if dist_sq > V::Scalar::epsilon() {
            let dist = dist_sq.sqrt();
            // Weight by inverse distance: closer = stronger push
            force = force.add(offset.normalize_or_zero().scale(dist.recip()));
            count += 1;
        }
    }

    if count > 0 {
        // Steer: desired - current
        let desired = force.normalize_or_zero().scale(agent.max_speed);
        SteeringOutput::new(desired.sub(agent.velocity))
    } else {
        SteeringOutput::zero()
    }
}
```

#### `alignment.rs`

```rust
use crate::float::Float;
use crate::vec::Vec;
use crate::agent::Agent;
use crate::steering::SteeringOutput;

/// Alignment: steer to match the average heading of nearby neighbors.
///
/// **Algorithm**:
/// ```text
/// avg_velocity = zero
/// count = 0
/// for each neighbor_velocity in neighbor_velocities:
///     avg_velocity += neighbor_velocity
///     count += 1
/// if count == 0: return zero
/// avg_velocity /= count
/// desired = normalize(avg_velocity) * max_speed
/// steering = desired - velocity
/// ```
///
/// # Arguments
/// - `neighbor_velocities` — iterator of neighbor velocity vectors
pub fn alignment<V: Vec>(agent: &Agent<V>, neighbor_velocities: impl Iterator<Item = V>) -> SteeringOutput<V> {
    let mut sum = V::zero();
    let mut count: u32 = 0;

    for vel in neighbor_velocities {
        sum = sum.add(vel);
        count += 1;
    }

    if count == 0 {
        return SteeringOutput::zero();
    }

    let avg = sum.scale(V::Scalar::one() / V::Scalar::from_f32(count as f32));
    let desired = avg.normalize_or_zero().scale(agent.max_speed);
    SteeringOutput::new(desired.sub(agent.velocity))
}
```

#### `cohesion.rs`

```rust
use crate::float::Float;
use crate::vec::Vec;
use crate::agent::Agent;
use crate::steering::SteeringOutput;
use crate::seek;

/// Cohesion: steer toward the center of mass of nearby neighbors.
///
/// **Algorithm**:
/// ```text
/// center = zero
/// count = 0
/// for each neighbor_position in neighbor_positions:
///     center += neighbor_position
///     count += 1
/// if count == 0: return zero
/// center /= count
/// return seek(agent, center)
/// ```
///
/// # Arguments
/// - `neighbor_positions` — iterator of neighbor world positions
pub fn cohesion<V: Vec>(agent: &Agent<V>, neighbor_positions: impl Iterator<Item = V>) -> SteeringOutput<V> {
    let mut center = V::zero();
    let mut count: u32 = 0;

    for pos in neighbor_positions {
        center = center.add(pos);
        count += 1;
    }

    if count == 0 {
        return SteeringOutput::zero();
    }

    let center = center.scale(V::Scalar::one() / V::Scalar::from_f32(count as f32));
    seek::seek(agent, center)
}
```

#### `flock.rs`

```rust
use crate::float::Float;
use crate::vec::Vec;
use crate::agent::Agent;
use crate::steering::SteeringOutput;
use crate::separation;
use crate::alignment;
use crate::cohesion;

/// Weights for the three flocking components.
#[derive(Copy, Clone, Debug)]
pub struct FlockWeights<F: Float> {
    pub separation: F,
    pub alignment: F,
    pub cohesion: F,
}

impl<F: Float> FlockWeights<F> {
    pub fn new(separation: F, alignment: F, cohesion: F) -> Self {
        Self { separation, alignment, cohesion }
    }

    /// Classic Reynolds weights (1.5, 1.0, 1.0).
    pub fn default_reynolds() -> Self {
        Self {
            separation: F::one() + F::half(),
            alignment: F::one(),
            cohesion: F::one(),
        }
    }
}

/// Combined flocking behavior: separation + alignment + cohesion.
///
/// **Algorithm**:
/// ```text
/// sep = separation(agent, positions) * weights.separation
/// ali = alignment(agent, velocities) * weights.alignment
/// coh = cohesion(agent, positions) * weights.cohesion
/// return sep + ali + coh
/// ```
///
/// # Arguments
/// - `positions` — slice of neighbor world positions
/// - `velocities` — slice of neighbor velocities (same order as positions)
/// - `weights` — how much each component contributes
///
/// Both slices must have the same length. Passing different lengths will
/// use the shorter of the two for alignment/cohesion.
pub fn flock<V: Vec>(
    agent: &Agent<V>,
    positions: &[V],
    velocities: &[V],
    weights: &FlockWeights<V::Scalar>,
) -> SteeringOutput<V> {
    let sep = separation::separation(agent, positions.iter().copied());
    let ali = alignment::alignment(agent, velocities.iter().copied());
    let coh = cohesion::cohesion(agent, positions.iter().copied());

    let combined = sep.scale(weights.separation)
        .add(ali.scale(weights.alignment))
        .add(coh.scale(weights.cohesion));

    combined
}
```

---

### Phase 5: Obstacle Avoidance (`avoid.rs`)

```rust
use crate::float::Float;
use crate::vec::{Vec, Vec2};
use crate::agent::Agent;
use crate::steering::SteeringOutput;

/// A circular obstacle.
#[derive(Copy, Clone, Debug)]
pub struct Circle<V: Vec> {
    pub center: V,
    pub radius: V::Scalar,
}

/// An axis-aligned bounding box.
#[derive(Copy, Clone, Debug)]
pub struct Aabb<V: Vec> {
    pub min: V,
    pub max: V,
}

/// An infinite wall defined by a point and outward normal.
#[derive(Copy, Clone, Debug)]
pub struct Wall<V: Vec> {
    pub point: V,
    pub normal: V, // Must be unit length
}

/// Avoid circular obstacles using ray-circle intersection.
///
/// **Algorithm**:
/// ```text
/// ahead = position + normalize(velocity) * detection_length
/// for each circle obstacle:
///     // Ray-circle intersection test
///     to_center = circle.center - position
///     proj = dot(to_center, heading)  // project center onto ray
///     if proj < 0: skip (behind agent)
///     // Closest point on ray to circle center
///     closest = position + heading * clamp(proj, 0, detection_length)
///     dist = distance(closest, circle.center)
///     if dist < circle.radius:
///         // Collision — steer away laterally
///         penetration = circle.radius - dist
///         avoidance_dir = normalize(closest - circle.center)
///         force += avoidance_dir * (penetration / circle.radius) * max_force
/// return strongest avoidance force
/// ```
pub fn avoid_circles<V: Vec>(
    agent: &Agent<V>,
    obstacles: &[Circle<V>],
    detection_length: V::Scalar,
) -> SteeringOutput<V> {
    let heading = agent.heading();
    let mut force = V::zero();
    let mut closest_dist = detection_length;

    for obs in obstacles {
        let to_center = obs.center.sub(agent.position);
        let proj = to_center.dot(heading);

        // Skip if obstacle is behind agent
        if proj < V::Scalar::zero() {
            continue;
        }

        // Clamp projection to detection range
        let proj_clamped = proj.min(detection_length);

        // Closest point on the detection ray to obstacle center
        let closest_point = agent.position.add(heading.scale(proj_clamped));
        let dist_to_center = closest_point.distance(obs.center);

        if dist_to_center < obs.radius && proj_clamped < closest_dist {
            // This is the nearest threatening obstacle
            closest_dist = proj_clamped;
            let avoidance = closest_point.sub(obs.center).normalize_or_zero();
            let urgency = (obs.radius - dist_to_center) / obs.radius;
            force = avoidance.scale(agent.max_force * urgency);
        }
    }

    SteeringOutput::new(force)
}

/// Avoid AABB obstacles.
///
/// **Algorithm**: Similar to circle avoidance but uses ray-AABB intersection.
/// The avoidance direction points from the AABB center toward the agent.
///
/// **Ray-AABB intersection (slab method)**:
/// For each axis, compute entry and exit distances along the ray.
/// The ray hits the AABB if max(t_entry) < min(t_exit) and the hit is
/// within detection range.
pub fn avoid_aabbs<V: Vec>(
    agent: &Agent<V>,
    obstacles: &[Aabb<V>],
    detection_length: V::Scalar,
) -> SteeringOutput<V> {
    // Note: Full implementation requires per-component access.
    // This stub works for Vec2 by using the dot product approach.
    // The actual implementation needs to check if the ray from
    // agent.position along agent.heading() intersects each AABB
    // within detection_length, then steer away from the nearest hit.

    let heading = agent.heading();
    let mut force = V::zero();
    let _best_t = detection_length;

    for obs in obstacles {
        // Approximate: check if detection ray endpoint is inside AABB
        // or if agent is very close to AABB center
        let aabb_center = obs.min.add(obs.max).scale(V::Scalar::half());
        let to_center = aabb_center.sub(agent.position);
        let proj = to_center.dot(heading);

        if proj > V::Scalar::zero() && proj < detection_length {
            let closest_on_ray = agent.position.add(heading.scale(proj));
            let half_extents = obs.max.sub(obs.min).scale(V::Scalar::half());
            let local = closest_on_ray.sub(aabb_center);

            // Check if point is within AABB (approximate with distance)
            let dist_to_surface = local.length() - half_extents.length();
            if dist_to_surface < V::Scalar::zero() {
                let avoidance = agent.position.sub(aabb_center).normalize_or_zero();
                let urgency = (-dist_to_surface) / half_extents.length();
                force = avoidance.scale(agent.max_force * urgency);
            }
        }
    }

    SteeringOutput::new(force)
}

/// Avoid walls using feeler whiskers.
///
/// **Algorithm**:
/// ```text
/// Create 3 feelers (rays):
///   - ahead: position + heading * feeler_length
///   - left:  position + rotate(heading, +30deg) * feeler_length * 0.7
///   - right: position + rotate(heading, -30deg) * feeler_length * 0.7
///
/// For each feeler, for each wall:
///   signed_dist = dot(feeler_tip - wall.point, wall.normal)
///   if signed_dist < 0:  // feeler penetrates wall
///       force += wall.normal * abs(signed_dist)
/// ```
///
/// This is a 2D-focused implementation. For 3D, extend with additional
/// feelers (up/down).
pub fn avoid_walls<V: Vec>(
    agent: &Agent<V>,
    walls: &[Wall<V>],
    feeler_length: V::Scalar,
) -> SteeringOutput<V> {
    let heading = agent.heading();

    // For simplicity, use a single ahead feeler for the generic version.
    // The Vec2-specific implementation below adds lateral feelers.
    let feeler_tip = agent.position.add(heading.scale(feeler_length));

    let mut force = V::zero();

    for wall in walls {
        let to_tip = feeler_tip.sub(wall.point);
        let signed_dist = to_tip.dot(wall.normal);

        if signed_dist < V::Scalar::zero() {
            // Feeler penetrates wall — push back along normal
            force = force.add(wall.normal.scale(-signed_dist));
        }
    }

    SteeringOutput::new(force)
}

/// 2D wall avoidance with three feeler whiskers (ahead, left 30deg, right 30deg).
pub fn avoid_walls_2d<F: Float>(
    agent: &Agent<Vec2<F>>,
    walls: &[Wall<Vec2<F>>],
    feeler_length: F,
) -> SteeringOutput<Vec2<F>> {
    let heading = agent.heading();
    let short_length = feeler_length * F::from_f32(0.7);

    // Three feelers
    let angle_offset = F::from_f32(core::f32::consts::FRAC_PI_6); // 30 degrees
    let heading_angle = heading.angle();

    let feelers = [
        agent.position.add(heading.scale(feeler_length)),                              // ahead
        agent.position.add(Vec2::from_angle(heading_angle + angle_offset).scale(short_length)), // left
        agent.position.add(Vec2::from_angle(heading_angle - angle_offset).scale(short_length)), // right
    ];

    let mut force = Vec2::zero();

    for feeler_tip in &feelers {
        for wall in walls {
            let to_tip = feeler_tip.sub(wall.point);
            let signed_dist = to_tip.dot(wall.normal);

            if signed_dist < F::zero() {
                force = force.add(wall.normal.scale(-signed_dist));
            }
        }
    }

    SteeringOutput::new(force)
}
```

---

### Phase 6: Spatial Query (`spatial.rs`)

```rust
use alloc::vec::Vec as AllocVec;
use crate::float::Float;
use crate::vec::Vec;

/// Information about a nearby neighbor returned from spatial queries.
#[derive(Copy, Clone, Debug)]
pub struct NeighborInfo<V: Vec> {
    pub position: V,
    pub velocity: V,
    pub distance: V::Scalar,
}

/// Trait for spatial lookups. Users can provide their own spatial index
/// (grid, quadtree, KD-tree) or use the built-in brute-force query.
pub trait SpatialQuery<V: Vec> {
    /// Find all neighbors within `radius` of `center`.
    /// Returns a vector of neighbor info sorted by distance (nearest first).
    fn query_radius(&self, center: V, radius: V::Scalar) -> AllocVec<NeighborInfo<V>>;
}

/// Simple O(n^2) spatial query. Good enough for small agent counts (<200).
pub struct BruteForceQuery<V: Vec> {
    positions: AllocVec<V>,
    velocities: AllocVec<V>,
}

impl<V: Vec> BruteForceQuery<V> {
    pub fn new() -> Self {
        Self {
            positions: AllocVec::new(),
            velocities: AllocVec::new(),
        }
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.positions.clear();
        self.velocities.clear();
    }

    /// Add an agent to the spatial index.
    pub fn insert(&mut self, position: V, velocity: V) {
        self.positions.push(position);
        self.velocities.push(velocity);
    }

    /// Query all agents within `radius` of `center`.
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
        results.sort_by(|a, b| a.distance.to_f32().partial_cmp(&b.distance.to_f32()).unwrap());
        results
    }
}

impl<V: Vec> SpatialQuery<V> for BruteForceQuery<V> {
    fn query_radius(&self, center: V, radius: V::Scalar) -> AllocVec<NeighborInfo<V>> {
        self.query_radius(center, radius)
    }
}
```

---

### Phase 7: Path Following + Flow Fields (`path.rs` + `flow.rs`)

#### `path.rs`

```rust
use alloc::vec::Vec as AllocVec;
use crate::float::Float;
use crate::vec::Vec;
use crate::agent::Agent;
use crate::steering::SteeringOutput;
use crate::seek;

/// A polyline path defined by a series of waypoints.
pub struct Path<V: Vec> {
    /// Waypoints in order.
    pub points: AllocVec<V>,
    /// Cumulative distances along the path (same length as points).
    /// segments[0] = 0, segments[i] = distance from start to points[i].
    pub cumulative: AllocVec<V::Scalar>,
    /// Total path length.
    pub total_length: V::Scalar,
}

impl<V: Vec> Path<V> {
    /// Create a path from waypoints.
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

    /// Find the closest point on the path to `pos`.
    /// Returns (closest_point, parameter) where parameter is distance along path [0, total_length].
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

            // Project pos onto segment, clamp to [0, 1]
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

    /// Get the point at a given parameter (distance along path).
    pub fn point_at(&self, param: V::Scalar) -> V {
        if self.points.len() < 2 {
            return if self.points.is_empty() { V::zero() } else { self.points[0] };
        }

        let param = param.max(V::Scalar::zero()).min(self.total_length);

        // Find which segment this parameter falls on
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

/// Path following: seek a point ahead on the path.
///
/// **Algorithm**:
/// ```text
/// (closest, param) = path.closest_point(agent.position)
/// target_param = param + ahead_distance
/// target = path.point_at(target_param)
/// return seek(agent, target)
/// ```
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
```

#### `flow.rs`

```rust
use alloc::vec::Vec as AllocVec;
use alloc::collections::VecDeque;
use crate::float::Float;
use crate::vec::Vec2;
use crate::agent::Agent;
use crate::steering::SteeringOutput;

/// A 2D grid of direction vectors for navigation.
///
/// Each cell stores a unit vector pointing toward the goal. Agents sample
/// the field at their position and steer in that direction.
pub struct FlowField<F: Float> {
    /// Grid width in cells.
    pub width: usize,
    /// Grid height in cells.
    pub height: usize,
    /// World-space size of each cell.
    pub cell_size: F,
    /// Direction vectors, row-major: directions[y * width + x].
    pub directions: AllocVec<Vec2<F>>,
}

impl<F: Float> FlowField<F> {
    /// Create an empty flow field.
    pub fn new(width: usize, height: usize, cell_size: F) -> Self {
        Self {
            width,
            height,
            cell_size,
            directions: AllocVec::from_iter(core::iter::repeat(Vec2::zero()).take(width * height)),
        }
    }

    /// Set the direction at grid cell (x, y).
    pub fn set(&mut self, x: usize, y: usize, direction: Vec2<F>) {
        if x < self.width && y < self.height {
            self.directions[y * self.width + x] = direction;
        }
    }

    /// Get the direction at grid cell (x, y).
    pub fn get(&self, x: usize, y: usize) -> Vec2<F> {
        if x < self.width && y < self.height {
            self.directions[y * self.width + x]
        } else {
            Vec2::zero()
        }
    }

    /// Sample the flow field at a world position using nearest-neighbor lookup.
    /// For smoother results, use `sample_bilinear`.
    pub fn sample(&self, world_pos: Vec2<F>) -> Vec2<F> {
        let gx = (world_pos.x / self.cell_size).to_f32() as isize;
        let gy = (world_pos.y / self.cell_size).to_f32() as isize;

        if gx >= 0 && gy >= 0 && (gx as usize) < self.width && (gy as usize) < self.height {
            self.get(gx as usize, gy as usize)
        } else {
            Vec2::zero()
        }
    }

    /// Sample with bilinear interpolation for smoother agent movement.
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
            if x >= 0 && y >= 0 && (x as usize) < self.width && (y as usize) < self.height {
                self.get(x as usize, y as usize)
            } else {
                Vec2::zero()
            }
        };

        let d00 = get_safe(x0, y0);
        let d10 = get_safe(x1, y0);
        let d01 = get_safe(x0, y1);
        let d11 = get_safe(x1, y1);

        // Bilinear interpolation
        let one_minus_tx = F::one() - tx;
        let one_minus_ty = F::one() - ty;

        let top = d00.scale(one_minus_tx).add(d10.scale(tx));
        let bottom = d01.scale(one_minus_tx).add(d11.scale(tx));
        let result = top.scale(one_minus_ty).add(bottom.scale(ty));

        result.normalize_or_zero()
    }
}

/// Generate a flow field using BFS from a target position.
///
/// **Algorithm**:
/// ```text
/// 1. Convert target to grid coordinates (tx, ty)
/// 2. Initialize distance grid to MAX for all cells
/// 3. Set distance[ty][tx] = 0
/// 4. BFS from target:
///    - For each neighbor of current cell:
///      - If not blocked and distance > current + 1:
///        - Set distance = current + 1
///        - Set direction = normalize(current_cell - neighbor_cell)
///        - Add to queue
/// 5. Result: every reachable cell has a direction pointing toward the target
/// ```
///
/// # Arguments
/// - `width`, `height` — grid dimensions in cells
/// - `cell_size` — world-space size of each cell
/// - `target` — world-space target position (BFS source)
/// - `blocked` — closure returning true if cell (x, y) is impassable
pub fn generate_toward<F: Float>(
    width: usize,
    height: usize,
    cell_size: F,
    target: Vec2<F>,
    blocked: &dyn Fn(usize, usize) -> bool,
) -> FlowField<F> {
    let mut field = FlowField::new(width, height, cell_size);

    // Convert target to grid coords
    let tx = (target.x / cell_size).to_f32() as isize;
    let ty = (target.y / cell_size).to_f32() as isize;

    if tx < 0 || ty < 0 || tx as usize >= width || ty as usize >= height {
        return field; // Target out of bounds
    }

    let tx = tx as usize;
    let ty = ty as usize;

    // Distance grid (u32::MAX = unvisited)
    let mut dist = AllocVec::from_iter(core::iter::repeat(u32::MAX).take(width * height));

    let idx = |x: usize, y: usize| -> usize { y * width + x };

    // BFS
    let mut queue = VecDeque::new();
    dist[idx(tx, ty)] = 0;
    queue.push_back((tx, ty));

    // 4-directional neighbors (can extend to 8 for diagonals)
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

                // Direction points FROM this cell TOWARD the target (i.e., toward current cell)
                let direction = Vec2::new(
                    F::from_f32(cx as f32 - nx as f32),
                    F::from_f32(cy as f32 - ny as f32),
                ).normalize_or_zero();

                field.set(nx, ny, direction);
                queue.push_back((nx, ny));
            }
        }
    }

    field
}

/// Follow a flow field: sample the field at the agent's position and steer in that direction.
///
/// **Algorithm**:
/// ```text
/// direction = field.sample(agent.position)  // or sample_bilinear
/// desired_velocity = direction * max_speed
/// steering = desired_velocity - velocity
/// ```
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
```

---

### Phase 8: Formations (`formation.rs`)

```rust
use alloc::vec::Vec as AllocVec;
use crate::float::Float;
use crate::vec::{Vec, Vec2};
use crate::agent::Agent;
use crate::steering::SteeringOutput;
use crate::seek;

/// A slot position relative to the formation center.
#[derive(Copy, Clone, Debug)]
pub struct FormationSlot<V: Vec> {
    /// Offset from formation center in local space.
    /// X = right, Y = forward (2D) or Z = forward (3D).
    pub offset: V,
}

/// Trait for formation shape generators.
pub trait FormationPattern<V: Vec> {
    /// Generate slot offsets for `count` agents in local space.
    fn slots(&self, count: usize) -> AllocVec<FormationSlot<V>>;
}

/// Circular formation: agents evenly spaced around a circle.
#[derive(Copy, Clone, Debug)]
pub struct CircleFormation<F: Float> {
    pub radius: F,
}

impl<F: Float> FormationPattern<Vec2<F>> for CircleFormation<F> {
    /// Place agents evenly around a circle of `radius`.
    ///
    /// **Math**: angle_step = 2*PI / count; slot[i] = (cos(i*step)*r, sin(i*step)*r)
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

/// V/chevron formation.
#[derive(Copy, Clone, Debug)]
pub struct VFormation<F: Float> {
    /// Spacing between agents along the V arms.
    pub spacing: F,
    /// Half-angle of the V in radians (e.g., PI/6 for 30 degrees).
    pub angle: F,
}

impl<F: Float> FormationPattern<Vec2<F>> for VFormation<F> {
    /// Place agents in a V shape. Agent 0 is the point.
    /// Subsequent agents alternate left/right along the V arms.
    ///
    /// **Math**:
    /// - Left arm direction: (-sin(angle), -cos(angle))
    /// - Right arm direction: (sin(angle), -cos(angle))
    /// - Agent i (1-indexed): arm = i%2, distance = ceil(i/2) * spacing
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

/// Rectangular grid formation.
#[derive(Copy, Clone, Debug)]
pub struct GridFormation<F: Float> {
    /// Number of columns.
    pub cols: usize,
    /// Spacing between agents.
    pub spacing: F,
}

impl<F: Float> FormationPattern<Vec2<F>> for GridFormation<F> {
    /// Place agents in a rectangular grid, centered on the origin.
    ///
    /// **Math**: row = i / cols, col = i % cols, offset centered.
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

/// Single-file column formation.
#[derive(Copy, Clone, Debug)]
pub struct ColumnFormation<F: Float> {
    /// Spacing between agents.
    pub spacing: F,
}

impl<F: Float> FormationPattern<Vec2<F>> for ColumnFormation<F> {
    /// Place agents in a single column behind the leader.
    ///
    /// **Math**: slot[i] = (0, -i * spacing)
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

/// Leader-follow formation: fixed offset behind leader.
#[derive(Copy, Clone, Debug)]
pub struct LeaderFollow<F: Float> {
    /// Offset behind the leader in local space.
    pub offset: Vec2<F>,
}

impl<F: Float> FormationPattern<Vec2<F>> for LeaderFollow<F> {
    /// All agents get the same offset (they all follow the leader at the same position).
    /// Typically used per-agent with unique offsets — this provides the base pattern.
    fn slots(&self, count: usize) -> AllocVec<FormationSlot<Vec2<F>>> {
        let mut result = AllocVec::with_capacity(count);
        for i in 0..count {
            // Stagger slightly so they don't all stack
            let stagger = F::from_f32(i as f32) * self.offset.y;
            result.push(FormationSlot {
                offset: Vec2::new(self.offset.x, stagger),
            });
        }
        result
    }
}

/// A concrete formation instance with world-space center and heading.
pub struct Formation<V: Vec> {
    /// World-space center of the formation.
    pub center: V,
    /// Heading direction (unit vector) of the formation.
    pub heading: V,
    /// Computed slot offsets in local space.
    pub slots: AllocVec<FormationSlot<V>>,
}

impl<F: Float> Formation<Vec2<F>> {
    /// Create a formation from a pattern.
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

    /// Get the world-space position of a slot.
    ///
    /// **Math**: Rotate the local offset by the formation heading, then translate
    /// by the formation center.
    /// ```text
    /// rotation angle = atan2(heading.y, heading.x)
    /// rotated.x = offset.x * cos(angle) - offset.y * sin(angle)
    /// rotated.y = offset.x * sin(angle) + offset.y * cos(angle)
    /// world = center + rotated
    /// ```
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

/// Steer an agent toward its assigned formation slot using arrive behavior.
pub fn steer_to_slot<F: Float>(
    agent: &Agent<Vec2<F>>,
    formation: &Formation<Vec2<F>>,
    slot_idx: usize,
    slowing_radius: F,
) -> SteeringOutput<Vec2<F>> {
    let target = formation.world_slot_position(slot_idx);
    seek::arrive(agent, target, slowing_radius)
}
```

---

### Phase 9: Behavior Composition (`combine.rs`)

```rust
use alloc::vec::Vec as AllocVec;
use crate::float::Float;
use crate::vec::Vec;
use crate::steering::SteeringOutput;

/// A steering force paired with a weight for blending.
#[derive(Copy, Clone, Debug)]
pub struct WeightedBehavior<V: Vec> {
    pub force: SteeringOutput<V>,
    pub weight: V::Scalar,
}

/// Blend multiple steering behaviors using weighted sum.
///
/// **Algorithm**:
/// ```text
/// result = zero
/// for each (force, weight) in behaviors:
///     result += force * weight
/// result = truncate(result, max_force)
/// ```
///
/// Weights do NOT need to sum to 1.0 — the result is truncated to max_force.
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

/// Priority-based behavior selection: return the first non-negligible force.
///
/// **Algorithm**:
/// ```text
/// for each force in behaviors:
///     if length(force) > threshold:
///         return force
/// return zero
/// ```
///
/// Higher-priority behaviors are listed first. The first behavior that
/// produces a significant force "wins" and all lower-priority behaviors
/// are ignored. Useful for "avoid obstacle OR flock OR wander" priority chains.
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

/// Builder for accumulating weighted behaviors before blending.
pub struct BehaviorPipeline<V: Vec> {
    behaviors: AllocVec<WeightedBehavior<V>>,
    max_force: V::Scalar,
}

impl<V: Vec> BehaviorPipeline<V> {
    /// Create a new pipeline with a maximum output force.
    pub fn new(max_force: V::Scalar) -> Self {
        Self {
            behaviors: AllocVec::new(),
            max_force,
        }
    }

    /// Add a behavior with a weight.
    pub fn add(&mut self, force: SteeringOutput<V>, weight: V::Scalar) -> &mut Self {
        self.behaviors.push(WeightedBehavior { force, weight });
        self
    }

    /// Blend all behaviors using weighted sum, truncated to max_force.
    pub fn blend(&self) -> SteeringOutput<V> {
        weighted_blend(&self.behaviors, self.max_force)
    }

    /// Select the first significant behavior (priority-based).
    pub fn select(&self, threshold: V::Scalar) -> SteeringOutput<V> {
        let forces: AllocVec<SteeringOutput<V>> = self.behaviors.iter().map(|b| b.force.scale(b.weight)).collect();
        priority_select(&forces, threshold)
    }

    /// Clear all behaviors for reuse.
    pub fn clear(&mut self) {
        self.behaviors.clear();
    }
}
```

---

### Phase 10: Infrastructure (`config.rs` + `observer.rs` + `error.rs`)

#### `config.rs`

```rust
use crate::float::Float;
use crate::flock::FlockWeights;

/// Configuration for flocking simulation.
///
/// Use the builder pattern to construct:
/// ```ignore
/// let config = FlockConfig::<f32>::new()
///     .perception_radius(50.0)
///     .separation_radius(25.0)
///     .max_speed(100.0)
///     .max_force(200.0)
///     .weights(FlockWeights::default_reynolds());
/// ```
#[derive(Copy, Clone, Debug)]
pub struct FlockConfig<F: Float> {
    /// Radius within which agents perceive neighbors for alignment/cohesion.
    pub perception_radius: F,
    /// Radius within which agents are repelled (separation). Should be < perception_radius.
    pub separation_radius: F,
    /// Weights for the three flocking components.
    pub weights: FlockWeights<F>,
    /// Maximum agent speed.
    pub max_speed: F,
    /// Maximum steering force.
    pub max_force: F,
}

impl<F: Float> FlockConfig<F> {
    /// Create with sensible defaults.
    pub fn new() -> Self {
        Self {
            perception_radius: F::from_f32(50.0),
            separation_radius: F::from_f32(25.0),
            weights: FlockWeights::default_reynolds(),
            max_speed: F::from_f32(100.0),
            max_force: F::from_f32(200.0),
        }
    }

    pub fn perception_radius(mut self, r: F) -> Self { self.perception_radius = r; self }
    pub fn separation_radius(mut self, r: F) -> Self { self.separation_radius = r; self }
    pub fn weights(mut self, w: FlockWeights<F>) -> Self { self.weights = w; self }
    pub fn max_speed(mut self, s: F) -> Self { self.max_speed = s; self }
    pub fn max_force(mut self, f: F) -> Self { self.max_force = f; self }
}
```

#### `observer.rs`

```rust
use crate::vec::Vec;
use crate::agent::Agent;
use crate::steering::SteeringOutput;

/// Trait for observing steering decisions.
///
/// All methods have default no-op implementations. Implement the ones
/// you care about for debugging, visualization, or logging.
pub trait SteeringObserver<V: Vec> {
    /// Called when a seek/arrive behavior computes a force.
    fn on_seek(&mut self, _agent: &Agent<V>, _target: V, _result: &SteeringOutput<V>) {}

    /// Called when a flock behavior computes a combined force.
    fn on_flock(
        &mut self,
        _agent: &Agent<V>,
        _neighbor_count: usize,
        _result: &SteeringOutput<V>,
    ) {}

    /// Called when an avoidance behavior detects obstacles.
    fn on_avoid(
        &mut self,
        _agent: &Agent<V>,
        _obstacle_count: usize,
        _result: &SteeringOutput<V>,
    ) {}

    /// Called when behaviors are blended.
    fn on_blend(
        &mut self,
        _input_count: usize,
        _result: &SteeringOutput<V>,
    ) {}
}

/// No-op observer that does nothing. Zero overhead when not observing.
pub struct NoOpSteeringObserver;

impl<V: Vec> SteeringObserver<V> for NoOpSteeringObserver {}
```

#### `error.rs`

```rust
use core::fmt;

/// Errors that can occur during steering computations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SteeringError {
    /// Agent has zero velocity where a heading is required.
    ZeroVelocity,
    /// No neighbors provided for a group behavior.
    NoNeighbors,
    /// A weight value is invalid (negative or NaN).
    InvalidWeight,
    /// Path has no waypoints.
    PathEmpty,
    /// Position is outside the flow field bounds.
    FlowFieldOutOfBounds,
}

impl fmt::Display for SteeringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SteeringError::ZeroVelocity => write!(f, "agent has zero velocity"),
            SteeringError::NoNeighbors => write!(f, "no neighbors provided for group behavior"),
            SteeringError::InvalidWeight => write!(f, "invalid weight value"),
            SteeringError::PathEmpty => write!(f, "path has no waypoints"),
            SteeringError::FlowFieldOutOfBounds => write!(f, "position outside flow field bounds"),
        }
    }
}
```

---

### Phase 11: Wire Up `lib.rs`

Update `src/lib.rs` to re-export primary types (after all modules are implemented):

```rust
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
```

---

### Phase 12: Tests

Write these as unit tests in each module and/or integration tests in `tests/`.

#### Individual Behavior Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec::Vec2;

    fn test_agent() -> Agent<Vec2<f32>> {
        Agent::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            1.0,   // mass
            10.0,  // max_speed
            20.0,  // max_force
        )
    }

    #[test]
    fn seek_moves_toward_target() {
        let agent = test_agent();
        let target = Vec2::new(10.0, 0.0);
        let result = seek(&agent, target);
        // Force should point toward target (positive X)
        assert!(result.linear.x > 0.0);
    }

    #[test]
    fn flee_moves_away_from_target() {
        let agent = test_agent();
        let target = Vec2::new(10.0, 0.0);
        let result = flee(&agent, target);
        // Force should point away from target (negative X)
        assert!(result.linear.x < 0.0);
    }

    #[test]
    fn arrive_decelerates_near_target() {
        let mut agent = test_agent();
        agent.position = Vec2::new(8.0, 0.0); // Close to target
        let target = Vec2::new(10.0, 0.0);
        let slowing_radius = 5.0;

        let far_result = seek(&test_agent(), target);
        let near_result = arrive(&agent, target, slowing_radius);

        // Force should be weaker when close (inside slowing radius)
        assert!(near_result.linear.length() < far_result.linear.length());
    }

    #[test]
    fn arrive_at_target_returns_braking_force() {
        let mut agent = test_agent();
        agent.position = Vec2::new(10.0, 0.0);
        let target = Vec2::new(10.0, 0.0);
        let result = arrive(&agent, target, 5.0);
        // Should brake (oppose current velocity)
        assert!(result.linear.x < 0.0);
    }

    #[test]
    fn pursue_intercepts_moving_target() {
        let agent = test_agent();
        let mut target = test_agent();
        target.position = Vec2::new(20.0, 0.0);
        target.velocity = Vec2::new(0.0, 5.0); // Moving up

        let result = pursue(&agent, &target);
        // Should aim ahead of target (positive Y component)
        assert!(result.linear.y > 0.0);
    }

    #[test]
    fn evade_moves_away_from_predicted_position() {
        let agent = test_agent();
        let mut target = test_agent();
        target.position = Vec2::new(5.0, 0.0);
        target.velocity = Vec2::new(0.0, 5.0);

        let result = evade(&agent, &target);
        // Should have a negative X component (moving away)
        assert!(result.linear.x < 0.0 || result.linear.y < 0.0);
    }

    #[test]
    fn wander_stays_near_forward() {
        use rand::SeedableRng;
        use rand::rngs::SmallRng;

        let agent = test_agent();
        let params = WanderParams::new(2.0, 1.0, 0.3);
        let state = WanderState::default_val();
        let mut rng = SmallRng::seed_from_u64(42);

        let (result, _new_state) = wander_2d(&agent, &state, &params, &mut rng);

        // Wander force should not be zero
        assert!(result.linear.length() > 0.0);
    }
}
```

#### Group Behavior Tests

```rust
#[test]
fn separation_pushes_agents_apart() {
    let agent = test_agent();
    let neighbors = vec![Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)];
    let result = separation(&agent, neighbors.into_iter());
    // Force should push away from neighbors
    assert!(result.linear.length() > 0.0);
}

#[test]
fn cohesion_pulls_toward_center() {
    let agent = test_agent();
    let neighbors = vec![Vec2::new(10.0, 0.0), Vec2::new(10.0, 10.0)];
    let result = cohesion(&agent, neighbors.into_iter());
    // Force should pull toward center of neighbors (positive X)
    assert!(result.linear.x > 0.0);
}

#[test]
fn alignment_matches_heading() {
    let agent = test_agent(); // heading (1, 0)
    let neighbor_vels = vec![Vec2::new(0.0, 5.0), Vec2::new(0.0, 3.0)];
    let result = alignment(&agent, neighbor_vels.into_iter());
    // Neighbors moving up, so alignment should push agent upward
    assert!(result.linear.y > 0.0);
}

#[test]
fn flock_stable_over_1000_steps() {
    // 10 agents in a cluster, flock for 1000 steps, verify no divergence
    let weights = FlockWeights::default_reynolds();
    let mut agents: Vec<Agent<Vec2<f32>>> = (0..10)
        .map(|i| Agent::new(
            Vec2::new(i as f32 * 2.0, 0.0),
            Vec2::new(1.0, 0.0),
            1.0, 5.0, 10.0,
        ))
        .collect();

    for _ in 0..1000 {
        let positions: Vec<_> = agents.iter().map(|a| a.position).collect();
        let velocities: Vec<_> = agents.iter().map(|a| a.velocity).collect();

        let mut new_agents = Vec::new();
        for (idx, agent) in agents.iter().enumerate() {
            // Exclude self from neighbors
            let neighbor_pos: Vec<_> = positions.iter().enumerate()
                .filter(|(i, _)| *i != idx).map(|(_, p)| *p).collect();
            let neighbor_vel: Vec<_> = velocities.iter().enumerate()
                .filter(|(i, _)| *i != idx).map(|(_, v)| *v).collect();

            let steering = flock(agent, &neighbor_pos, &neighbor_vel, &weights);
            new_agents.push(apply_steering(agent, &steering, 0.016));
        }
        agents = new_agents;
    }

    // All agents should still be within a reasonable range (not diverged to infinity)
    for agent in &agents {
        assert!(agent.position.length() < 10000.0, "Agent diverged: {:?}", agent.position);
    }
}
```

#### Obstacle Avoidance Tests

```rust
#[test]
fn avoid_steers_around_circle() {
    let agent = Agent::new(
        Vec2::new(0.0, 0.0),
        Vec2::new(1.0, 0.0), // heading right
        1.0, 10.0, 20.0,
    );
    let obstacle = Circle { center: Vec2::new(5.0, 0.0), radius: 2.0 };
    let result = avoid_circles(&agent, &[obstacle], 10.0);
    // Should produce a lateral force (Y component) to steer around
    assert!(result.linear.length() > 0.0);
}
```

#### Flow Field Tests

```rust
#[test]
fn flow_field_bfs_points_toward_target() {
    let field = generate_toward(10, 10, 1.0, Vec2::new(5.0, 5.0), &|_, _| false);
    // Cell (0, 5) should point right (toward target at x=5)
    let dir = field.get(0, 5);
    assert!(dir.x > 0.0);
    // Cell (9, 5) should point left
    let dir = field.get(9, 5);
    assert!(dir.x < 0.0);
}
```

#### Formation Tests

```rust
#[test]
fn circle_formation_slots_equidistant() {
    let pattern = CircleFormation { radius: 10.0f32 };
    let slots = pattern.slots(4);
    assert_eq!(slots.len(), 4);
    // All slots should be at distance `radius` from origin
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
```

#### Composition Tests

```rust
#[test]
fn weighted_blend_respects_weights() {
    let a = SteeringOutput::new(Vec2::new(10.0, 0.0));
    let b = SteeringOutput::new(Vec2::new(0.0, 10.0));

    let result = weighted_blend(&[
        WeightedBehavior { force: a, weight: 1.0 },
        WeightedBehavior { force: b, weight: 0.0 },
    ], 100.0);

    // Only behavior A should contribute
    assert!(result.linear.x > 0.0);
    assert!(result.linear.y.abs() < 0.001);
}

#[test]
fn priority_select_returns_first_significant() {
    let zero = SteeringOutput::<Vec2<f32>>::zero();
    let force = SteeringOutput::new(Vec2::new(5.0, 0.0));

    let result = priority_select(&[zero, force], 0.1);
    assert!(result.linear.x > 0.0); // Should return the second (first significant)
}
```

#### Determinism Test

```rust
#[test]
fn determinism_same_seed_same_result() {
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    let agent = test_agent();
    let params = WanderParams::new(2.0, 1.0, 0.5);

    for _ in 0..100 {
        let state = WanderState::default_val();
        let mut rng1 = SmallRng::seed_from_u64(12345);
        let mut rng2 = SmallRng::seed_from_u64(12345);

        let (r1, s1) = wander_2d(&agent, &state, &params, &mut rng1);
        let (r2, s2) = wander_2d(&agent, &state, &params, &mut rng2);

        assert_eq!(r1.linear.x, r2.linear.x);
        assert_eq!(r1.linear.y, r2.linear.y);
        assert_eq!(s1.wander_angle, s2.wander_angle);
    }
}
```

---

### Phase 13: WASM Demo

The demo should have 5 tabs, each showcasing a different navex capability.

#### Tab 1: Flocking
- 200+ boids rendered as triangles pointing in heading direction
- Three sliders: separation weight (0-3), alignment weight (0-3), cohesion weight (0-3)
- Click canvas to add attractor (left click) or repulsor (right click)
- Toggle button for predator (single agent that boids evade)
- Boids wrap around screen edges

#### Tab 2: Steering
- Single agent (large triangle), click to set target
- Behavior selector: seek, arrive, wander, pursue, evade
- When pursue/evade: a second agent moves in a circle as the target
- Force vectors rendered as colored arrows:
  - Red: seek/arrive force
  - Blue: current velocity
  - Green: resulting heading
- Trail showing agent path

#### Tab 3: Obstacles
- Agents (20) seeking mouse position
- Draggable circle obstacles (click and drag to create/move)
- Draggable AABB obstacles (shift+click)
- Show detection rays as thin lines
- Avoidance force shown as yellow arrows

#### Tab 4: Flow Field
- Grid of direction arrows (colored by distance to target)
- Click to set target position
- BFS regenerates flow field
- 50 agents follow the field
- Toggle blocked cells by clicking with shift held
- Heatmap overlay showing distance

#### Tab 5: Formations
- Drag leader agent (follows mouse)
- 12 follower agents maintain formation
- Formation selector dropdown: Circle, V, Grid, Column
- Sliders: formation radius/spacing, slowing radius
- Show formation slots as ghost markers

#### Implementation pattern:
```rust
// demo-wasm/src/lib.rs
use wasm_bindgen::prelude::*;
use navex::*;

#[wasm_bindgen]
pub struct FlockingDemo {
    agents: Vec<Agent<Vec2<f32>>>,
    weights: FlockWeights<f32>,
    // ...
}

#[wasm_bindgen]
impl FlockingDemo {
    pub fn new(count: usize, width: f32, height: f32) -> Self { ... }
    pub fn set_weights(&mut self, sep: f32, ali: f32, coh: f32) { ... }
    pub fn tick(&mut self, dt: f32) { ... }
    pub fn positions(&self) -> Vec<f32> { ... } // Flat [x, y, x, y, ...] for JS
    pub fn headings(&self) -> Vec<f32> { ... }  // Flat [angle, angle, ...]
}
```

JavaScript side (`demo-wasm/www/main.js`):
```javascript
import init, { FlockingDemo, SteeringDemo, ... } from '../pkg/navex_demo.js';

let canvas, ctx;
let activeTab = 'flocking';
let demos = {};

async function start() {
    await init();
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');

    demos.flocking = FlockingDemo.new(200, canvas.width, canvas.height);
    // ... initialize other demos

    requestAnimationFrame(loop);
}

function loop(timestamp) {
    const dt = /* compute delta */ 0.016;

    switch (activeTab) {
        case 'flocking':
            demos.flocking.tick(dt);
            renderFlocking();
            break;
        // ... other tabs
    }

    requestAnimationFrame(loop);
}

function renderFlocking() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const positions = demos.flocking.positions();
    const headings = demos.flocking.headings();

    for (let i = 0; i < positions.length; i += 2) {
        const x = positions[i];
        const y = positions[i + 1];
        const angle = headings[i / 2];
        drawBoid(ctx, x, y, angle);
    }
}

function drawBoid(ctx, x, y, angle) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.moveTo(6, 0);
    ctx.lineTo(-4, -3);
    ctx.lineTo(-4, 3);
    ctx.closePath();
    ctx.fillStyle = '#4af';
    ctx.fill();
    ctx.restore();
}

start();
```

---

## Algorithm References

### Seek / Flee

```
desired_velocity = normalize(target - position) * max_speed
steering_force = desired_velocity - current_velocity
```

Flee is the same but `normalize(position - target)`.

### Arrive (Deceleration)

```
offset = target - position
distance = length(offset)
if distance < slowing_radius:
    desired_speed = max_speed * (distance / slowing_radius)
else:
    desired_speed = max_speed
desired_velocity = normalize(offset) * desired_speed
steering = desired_velocity - velocity
```

### Pursue (Prediction)

```
to_target = target.position - agent.position
distance = length(to_target)
closing_speed = agent.max_speed + target.speed
prediction_time = distance / closing_speed
future_position = target.position + target.velocity * prediction_time
steering = seek(agent, future_position)
```

### Wander (Circle Method)

```
circle_center = position + normalize(velocity) * circle_distance
wander_angle += random(-angle_change, +angle_change)
offset = Vec2(cos(wander_angle), sin(wander_angle)) * circle_radius
target = circle_center + offset
steering = seek(agent, target)
```

### Reynolds Flocking

```
separation_force = sum( normalize(position - neighbor.position) / distance ) for each neighbor
alignment_force = normalize(average(neighbor.velocity)) * max_speed - velocity
cohesion_force = seek(agent, average(neighbor.position))

total = separation * w_sep + alignment * w_ali + cohesion * w_coh
```

Classic weights: separation=1.5, alignment=1.0, cohesion=1.0

### Ray-Circle Intersection

```
// Ray: origin + t * direction, t >= 0
// Circle: center, radius

oc = origin - center
a = dot(direction, direction)      // = 1 if direction is unit
b = 2 * dot(oc, direction)
c = dot(oc, oc) - radius^2

discriminant = b^2 - 4*a*c
if discriminant < 0: no intersection

t1 = (-b - sqrt(discriminant)) / (2*a)
t2 = (-b + sqrt(discriminant)) / (2*a)

// Nearest intersection: smallest positive t
```

### Ray-AABB Intersection (Slab Method)

```
for each axis (x, y, z):
    if direction[axis] != 0:
        t_min[axis] = (aabb.min[axis] - origin[axis]) / direction[axis]
        t_max[axis] = (aabb.max[axis] - origin[axis]) / direction[axis]
        if t_min[axis] > t_max[axis]: swap them
    else:
        if origin[axis] < aabb.min[axis] or origin[axis] > aabb.max[axis]:
            no intersection

t_enter = max(t_min[x], t_min[y], t_min[z])
t_exit  = min(t_max[x], t_max[y], t_max[z])

if t_enter > t_exit or t_exit < 0: no intersection
hit_t = max(t_enter, 0)
```

### BFS Flow Field Generation

```
function generate_flow_field(grid, target):
    distance[all] = INFINITY
    distance[target] = 0
    queue = [target]

    while queue not empty:
        cell = queue.pop_front()
        for each neighbor of cell:
            if not blocked(neighbor) and distance[cell] + 1 < distance[neighbor]:
                distance[neighbor] = distance[cell] + 1
                direction[neighbor] = normalize(cell - neighbor)  // point toward target
                queue.push_back(neighbor)
```

### Formation Slot Rotation

```
// Rotate local offset by formation heading angle
angle = atan2(heading.y, heading.x)
rotated.x = offset.x * cos(angle) - offset.y * sin(angle)
rotated.y = offset.x * sin(angle) + offset.y * cos(angle)
world_position = formation.center + rotated
```
