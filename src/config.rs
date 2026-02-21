//! Configuration types — FlockConfig and builder pattern.

use crate::float::Float;
use crate::flock::FlockWeights;

/// Builder-style configuration for flocking behavior.
///
/// Provides sensible defaults and a chainable API for adjusting perception
/// radii, flock sub-behavior weights, and movement limits.
#[derive(Copy, Clone, Debug)]
pub struct FlockConfig<F: Float> {
    pub perception_radius: F,
    pub separation_radius: F,
    pub weights: FlockWeights<F>,
    pub max_speed: F,
    pub max_force: F,
}

impl<F: Float> FlockConfig<F> {
    /// Creates a new config with sensible defaults.
    pub fn new() -> Self {
        Self {
            perception_radius: F::from_f32(50.0),
            separation_radius: F::from_f32(25.0),
            weights: FlockWeights::default_reynolds(),
            max_speed: F::from_f32(100.0),
            max_force: F::from_f32(200.0),
        }
    }

    /// Sets the perception radius (how far the agent can see neighbors).
    pub fn perception_radius(mut self, r: F) -> Self {
        self.perception_radius = r;
        self
    }

    /// Sets the separation radius (minimum comfortable distance).
    pub fn separation_radius(mut self, r: F) -> Self {
        self.separation_radius = r;
        self
    }

    /// Sets the flock sub-behavior weights.
    pub fn weights(mut self, w: FlockWeights<F>) -> Self {
        self.weights = w;
        self
    }

    /// Sets the maximum speed.
    pub fn max_speed(mut self, s: F) -> Self {
        self.max_speed = s;
        self
    }

    /// Sets the maximum steering force.
    pub fn max_force(mut self, f: F) -> Self {
        self.max_force = f;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_builder() {
        let config = FlockConfig::<f32>::new()
            .perception_radius(100.0)
            .max_speed(50.0);
        assert!((config.perception_radius - 100.0).abs() < 0.01);
        assert!((config.max_speed - 50.0).abs() < 0.01);
    }
}
