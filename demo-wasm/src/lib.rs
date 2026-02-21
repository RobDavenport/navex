//! WASM bindings for the navex interactive demo.
//!
//! Provides 5 demo modes: Flocking, Steering, Obstacles, Flow Field, Formations.

use wasm_bindgen::prelude::*;
use navex::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_core::RngCore;

extern crate alloc;
use alloc::vec::Vec as AllocVec;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn rand_f32(rng: &mut SmallRng) -> f32 {
    rng.next_u32() as f32 / u32::MAX as f32
}

fn rand_range(rng: &mut SmallRng, lo: f32, hi: f32) -> f32 {
    lo + rand_f32(rng) * (hi - lo)
}

// ---------------------------------------------------------------------------
// 1. FlockingDemo
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct FlockingDemo {
    agents: AllocVec<Agent<Vec2<f32>>>,
    weights: FlockWeights<f32>,
    width: f32,
    height: f32,
}

#[wasm_bindgen]
impl FlockingDemo {
    pub fn new(count: usize, width: f32, height: f32) -> Self {
        let mut rng = SmallRng::seed_from_u64(42);
        let agents = (0..count)
            .map(|_| {
                let x = rand_f32(&mut rng) * width;
                let y = rand_f32(&mut rng) * height;
                let vx = rand_f32(&mut rng) * 2.0 - 1.0;
                let vy = rand_f32(&mut rng) * 2.0 - 1.0;
                let vel = Vec2::new(vx, vy).normalize_or_zero().scale(50.0);
                Agent::new(Vec2::new(x, y), vel, 1.0, 100.0, 200.0)
            })
            .collect();
        Self {
            agents,
            weights: FlockWeights::default_reynolds(),
            width,
            height,
        }
    }

    pub fn set_weights(&mut self, sep: f32, ali: f32, coh: f32) {
        self.weights = FlockWeights::new(sep, ali, coh);
    }

    pub fn tick(&mut self, dt: f32) {
        let positions: AllocVec<_> = self.agents.iter().map(|a| a.position).collect();
        let velocities: AllocVec<_> = self.agents.iter().map(|a| a.velocity).collect();

        for i in 0..self.agents.len() {
            let nearby: AllocVec<usize> = (0..positions.len())
                .filter(|j| *j != i && positions[*j].distance(positions[i]) < 80.0)
                .collect();
            let np: AllocVec<_> = nearby.iter().map(|j| positions[*j]).collect();
            let nv: AllocVec<_> = nearby.iter().map(|j| velocities[*j]).collect();

            let steering = flock(&self.agents[i], &np, &nv, &self.weights);
            self.agents[i] = apply_steering(&self.agents[i], &steering, dt);

            // Edge wrapping
            if self.agents[i].position.x < 0.0 {
                self.agents[i].position.x = self.agents[i].position.x + self.width;
            }
            if self.agents[i].position.x > self.width {
                self.agents[i].position.x = self.agents[i].position.x - self.width;
            }
            if self.agents[i].position.y < 0.0 {
                self.agents[i].position.y = self.agents[i].position.y + self.height;
            }
            if self.agents[i].position.y > self.height {
                self.agents[i].position.y = self.agents[i].position.y - self.height;
            }
        }
    }

    pub fn positions(&self) -> AllocVec<f32> {
        self.agents
            .iter()
            .flat_map(|a| [a.position.x, a.position.y])
            .collect()
    }

    pub fn headings(&self) -> AllocVec<f32> {
        self.agents
            .iter()
            .map(|a| a.velocity.y.atan2(a.velocity.x))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// 2. SteeringDemo
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct SteeringDemo {
    agent: Agent<Vec2<f32>>,
    target: Vec2<f32>,
    target_agent: Agent<Vec2<f32>>,
    behavior: u32,
    wander_state: WanderState<f32>,
    wander_params: WanderParams<f32>,
    rng: SmallRng,
    width: f32,
    height: f32,
    trail: AllocVec<f32>,
    last_steering: Vec2<f32>,
    time: f32,
}

#[wasm_bindgen]
impl SteeringDemo {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            agent: Agent::new(
                Vec2::new(width * 0.5, height * 0.5),
                Vec2::new(30.0, 0.0),
                1.0,
                120.0,
                200.0,
            ),
            target: Vec2::new(width * 0.75, height * 0.5),
            target_agent: Agent::new(
                Vec2::new(width * 0.75, height * 0.25),
                Vec2::new(-40.0, 30.0),
                1.0,
                60.0,
                100.0,
            ),
            behavior: 0,
            wander_state: WanderState::default_val(),
            wander_params: WanderParams::new(40.0, 20.0, 0.5),
            rng: SmallRng::seed_from_u64(123),
            width,
            height,
            trail: AllocVec::new(),
            last_steering: Vec2::zero(),
            time: 0.0,
        }
    }

    pub fn set_target(&mut self, x: f32, y: f32) {
        self.target = Vec2::new(x, y);
    }

    pub fn set_behavior(&mut self, id: u32) {
        self.behavior = id;
        self.trail.clear();
    }

    pub fn tick(&mut self, dt: f32) {
        self.time += dt;

        // Move target agent in a circle for pursue/evade
        if self.behavior == 3 || self.behavior == 4 {
            let angle = self.time * 0.8;
            let cx = self.width * 0.5;
            let cy = self.height * 0.5;
            let radius = 150.0;
            let new_pos = Vec2::new(
                cx + radius * angle.cos(),
                cy + radius * angle.sin(),
            );
            let new_vel = Vec2::new(
                -radius * 0.8 * angle.sin(),
                radius * 0.8 * angle.cos(),
            );
            self.target_agent.position = new_pos;
            self.target_agent.velocity = new_vel;
        }

        let steering = match self.behavior {
            0 => seek(&self.agent, self.target),
            1 => arrive(&self.agent, self.target, 100.0),
            2 => {
                let (s, new_state) =
                    wander_2d(&self.agent, &self.wander_state, &self.wander_params, &mut self.rng);
                self.wander_state = new_state;
                s
            }
            3 => pursue(&self.agent, &self.target_agent),
            4 => evade(&self.agent, &self.target_agent),
            _ => SteeringOutput::zero(),
        };

        self.last_steering = steering.linear;
        self.agent = apply_steering(&self.agent, &steering, dt);

        // Edge wrapping
        if self.agent.position.x < 0.0 {
            self.agent.position.x = self.agent.position.x + self.width;
        }
        if self.agent.position.x > self.width {
            self.agent.position.x = self.agent.position.x - self.width;
        }
        if self.agent.position.y < 0.0 {
            self.agent.position.y = self.agent.position.y + self.height;
        }
        if self.agent.position.y > self.height {
            self.agent.position.y = self.agent.position.y - self.height;
        }

        // Record trail (keep last 200 points)
        self.trail.push(self.agent.position.x);
        self.trail.push(self.agent.position.y);
        if self.trail.len() > 400 {
            self.trail.drain(0..2);
        }
    }

    pub fn positions(&self) -> AllocVec<f32> {
        alloc::vec![self.agent.position.x, self.agent.position.y]
    }

    pub fn headings(&self) -> AllocVec<f32> {
        alloc::vec![self.agent.velocity.y.atan2(self.agent.velocity.x)]
    }

    pub fn get_target(&self) -> AllocVec<f32> {
        alloc::vec![self.target.x, self.target.y]
    }

    pub fn trail(&self) -> AllocVec<f32> {
        self.trail.clone()
    }

    pub fn force_vectors(&self) -> AllocVec<f32> {
        alloc::vec![
            self.agent.velocity.x,
            self.agent.velocity.y,
            self.last_steering.x,
            self.last_steering.y
        ]
    }

    pub fn target_agent_pos(&self) -> AllocVec<f32> {
        alloc::vec![
            self.target_agent.position.x,
            self.target_agent.position.y,
            self.target_agent.velocity.x,
            self.target_agent.velocity.y
        ]
    }
}

// ---------------------------------------------------------------------------
// 3. ObstaclesDemo
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct ObstaclesDemo {
    agents: AllocVec<Agent<Vec2<f32>>>,
    circles: AllocVec<Circle<Vec2<f32>>>,
    target: Vec2<f32>,
    width: f32,
    height: f32,
}

#[wasm_bindgen]
impl ObstaclesDemo {
    pub fn new(count: usize, width: f32, height: f32) -> Self {
        let mut rng = SmallRng::seed_from_u64(99);
        let agents = (0..count)
            .map(|_| {
                let x = rand_range(&mut rng, 50.0, width - 50.0);
                let y = rand_range(&mut rng, 50.0, height - 50.0);
                let vx = rand_f32(&mut rng) * 2.0 - 1.0;
                let vy = rand_f32(&mut rng) * 2.0 - 1.0;
                let vel = Vec2::new(vx, vy).normalize_or_zero().scale(40.0);
                Agent::new(Vec2::new(x, y), vel, 1.0, 80.0, 160.0)
            })
            .collect();

        // Add some default obstacles
        let circles = alloc::vec![
            Circle { center: Vec2::new(200.0, 200.0), radius: 40.0 },
            Circle { center: Vec2::new(500.0, 300.0), radius: 50.0 },
            Circle { center: Vec2::new(350.0, 450.0), radius: 35.0 },
            Circle { center: Vec2::new(600.0, 150.0), radius: 45.0 },
        ];

        Self {
            agents,
            circles,
            target: Vec2::new(width * 0.5, height * 0.5),
            width,
            height,
        }
    }

    pub fn set_target(&mut self, x: f32, y: f32) {
        self.target = Vec2::new(x, y);
    }

    pub fn add_circle(&mut self, x: f32, y: f32, r: f32) {
        self.circles.push(Circle {
            center: Vec2::new(x, y),
            radius: r,
        });
    }

    pub fn clear_circles(&mut self) {
        self.circles.clear();
    }

    pub fn tick(&mut self, dt: f32) {
        for i in 0..self.agents.len() {
            let seek_force = seek(&self.agents[i], self.target);
            let avoid_force = avoid_circles(&self.agents[i], &self.circles, 100.0);
            let combined = seek_force.scale(0.6).add(avoid_force.scale(2.0));
            self.agents[i] = apply_steering(&self.agents[i], &combined, dt);

            // Edge wrapping
            if self.agents[i].position.x < 0.0 {
                self.agents[i].position.x = self.agents[i].position.x + self.width;
            }
            if self.agents[i].position.x > self.width {
                self.agents[i].position.x = self.agents[i].position.x - self.width;
            }
            if self.agents[i].position.y < 0.0 {
                self.agents[i].position.y = self.agents[i].position.y + self.height;
            }
            if self.agents[i].position.y > self.height {
                self.agents[i].position.y = self.agents[i].position.y - self.height;
            }
        }
    }

    pub fn positions(&self) -> AllocVec<f32> {
        self.agents
            .iter()
            .flat_map(|a| [a.position.x, a.position.y])
            .collect()
    }

    pub fn headings(&self) -> AllocVec<f32> {
        self.agents
            .iter()
            .map(|a| a.velocity.y.atan2(a.velocity.x))
            .collect()
    }

    pub fn obstacle_data(&self) -> AllocVec<f32> {
        self.circles
            .iter()
            .flat_map(|c| [c.center.x, c.center.y, c.radius])
            .collect()
    }
}

// ---------------------------------------------------------------------------
// 4. FlowFieldDemo
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct FlowFieldDemo {
    field: FlowField<f32>,
    agents: AllocVec<Agent<Vec2<f32>>>,
    blocked: AllocVec<bool>,
    grid_width: usize,
    grid_height: usize,
    cell_size: f32,
    target: Vec2<f32>,
}

#[wasm_bindgen]
impl FlowFieldDemo {
    pub fn new(grid_w: usize, grid_h: usize, cell_size: f32, agent_count: usize) -> Self {
        let mut rng = SmallRng::seed_from_u64(77);
        let world_w = grid_w as f32 * cell_size;
        let world_h = grid_h as f32 * cell_size;
        let agents = (0..agent_count)
            .map(|_| {
                let x = rand_f32(&mut rng) * world_w;
                let y = rand_f32(&mut rng) * world_h;
                Agent::new(Vec2::new(x, y), Vec2::zero(), 1.0, 60.0, 120.0)
            })
            .collect();

        let target = Vec2::new(world_w * 0.5, world_h * 0.5);
        let blocked = AllocVec::from_iter(core::iter::repeat(false).take(grid_w * grid_h));

        let field = generate_toward(grid_w, grid_h, cell_size, target, &|_, _| false);

        Self {
            field,
            agents,
            blocked,
            grid_width: grid_w,
            grid_height: grid_h,
            cell_size,
            target,
        }
    }

    pub fn set_target(&mut self, x: f32, y: f32) {
        self.target = Vec2::new(x, y);
        self.regenerate_field();
    }

    pub fn toggle_blocked(&mut self, gx: usize, gy: usize) {
        if gx < self.grid_width && gy < self.grid_height {
            let idx = gy * self.grid_width + gx;
            self.blocked[idx] = !self.blocked[idx];
            self.regenerate_field();
        }
    }

    fn regenerate_field(&mut self) {
        let blocked = &self.blocked;
        let w = self.grid_width;
        self.field = generate_toward(
            self.grid_width,
            self.grid_height,
            self.cell_size,
            self.target,
            &|x, y| blocked[y * w + x],
        );
    }

    pub fn tick(&mut self, dt: f32) {
        let world_w = self.grid_width as f32 * self.cell_size;
        let world_h = self.grid_height as f32 * self.cell_size;

        for agent in &mut self.agents {
            let steering = flow_follow(agent, &self.field);
            *agent = apply_steering(agent, &steering, dt);

            // Clamp to world bounds
            if agent.position.x < 0.0 {
                agent.position.x = 0.0;
                agent.velocity.x = 0.0;
            }
            if agent.position.x >= world_w {
                agent.position.x = world_w - 0.1;
                agent.velocity.x = 0.0;
            }
            if agent.position.y < 0.0 {
                agent.position.y = 0.0;
                agent.velocity.y = 0.0;
            }
            if agent.position.y >= world_h {
                agent.position.y = world_h - 0.1;
                agent.velocity.y = 0.0;
            }
        }
    }

    pub fn positions(&self) -> AllocVec<f32> {
        self.agents
            .iter()
            .flat_map(|a| [a.position.x, a.position.y])
            .collect()
    }

    pub fn headings(&self) -> AllocVec<f32> {
        self.agents
            .iter()
            .map(|a| a.velocity.y.atan2(a.velocity.x))
            .collect()
    }

    pub fn field_directions(&self) -> AllocVec<f32> {
        self.field
            .directions
            .iter()
            .flat_map(|d| [d.x, d.y])
            .collect()
    }

    pub fn blocked_cells(&self) -> AllocVec<u8> {
        self.blocked.iter().map(|&b| if b { 1 } else { 0 }).collect()
    }

    pub fn grid_w(&self) -> usize {
        self.grid_width
    }

    pub fn grid_h(&self) -> usize {
        self.grid_height
    }

    pub fn cell_sz(&self) -> f32 {
        self.cell_size
    }
}

// ---------------------------------------------------------------------------
// 5. FormationsDemo
// ---------------------------------------------------------------------------

#[wasm_bindgen]
pub struct FormationsDemo {
    leader: Agent<Vec2<f32>>,
    followers: AllocVec<Agent<Vec2<f32>>>,
    leader_target: Vec2<f32>,
    pattern_type: u32,
    formation_radius: f32,
    slowing_radius: f32,
    #[allow(dead_code)]
    width: f32,
    #[allow(dead_code)]
    height: f32,
}

#[wasm_bindgen]
impl FormationsDemo {
    pub fn new(follower_count: usize, width: f32, height: f32) -> Self {
        let leader = Agent::new(
            Vec2::new(width * 0.5, height * 0.5),
            Vec2::new(0.0, 0.0),
            1.0,
            80.0,
            160.0,
        );

        let mut rng = SmallRng::seed_from_u64(55);
        let followers = (0..follower_count)
            .map(|_| {
                let x = rand_range(&mut rng, 100.0, width - 100.0);
                let y = rand_range(&mut rng, 100.0, height - 100.0);
                Agent::new(Vec2::new(x, y), Vec2::zero(), 1.0, 70.0, 140.0)
            })
            .collect();

        Self {
            leader,
            followers,
            leader_target: Vec2::new(width * 0.5, height * 0.5),
            pattern_type: 0,
            formation_radius: 60.0,
            slowing_radius: 40.0,
            width,
            height,
        }
    }

    pub fn set_leader_target(&mut self, x: f32, y: f32) {
        self.leader_target = Vec2::new(x, y);
    }

    pub fn set_pattern(&mut self, id: u32) {
        self.pattern_type = id;
    }

    pub fn set_radius(&mut self, r: f32) {
        self.formation_radius = r;
    }

    pub fn tick(&mut self, dt: f32) {
        // Move leader toward target using arrive
        let leader_steer = arrive(&self.leader, self.leader_target, 80.0);
        self.leader = apply_steering(&self.leader, &leader_steer, dt);

        // Build formation around leader
        let heading = if self.leader.velocity.length() > 1.0 {
            self.leader.velocity
        } else {
            Vec2::new(0.0, -1.0)
        };

        let pattern: &dyn FormationPattern<Vec2<f32>> = match self.pattern_type {
            0 => &CircleFormation {
                radius: self.formation_radius,
            },
            1 => &VFormation {
                spacing: self.formation_radius * 0.5,
                angle: core::f32::consts::FRAC_PI_6,
            },
            2 => &GridFormation {
                cols: 4,
                spacing: self.formation_radius * 0.4,
            },
            3 => &ColumnFormation {
                spacing: self.formation_radius * 0.4,
            },
            _ => &CircleFormation {
                radius: self.formation_radius,
            },
        };

        let formation = Formation::from_pattern(
            self.leader.position,
            heading,
            self.followers.len(),
            pattern,
        );

        for (i, follower) in self.followers.iter_mut().enumerate() {
            let steer = steer_to_slot(follower, &formation, i, self.slowing_radius);
            *follower = apply_steering(follower, &steer, dt);
        }
    }

    pub fn positions(&self) -> AllocVec<f32> {
        let mut out = AllocVec::with_capacity((1 + self.followers.len()) * 2);
        out.push(self.leader.position.x);
        out.push(self.leader.position.y);
        for f in &self.followers {
            out.push(f.position.x);
            out.push(f.position.y);
        }
        out
    }

    pub fn headings(&self) -> AllocVec<f32> {
        let mut out = AllocVec::with_capacity(1 + self.followers.len());
        out.push(self.leader.velocity.y.atan2(self.leader.velocity.x));
        for f in &self.followers {
            out.push(f.velocity.y.atan2(f.velocity.x));
        }
        out
    }

    pub fn slot_positions(&self) -> AllocVec<f32> {
        let heading = if self.leader.velocity.length() > 1.0 {
            self.leader.velocity
        } else {
            Vec2::new(0.0, -1.0)
        };

        let pattern: &dyn FormationPattern<Vec2<f32>> = match self.pattern_type {
            0 => &CircleFormation {
                radius: self.formation_radius,
            },
            1 => &VFormation {
                spacing: self.formation_radius * 0.5,
                angle: core::f32::consts::FRAC_PI_6,
            },
            2 => &GridFormation {
                cols: 4,
                spacing: self.formation_radius * 0.4,
            },
            3 => &ColumnFormation {
                spacing: self.formation_radius * 0.4,
            },
            _ => &CircleFormation {
                radius: self.formation_radius,
            },
        };

        let formation = Formation::from_pattern(
            self.leader.position,
            heading,
            self.followers.len(),
            pattern,
        );

        let mut out = AllocVec::with_capacity(self.followers.len() * 2);
        for i in 0..self.followers.len() {
            let pos = formation.world_slot_position(i);
            out.push(pos.x);
            out.push(pos.y);
        }
        out
    }

    pub fn follower_count(&self) -> usize {
        self.followers.len()
    }
}
