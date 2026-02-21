import init, {
    FlockingDemo,
    SteeringDemo,
    ObstaclesDemo,
    FlowFieldDemo,
    FormationsDemo,
} from '../pkg/navex_demo.js';

let canvas, ctx;
let activeTab = 'flocking';
let demos = {};
let lastTime = 0;
let frameCount = 0;
let fpsTime = 0;
let fps = 0;
let obstacleMode = false;

// Colors
const COL_BOID = '#4af';
const COL_BOID_STROKE = '#28e';
const COL_TARGET = '#f44';
const COL_TRAIL = 'rgba(68, 170, 255, 0.3)';
const COL_OBSTACLE = 'rgba(255, 80, 60, 0.25)';
const COL_OBSTACLE_STROKE = '#f54';
const COL_LEADER = '#fa4';
const COL_FOLLOWER = '#4af';
const COL_SLOT = 'rgba(255, 170, 68, 0.3)';
const COL_FIELD_ARROW = 'rgba(68, 170, 255, 0.25)';
const COL_BLOCKED = 'rgba(255, 50, 50, 0.4)';
const COL_VELOCITY = '#28f';
const COL_STEERING = '#f44';
const COL_TARGET_AGENT = '#4f4';

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function start() {
    await init();
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');

    demos.flocking = FlockingDemo.new(200, canvas.width, canvas.height);
    demos.steering = SteeringDemo.new(canvas.width, canvas.height);
    demos.obstacles = ObstaclesDemo.new(20, canvas.width, canvas.height);
    demos.flowfield = FlowFieldDemo.new(40, 30, 20.0, 50);
    demos.formations = FormationsDemo.new(12, canvas.width, canvas.height);

    setupEventHandlers();
    document.getElementById('stats').textContent = 'Ready';
    lastTime = performance.now();
    requestAnimationFrame(loop);
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

function loop(timestamp) {
    const dt = Math.min((timestamp - lastTime) / 1000, 0.05);
    lastTime = timestamp;

    // FPS counter
    frameCount++;
    fpsTime += dt;
    if (fpsTime >= 0.5) {
        fps = Math.round(frameCount / fpsTime);
        frameCount = 0;
        fpsTime = 0;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    switch (activeTab) {
        case 'flocking':
            tickAndRenderFlocking(dt);
            break;
        case 'steering':
            tickAndRenderSteering(dt);
            break;
        case 'obstacles':
            tickAndRenderObstacles(dt);
            break;
        case 'flowfield':
            tickAndRenderFlowField(dt);
            break;
        case 'formations':
            tickAndRenderFormations(dt);
            break;
    }

    document.getElementById('stats').textContent = `${fps} FPS`;
    requestAnimationFrame(loop);
}

// ---------------------------------------------------------------------------
// Drawing helpers
// ---------------------------------------------------------------------------

function drawBoid(x, y, angle, size, color, strokeColor) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.moveTo(size, 0);
    ctx.lineTo(-size * 0.6, -size * 0.45);
    ctx.lineTo(-size * 0.3, 0);
    ctx.lineTo(-size * 0.6, size * 0.45);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
    if (strokeColor) {
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 0.5;
        ctx.stroke();
    }
    ctx.restore();
}

function drawCircle(x, y, r, fillColor, strokeColor) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    if (fillColor) {
        ctx.fillStyle = fillColor;
        ctx.fill();
    }
    if (strokeColor) {
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }
}

function drawArrow(fromX, fromY, dx, dy, color, maxLen) {
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len < 0.5) return;
    const scale = maxLen ? Math.min(1, maxLen / len) : 1;
    const ex = fromX + dx * scale;
    const ey = fromY + dy * scale;
    const angle = Math.atan2(dy, dx);
    const headLen = 6;

    ctx.beginPath();
    ctx.moveTo(fromX, fromY);
    ctx.lineTo(ex, ey);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(ex, ey);
    ctx.lineTo(
        ex - headLen * Math.cos(angle - 0.4),
        ey - headLen * Math.sin(angle - 0.4)
    );
    ctx.lineTo(
        ex - headLen * Math.cos(angle + 0.4),
        ey - headLen * Math.sin(angle + 0.4)
    );
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.fill();
}

function drawTarget(x, y) {
    drawCircle(x, y, 8, null, COL_TARGET);
    ctx.beginPath();
    ctx.moveTo(x - 12, y);
    ctx.lineTo(x + 12, y);
    ctx.moveTo(x, y - 12);
    ctx.lineTo(x, y + 12);
    ctx.strokeStyle = COL_TARGET;
    ctx.lineWidth = 1;
    ctx.stroke();
}

// ---------------------------------------------------------------------------
// 1. Flocking
// ---------------------------------------------------------------------------

function tickAndRenderFlocking(dt) {
    const demo = demos.flocking;
    demo.tick(dt);

    const pos = demo.positions();
    const hdg = demo.headings();
    const count = hdg.length;

    for (let i = 0; i < count; i++) {
        const x = pos[i * 2];
        const y = pos[i * 2 + 1];
        drawBoid(x, y, hdg[i], 6, COL_BOID, COL_BOID_STROKE);
    }
}

// ---------------------------------------------------------------------------
// 2. Steering
// ---------------------------------------------------------------------------

function tickAndRenderSteering(dt) {
    const demo = demos.steering;
    demo.tick(dt);

    const pos = demo.positions();
    const hdg = demo.headings();
    const target = demo.get_target();
    const trail = demo.trail();
    const forces = demo.force_vectors();
    const behavior = parseInt(document.getElementById('behavior-select').value);

    // Draw trail
    if (trail.length >= 4) {
        ctx.beginPath();
        ctx.moveTo(trail[0], trail[1]);
        for (let i = 2; i < trail.length; i += 2) {
            ctx.lineTo(trail[i], trail[i + 1]);
        }
        ctx.strokeStyle = COL_TRAIL;
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    // Draw target (for seek, arrive)
    if (behavior <= 1) {
        drawTarget(target[0], target[1]);
    }

    // Draw target agent (for pursue/evade)
    if (behavior === 3 || behavior === 4) {
        const ta = demo.target_agent_pos();
        drawBoid(ta[0], ta[1], Math.atan2(ta[3], ta[2]), 10, COL_TARGET_AGENT, '#2a2');

        // Draw predicted position line
        ctx.beginPath();
        ctx.setLineDash([4, 4]);
        ctx.moveTo(ta[0], ta[1]);
        ctx.lineTo(ta[0] + ta[2] * 0.5, ta[1] + ta[3] * 0.5);
        ctx.strokeStyle = 'rgba(68, 255, 68, 0.3)';
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Force vectors
    const ax = pos[0], ay = pos[1];

    // Velocity arrow (blue)
    drawArrow(ax, ay, forces[0] * 0.5, forces[1] * 0.5, COL_VELOCITY, 60);

    // Steering force arrow (red)
    drawArrow(ax, ay, forces[2] * 0.3, forces[3] * 0.3, COL_STEERING, 50);

    // Draw agent
    drawBoid(ax, ay, hdg[0], 10, COL_BOID, COL_BOID_STROKE);
}

// ---------------------------------------------------------------------------
// 3. Obstacles
// ---------------------------------------------------------------------------

function tickAndRenderObstacles(dt) {
    const demo = demos.obstacles;
    demo.tick(dt);

    const pos = demo.positions();
    const hdg = demo.headings();
    const obs = demo.obstacle_data();
    const count = hdg.length;

    // Draw obstacles
    for (let i = 0; i < obs.length; i += 3) {
        drawCircle(obs[i], obs[i + 1], obs[i + 2], COL_OBSTACLE, COL_OBSTACLE_STROKE);
    }

    // Draw target
    // We need to know the target - let's draw a crosshair at last click
    if (demos._obstacleTarget) {
        drawTarget(demos._obstacleTarget[0], demos._obstacleTarget[1]);
    }

    // Draw agents
    for (let i = 0; i < count; i++) {
        const x = pos[i * 2];
        const y = pos[i * 2 + 1];
        drawBoid(x, y, hdg[i], 6, COL_BOID, COL_BOID_STROKE);
    }
}

// ---------------------------------------------------------------------------
// 4. Flow Field
// ---------------------------------------------------------------------------

function tickAndRenderFlowField(dt) {
    const demo = demos.flowfield;
    demo.tick(dt);

    const gridW = demo.grid_w();
    const gridH = demo.grid_h();
    const cellSz = demo.cell_sz();
    const dirs = demo.field_directions();
    const blocked = demo.blocked_cells();
    const pos = demo.positions();
    const hdg = demo.headings();
    const count = hdg.length;

    // Draw grid
    for (let gy = 0; gy < gridH; gy++) {
        for (let gx = 0; gx < gridW; gx++) {
            const idx = gy * gridW + gx;
            const cx = gx * cellSz + cellSz * 0.5;
            const cy = gy * cellSz + cellSz * 0.5;

            if (blocked[idx]) {
                ctx.fillStyle = COL_BLOCKED;
                ctx.fillRect(gx * cellSz, gy * cellSz, cellSz, cellSz);
                continue;
            }

            const dx = dirs[idx * 2];
            const dy = dirs[idx * 2 + 1];
            if (Math.abs(dx) > 0.01 || Math.abs(dy) > 0.01) {
                const arrowLen = cellSz * 0.35;
                const angle = Math.atan2(dy, dx);
                ctx.beginPath();
                ctx.moveTo(cx - Math.cos(angle) * arrowLen, cy - Math.sin(angle) * arrowLen);
                ctx.lineTo(cx + Math.cos(angle) * arrowLen, cy + Math.sin(angle) * arrowLen);
                ctx.strokeStyle = COL_FIELD_ARROW;
                ctx.lineWidth = 1;
                ctx.stroke();

                // Tiny arrowhead
                const headLen = 3;
                ctx.beginPath();
                ctx.moveTo(
                    cx + Math.cos(angle) * arrowLen,
                    cy + Math.sin(angle) * arrowLen
                );
                ctx.lineTo(
                    cx + Math.cos(angle) * arrowLen - headLen * Math.cos(angle - 0.5),
                    cy + Math.sin(angle) * arrowLen - headLen * Math.sin(angle - 0.5)
                );
                ctx.lineTo(
                    cx + Math.cos(angle) * arrowLen - headLen * Math.cos(angle + 0.5),
                    cy + Math.sin(angle) * arrowLen - headLen * Math.sin(angle + 0.5)
                );
                ctx.closePath();
                ctx.fillStyle = COL_FIELD_ARROW;
                ctx.fill();
            }
        }
    }

    // Draw grid lines (very faint)
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 0.5;
    for (let gx = 0; gx <= gridW; gx++) {
        ctx.beginPath();
        ctx.moveTo(gx * cellSz, 0);
        ctx.lineTo(gx * cellSz, gridH * cellSz);
        ctx.stroke();
    }
    for (let gy = 0; gy <= gridH; gy++) {
        ctx.beginPath();
        ctx.moveTo(0, gy * cellSz);
        ctx.lineTo(gridW * cellSz, gy * cellSz);
        ctx.stroke();
    }

    // Draw agents
    for (let i = 0; i < count; i++) {
        const x = pos[i * 2];
        const y = pos[i * 2 + 1];
        drawBoid(x, y, hdg[i], 4, COL_BOID, null);
    }

    // Draw target
    if (demos._flowTarget) {
        drawTarget(demos._flowTarget[0], demos._flowTarget[1]);
    }
}

// ---------------------------------------------------------------------------
// 5. Formations
// ---------------------------------------------------------------------------

function tickAndRenderFormations(dt) {
    const demo = demos.formations;
    demo.tick(dt);

    const pos = demo.positions();
    const hdg = demo.headings();
    const slots = demo.slot_positions();
    const followerCount = demo.follower_count();

    // Draw slot ghosts
    for (let i = 0; i < slots.length; i += 2) {
        drawCircle(slots[i], slots[i + 1], 8, null, COL_SLOT);
    }

    // Draw lines from followers to slots
    for (let i = 0; i < followerCount; i++) {
        const fx = pos[(i + 1) * 2];
        const fy = pos[(i + 1) * 2 + 1];
        const sx = slots[i * 2];
        const sy = slots[i * 2 + 1];
        ctx.beginPath();
        ctx.moveTo(fx, fy);
        ctx.lineTo(sx, sy);
        ctx.strokeStyle = 'rgba(68, 170, 255, 0.1)';
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    // Draw followers
    for (let i = 1; i <= followerCount; i++) {
        const x = pos[i * 2];
        const y = pos[i * 2 + 1];
        drawBoid(x, y, hdg[i], 7, COL_FOLLOWER, COL_BOID_STROKE);
    }

    // Draw leader (larger, different color)
    drawBoid(pos[0], pos[1], hdg[0], 12, COL_LEADER, '#c82');

    // Draw leader target
    if (demos._formationTarget) {
        drawTarget(demos._formationTarget[0], demos._formationTarget[1]);
    }
}

// ---------------------------------------------------------------------------
// Event handlers
// ---------------------------------------------------------------------------

function setupEventHandlers() {
    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeTab = btn.dataset.tab;

            document.querySelectorAll('.control-panel').forEach(p => p.classList.remove('active'));
            const panel = document.getElementById('panel-' + activeTab);
            if (panel) panel.classList.add('active');
        });
    });

    // Canvas click
    canvas.addEventListener('click', (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (canvas.width / rect.width);
        const y = (e.clientY - rect.top) * (canvas.height / rect.height);

        switch (activeTab) {
            case 'steering':
                demos.steering.set_target(x, y);
                break;
            case 'obstacles':
                if (e.shiftKey) {
                    demos.obstacles.add_circle(x, y, 30 + Math.random() * 25);
                } else {
                    demos.obstacles.set_target(x, y);
                    demos._obstacleTarget = [x, y];
                }
                break;
            case 'flowfield':
                if (e.shiftKey) {
                    const demo = demos.flowfield;
                    const gx = Math.floor(x / demo.cell_sz());
                    const gy = Math.floor(y / demo.cell_sz());
                    demo.toggle_blocked(gx, gy);
                } else {
                    demos.flowfield.set_target(x, y);
                    demos._flowTarget = [x, y];
                }
                break;
            case 'formations':
                demos.formations.set_leader_target(x, y);
                demos._formationTarget = [x, y];
                break;
        }
    });

    // Flocking sliders
    const sepSlider = document.getElementById('sep');
    const aliSlider = document.getElementById('ali');
    const cohSlider = document.getElementById('coh');

    function updateFlockWeights() {
        const s = parseFloat(sepSlider.value);
        const a = parseFloat(aliSlider.value);
        const c = parseFloat(cohSlider.value);
        document.getElementById('sep-val').textContent = s.toFixed(1);
        document.getElementById('ali-val').textContent = a.toFixed(1);
        document.getElementById('coh-val').textContent = c.toFixed(1);
        demos.flocking.set_weights(s, a, c);
    }

    sepSlider.addEventListener('input', updateFlockWeights);
    aliSlider.addEventListener('input', updateFlockWeights);
    cohSlider.addEventListener('input', updateFlockWeights);

    // Steering behavior select
    document.getElementById('behavior-select').addEventListener('change', (e) => {
        demos.steering.set_behavior(parseInt(e.target.value));
    });

    // Obstacle buttons
    document.getElementById('add-circle-btn').addEventListener('click', () => {
        const x = 100 + Math.random() * 600;
        const y = 100 + Math.random() * 400;
        const r = 25 + Math.random() * 30;
        demos.obstacles.add_circle(x, y, r);
    });

    document.getElementById('clear-circles-btn').addEventListener('click', () => {
        demos.obstacles.clear_circles();
    });

    // Formation controls
    document.getElementById('pattern-select').addEventListener('change', (e) => {
        demos.formations.set_pattern(parseInt(e.target.value));
    });

    const radiusSlider = document.getElementById('formation-radius');
    radiusSlider.addEventListener('input', () => {
        const r = parseFloat(radiusSlider.value);
        document.getElementById('radius-val').textContent = r.toFixed(0);
        demos.formations.set_radius(r);
    });
}

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------

start().catch(err => {
    document.getElementById('stats').textContent = 'Error: ' + err.message;
    console.error(err);
});
