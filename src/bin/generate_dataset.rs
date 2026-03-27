//! Headless High-Fidelity Dataset Generator for DeepFish
//!
//! This generator is a bit-for-bit logical clone of the Bevy simulation.
//! It uses the exact same Boid steering, Acoustic interference, and
//! Orthographic projection logic as main.rs, but runs 100x faster.
//!
//! Usage:
//! ```bash
//! # Generate easy dataset (default)
//! cargo run --bin generate_dataset -- --output dataset/easy --samples 1000
//!
//! # Generate medium dataset (school independence, some heading changes)
//! cargo run --bin generate_dataset -- --output dataset/medium --samples 1000 --difficulty medium
//!
//! # Generate hard dataset (full independence, frequent heading changes)
//! cargo run --bin generate_dataset -- --output dataset/hard --samples 1000 --difficulty hard
//!
//! # Generate EXTREME dataset (depth randomization - breaks depth shortcuts!)
//! cargo run --bin generate_dataset -- --output dataset/extreme --samples 1000 --difficulty extreme
//! ```

use bevy::math::prelude::*;
use image::{RgbImage, Rgb};
use rand::prelude::*;
use serde::Serialize;
use std::fs;

// --- Constants (Mirrored from main.rs) ---
const TANK_SIZE: Vec3 = Vec3::new(20.0, 10.0, 20.0);
const FISH_COUNT: usize = 400;
const TRANSDUCER_CONE_ANGLE: f32 = 12.0;
const ECHOGRAM_HEIGHT: u32 = 256;
const BIRD_EYE_WIDTH: u32 = 512;
const BIRD_EYE_HEIGHT: u32 = 512;
const PING_HISTORY_LEN: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
enum Species {
    Kingfish,
    Snapper,
    Cod,
    Empty,
}

impl Species {
    fn name(&self) -> &'static str {
        match self {
            Species::Kingfish => "Kingfish",
            Species::Snapper => "Snapper",
            Species::Cod => "Cod",
            Species::Empty => "Empty",
        }
    }

    fn preferred_depth_range(&self) -> (f32, f32) {
        match self {
            Species::Kingfish => (3.0, 5.5),
            Species::Snapper => (2.0, 5.0),
            Species::Cod => (0.5, 3.0),
            Species::Empty => (0.0, 0.0),
        }
    }

    fn color(&self) -> [u8; 3] {
        match self {
            Species::Kingfish => [51, 128, 255], 
            Species::Snapper => [255, 77, 77],   
            Species::Cod => [128, 128, 51],      
            Species::Empty => [0, 0, 0],
        }
    }

    fn scale(&self) -> f32 {
        match self {
            Species::Kingfish => 4.0,
            Species::Snapper => 2.5,
            Species::Cod => 1.8,
            Species::Empty => 0.0,
        }
    }

    fn target_strength(&self) -> (f32, f32, f32) {
        match self {
            Species::Kingfish => (-32.0, -35.0, -38.0),
            Species::Snapper => (-45.0, -43.0, -42.0),
            Species::Cod => (-42.0, -39.0, -37.0),
            Species::Empty => (-99.0, -99.0, -99.0),
        }
    }

    fn speed(&self) -> f32 {
        match self {
            Species::Kingfish => 6.0,
            Species::Snapper => 3.0,
            Species::Cod => 1.5,
            Species::Empty => 0.0,
        }
    }
}

struct Boid {
    pos: Vec3,
    velocity: Vec3,
    acceleration: Vec3,
    species: Species,
    ts_phase: f32,
    max_speed: f32,
    max_force: f32,
    school_id: u32,
    base_heading: Vec3,
}

struct Mesh {
    vertices: Vec<Vec3>,
}

impl Mesh {
    fn load() -> Self {
        let path = "assets/models/fish.glb";
        let mut model_path = std::path::PathBuf::from(path);
        
        // Try to find assets folder by looking up to 3 levels up (for target/release or dataset/ runs)
        if !model_path.exists() {
            for depth in 1..=3 {
                let mut parent_path = std::path::PathBuf::from(".");
                for _ in 0..depth { parent_path.push(".."); }
                parent_path.push(path);
                if parent_path.exists() {
                    model_path = parent_path;
                    break;
                }
            }
        }

        let (gltf, buffers, _) = gltf::import(&model_path)
            .expect(&format!("Failed to load fish model at {:?}", model_path));
        let mut vertices = Vec::new();

        for mesh in gltf.meshes() {
            for primitive in mesh.primitives() {
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                if let Some(positions) = reader.read_positions() {
                    for pos in positions {
                        vertices.push(Vec3::from_slice(&pos));
                    }
                }
            }
        }
        Mesh { vertices }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
enum Difficulty {
    Easy,
    Medium,
    Hard,
    Extreme,  // NEW: Depth randomization
}

struct HeadlessSim {
    boids: Vec<Boid>,
    fish_mesh: Mesh,
    difficulty: Difficulty,
    species_in_beam: std::collections::HashSet<String>,
    species_counts: std::collections::HashMap<String, u32>,
}

impl HeadlessSim {
    fn new(dominant_species: Species, seed: u64, difficulty: Difficulty) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut boids = Vec::with_capacity(FISH_COUNT);
        let fish_mesh = Mesh::load();

        let species_list = [Species::Kingfish, Species::Snapper, Species::Cod];
        let school_count = 16;
        let fish_per_school = FISH_COUNT / school_count;

        for school_idx in 0..school_count {
            // Determine species for this school
            let species = if dominant_species == Species::Empty {
                // Truly empty: no fish spawned for Empty dominant species
                Species::Empty
            } else {
                // 75% chance school is the dominant species
                if rng.gen_bool(0.75) { dominant_species } else { *species_list.choose(&mut rng).unwrap() }
            };

            if species == Species::Empty { continue; }

            let school_id = school_idx as u32;
            
            // Heading depends on difficulty
            let base_heading = match difficulty {
                Difficulty::Easy => {
                    // Easy: Mostly East, with small random offset (+/- 15 degrees)
                    let angle = rng.gen_range(-15.0..15.0f32).to_radians();
                    Vec3::new(angle.cos(), 0.0, angle.sin())
                },
                Difficulty::Medium | Difficulty::Hard | Difficulty::Extreme => {
                    // Medium/Hard/Extreme: Completely random horizontal direction
                    let angle = rng.gen_range(0.0..360.0f32).to_radians();
                    Vec3::new(angle.cos(), 0.0, angle.sin())
                }
            };
            
            for _ in 0..fish_per_school {
                let (min_y, max_y) = species.preferred_depth_range();

                // Add some variety to depth range per school
                let depth_offset = rng.gen_range(-0.5..0.5);
                
                // Extreme mode: randomize depth across full water column
                let actual_y = if matches!(difficulty, Difficulty::Extreme) {
                    rng.gen_range(0.5..TANK_SIZE.y - 0.5) - TANK_SIZE.y / 2.0
                } else {
                    rng.gen_range(min_y..max_y) + depth_offset - TANK_SIZE.y / 2.0
                };
                
                let pos = Vec3::new(
                    rng.gen_range(-TANK_SIZE.x / 2.0..TANK_SIZE.x / 2.0),
                    actual_y,
                    rng.gen_range(-TANK_SIZE.z / 2.0..TANK_SIZE.z / 2.0),
                );
                
                // Add some speed variety (+/- 10%)
                let speed_mult = rng.gen_range(0.9..1.1);
                let speed = species.speed() * speed_mult;

                boids.push(Boid {
                    pos,
                    velocity: base_heading * speed,
                    acceleration: Vec3::ZERO,
                    species,
                    ts_phase: rng.gen_range(0.0..std::f32::consts::TAU),
                    max_speed: speed * 1.2,
                    max_force: 0.1,
                    school_id,
                    base_heading,
                });
            }
        }

        Self { boids, fish_mesh, difficulty, species_in_beam: std::collections::HashSet::new(), species_counts: std::collections::HashMap::new() }
    }

    fn step(&mut self, dt: f32, t: f32) {
        let boid_data: Vec<(Vec3, Vec3, u32)> = self.boids.iter().map(|b| (b.pos, b.velocity, b.school_id)).collect();
        
        // Steering weights depend on difficulty
        let (heading_strength, align_strength, cohere_strength) = match self.difficulty {
            Difficulty::Easy => (0.5, 0.3, 0.2),
            Difficulty::Medium => (0.15, 0.8, 0.5),
            Difficulty::Hard => (0.08, 1.5, 1.0),
            Difficulty::Extreme => (0.05, 2.0, 1.2),  // Even more chaotic
        };

        for i in 0..self.boids.len() {
            let mut separation = Vec3::ZERO;
            let mut alignment = Vec3::ZERO;
            let mut cohesion = Vec3::ZERO;
            let mut neighbor_count = 0;

            let pos = self.boids[i].pos;
            let vel = self.boids[i].velocity;
            let school_id = self.boids[i].school_id;

            for (other_pos, other_vel, other_school) in &boid_data {
                let is_schoolmate = *other_school == school_id;
                let interaction_radius = if is_schoolmate { 7.5 } else { 5.0 };
                let dist = pos.distance(*other_pos);

                if dist > 0.0 && dist < interaction_radius {
                    if dist < 1.5 {
                        separation += (pos - *other_pos).normalize_or_zero() / dist;
                    }
                    let weight = if is_schoolmate { 2.0 } else { 0.5 };
                    alignment += *other_vel * weight;
                    cohesion += *other_pos * weight;
                    neighbor_count += 1;
                }
            }

            let boid = &mut self.boids[i];
            
            // Hard/Extreme difficulty: Periodically change school heading
            if (self.difficulty == Difficulty::Hard || self.difficulty == Difficulty::Extreme) && t > 0.0 && (t % 30.0) < dt {
                let mut rng = thread_rng();
                let angle = rng.gen_range(-20.0..20.0f32).to_radians();
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let old_x = boid.base_heading.x;
                let old_z = boid.base_heading.z;
                boid.base_heading.x = old_x * cos_a - old_z * sin_a;
                boid.base_heading.z = old_x * sin_a + old_z * cos_a;
                
                // Add vertical variety (Hard and Extreme)
                boid.base_heading.y = rng.gen_range(-0.3..0.3f32);
                
                boid.base_heading = boid.base_heading.normalize_or_zero();
            }

            let mut accel = Vec3::ZERO;
            if neighbor_count > 0 {
                let count_f = neighbor_count as f32;
                alignment = (alignment / count_f).normalize_or_zero() * boid.max_speed;
                accel += (alignment - vel).clamp_length_max(boid.max_force) * align_strength;

                cohesion = (cohesion / count_f - pos).normalize_or_zero() * boid.max_speed;
                accel += (cohesion - vel).clamp_length_max(boid.max_force) * cohere_strength;

                if separation.length_squared() > 0.0 {
                    separation = (separation / count_f).normalize_or_zero() * boid.max_speed;
                    accel += (separation - vel).clamp_length_max(boid.max_force * 1.2) * 2.0;
                }
            }

            // School heading preference
            accel += (boid.base_heading - vel.normalize_or_zero()) * heading_strength * boid.max_speed;
            accel += vel.normalize_or_zero() * 0.1; // forward drive

            // Extreme mode: Add random vertical movement to break depth stratification
            if self.difficulty == Difficulty::Extreme {
                let mut rng = thread_rng();
                // Random vertical acceleration (up or down)
                let vertical_jitter = rng.gen_range(-0.5..0.5);
                accel.y += vertical_jitter;
                
                // Check bounds for teleportation (matching main.rs)
                let next_y = boid.pos.y + boid.velocity.y * dt;
                if next_y < -TANK_SIZE.y/2.0 + 0.5 || next_y > TANK_SIZE.y/2.0 - 0.5 {
                    boid.pos.y = rng.gen_range(-TANK_SIZE.y/2.0 + 0.5..TANK_SIZE.y/2.0 - 0.5);
                }
            }

            // Depth keeping (relaxed in Hard mode, disabled in Extreme)
            let (min_y, max_y) = boid.species.preferred_depth_range();
            let target_y = (min_y + max_y) / 2.0 - TANK_SIZE.y / 2.0;
            let depth_error = target_y - pos.y;
            let spring_strength = match self.difficulty {
                Difficulty::Hard => 0.05,
                Difficulty::Extreme => 0.0,
                _ => 0.5,
            };
            accel.y += depth_error * spring_strength;

            boid.acceleration = accel;
            boid.velocity += accel * dt;
            boid.velocity = boid.velocity.clamp_length_max(boid.max_speed);
            boid.pos += boid.velocity * dt;
            
            // Jitter and wrapping
            let jitter_intensity = match boid.species {
                Species::Kingfish => 0.2,
                Species::Snapper => 0.5,
                Species::Cod => 0.1,
                Species::Empty => 0.0,
            };
            boid.pos.y += (t * 2.0 + boid.ts_phase).sin() * jitter_intensity * 0.02;

            if boid.pos.x > TANK_SIZE.x / 2.0 { boid.pos.x = -TANK_SIZE.x / 2.0; }
            if boid.pos.x < -TANK_SIZE.x / 2.0 { boid.pos.x = TANK_SIZE.x / 2.0; }
            if boid.pos.z > TANK_SIZE.z / 2.0 { boid.pos.z = -TANK_SIZE.z / 2.0; }
            if boid.pos.z < -TANK_SIZE.z / 2.0 { boid.pos.z = TANK_SIZE.z / 2.0; }
        }
    }

    fn generate_ping(&mut self, t: f32) -> Vec<u8> {
        let mut ping = vec![0u8; ECHOGRAM_HEIGHT as usize * 3];
        let mut noise_rng = thread_rng();

        for y in 0..ECHOGRAM_HEIGHT as usize {
            let depth_ratio = y as f32 / ECHOGRAM_HEIGHT as f32;
            let noise_mult = 1.0 + 0.3 * (depth_ratio * (1.0 - depth_ratio)).sin().abs();
            let noise = (noise_rng.gen_range(0..12) as f32 * noise_mult) as u8;
            ping[y * 3] = 2 + noise;
            ping[y * 3 + 1] = 5 + noise;
            ping[y * 3 + 2] = 15 + noise;
        }

        let transducer_pos = Vec3::new(0.0, 20.0, 0.0);
        let half_angle_rad = (TRANSDUCER_CONE_ANGLE / 2.0).to_radians();
        let side_lobe_angle = (TRANSDUCER_CONE_ANGLE * 1.5).to_radians();
        
        // Clear and track species in beam for multi-label metadata
        self.species_in_beam.clear();
        self.species_counts.clear();

        for b in &self.boids {
            let to_fish = b.pos - transducer_pos;
            let distance = to_fish.length();
            let direction = to_fish / distance;
            let angle = direction.dot(Vec3::NEG_Y).acos();

            if angle < side_lobe_angle {
                let beam_factor = if angle < half_angle_rad {
                    (1.0 - (angle / half_angle_rad)).powi(2)
                } else {
                    (1.0 - (angle / side_lobe_angle)).powi(2) * 0.1
                };
                
                // Track species in beam and count individuals
                self.species_in_beam.insert(b.species.name().to_string());
                *self.species_counts.entry(b.species.name().to_string()).or_insert(0) += 1;

                let tilt_factor = 1.0 - direction.dot(b.velocity.normalize_or_zero()).abs();

                // EXACT DEPTH RATIO FROM main.rs
                let depth_ratio = (distance / (TANK_SIZE.y * 3.0)).clamp(0.0, 1.0);
                let y_center = (depth_ratio * (ECHOGRAM_HEIGHT as f32 - 1.0)) as i32;

                for dy in -3..=3 {
                    let y = y_center + dy;
                    if y < 0 || y >= ECHOGRAM_HEIGHT as i32 { continue; }
                    let pulse_factor = 1.0 - (dy.abs() as f32 / 3.0);
                    let (ts_38, ts_120, ts_200) = b.species.target_strength();

                    let ts_var_38 = 0.7 + 0.3 * (t * 2.0 + b.ts_phase).sin();
                    let ts_var_120 = 0.8 + 0.2 * (t * 2.5 + b.ts_phase * 1.2).sin();
                    let ts_var_200 = 0.85 + 0.15 * (t * 3.0 + b.ts_phase * 1.4).sin();

                    let calc_i = |ts: f32, var: f32| {
                        ((ts + 65.0) / 45.0).clamp(0.0, 1.0) * beam_factor * pulse_factor * tilt_factor * var
                    };

                    let idx = y as usize * 3;
                    ping[idx] = ping[idx].saturating_add((255.0 * calc_i(ts_38, ts_var_38)) as u8);
                    ping[idx + 1] = ping[idx + 1].saturating_add((255.0 * calc_i(ts_120, ts_var_120)) as u8);
                    ping[idx + 2] = ping[idx + 2].saturating_add((255.0 * calc_i(ts_200, ts_var_200)) as u8);
                }
            }
        }
        ping
    }

    fn render_visual(&self) -> RgbImage {
        let mut img = RgbImage::new(BIRD_EYE_WIDTH, BIRD_EYE_HEIGHT);
        for p in img.pixels_mut() { *p = Rgb([5, 10, 30]); }

        // Render grid lines (matching main.rs)
        let grid_color = Rgb([51, 51, 102]); // srgb(0.2, 0.2, 0.4)
        for i in -5..=5 {
            let x_val = i as f32 * 4.0;
            let z_val = i as f32 * 4.0;
            
            // X-grid lines
            let x_norm = (x_val / 5.0) + 0.5;
            if x_norm >= 0.0 && x_norm < 1.0 {
                let px = (x_norm * BIRD_EYE_WIDTH as f32) as u32;
                for py in 0..BIRD_EYE_HEIGHT {
                    img.put_pixel(px, py, grid_color);
                }
            }
            // Z-grid lines
            let z_norm = (z_val / 5.0) + 0.5;
            if z_norm >= 0.0 && z_norm < 1.0 {
                let py = (z_norm * BIRD_EYE_HEIGHT as f32) as u32;
                for px in 0..BIRD_EYE_WIDTH {
                    img.put_pixel(px, py, grid_color);
                }
            }
        }

        for b in &self.boids {
            let color = b.species.color();
            let scale = b.species.scale();
            let velocity = b.velocity.normalize_or_zero();
            
            let angle = velocity.z.atan2(velocity.x);
            let rotation = Quat::from_rotation_y(-angle);

            for &v in &self.fish_mesh.vertices {
                let world_v = b.pos + (rotation * (v * scale));
                
                // Bird's-eye projection (X-Z plane)
                let x_norm = (world_v.x / 5.0) + 0.5;
                let z_norm = (world_v.z / 5.0) + 0.5;
                
                let px = (x_norm * BIRD_EYE_WIDTH as f32) as i32;
                let py = (z_norm * BIRD_EYE_HEIGHT as f32) as i32;
                
                // Splat vertex onto image with a small 2x2 brush for thickness
                for dx in 0..2 {
                    for dy in 0..2 {
                        let x = px + dx;
                        let y = py + dy;
                        if x >= 0 && x < BIRD_EYE_WIDTH as i32 && y >= 0 && y < BIRD_EYE_HEIGHT as i32 {
                            img.put_pixel(x as u32, y as u32, Rgb(color));
                        }
                    }
                }
            }
        }
        img
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut output_dir = "dataset/easy".to_string();
    let mut samples_per_species = 333;
    let mut difficulty = Difficulty::Easy;
    let mut show_help = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => { show_help = true; }
            "--output" | "-o" => { i += 1; if i < args.len() { output_dir = args[i].clone(); } }
            "--samples" | "-n" => { i += 1; if i < args.len() { samples_per_species = args[i].parse().unwrap_or(333); } }
            "--difficulty" | "-d" => { 
                i += 1; 
                if i < args.len() { 
                    difficulty = match args[i].to_lowercase().as_str() {
                        "easy" => Difficulty::Easy,
                        "medium" => Difficulty::Medium,
                        "hard" => Difficulty::Hard,
                        "extreme" => Difficulty::Extreme,
                        _ => Difficulty::Easy,
                    };
                } 
            }
            "--extreme" => {
                difficulty = Difficulty::Extreme;
            }
            _ => {}
        }
        i += 1;
    }

    if show_help {
        println!("🐟 DeepFish Bit-Fidelity Dataset Generator");
        println!();
        println!("USAGE:");
        println!("    cargo run --bin generate_dataset -- [OPTIONS]");
        println!();
        println!("OPTIONS:");
        println!("    -h, --help                    Show this help message");
        println!("    -o, --output <DIR>            Output directory (default: dataset/easy)");
        println!("    -n, --samples <N>             Samples per species (default: 333)");
        println!("    -d, --difficulty <LEVEL>      Difficulty level: easy, medium, hard, extreme (default: easy)");
        println!();
        println!("DIFFICULTY LEVELS:");
        println!("    easy      - All fish swim east in parallel, no heading changes");
        println!("    medium    - Schools have independent directions, heading changes every ~90s");
        println!("    hard      - Full flocking behavior, heading changes every ~30s");
        println!("    extreme   - DEPTH RANDOMIZATION + chaotic swimming (breaks depth shortcuts!)");
        println!();
        println!("EXAMPLES:");
        println!("    # Generate easy dataset (1000 samples per species)");
        println!("    cargo run --bin generate_dataset -- -o dataset/easy -n 1000");
        println!();
        println!("    # Generate medium dataset");
        println!("    cargo run --bin generate_dataset -- -o dataset/medium -n 1000 -d medium");
        println!();
        println!("    # Generate hard dataset");
        println!("    cargo run --bin generate_dataset -- -o dataset/hard -n 1000 -d hard");
        println!();
        println!("    # Generate EXTREME dataset (depth randomization!)");
        println!("    cargo run --bin generate_dataset -- -o dataset/extreme -n 1000 -d extreme");
        return;
    }

    println!("🐟 DeepFish Bit-Fidelity Dataset Generator");
    println!("   Difficulty: {:?} ({})", difficulty, match difficulty {
        Difficulty::Easy => "All fish swim east, no heading changes",
        Difficulty::Medium => "School independence, heading changes every ~90s",
        Difficulty::Hard => "Full flocking, heading changes every ~30s",
        Difficulty::Extreme => "DEPTH RANDOMIZATION + chaotic swimming!",
    });
    println!("   Output: {}", output_dir);
    println!("   Samples per species: {}", samples_per_species);
    fs::create_dir_all(&output_dir).unwrap();

    let species_list = [Species::Kingfish, Species::Snapper, Species::Cod, Species::Empty];
    let mut frame_count = 0;
    let mut master_rng = StdRng::seed_from_u64(1337);

    for &species in &species_list {
        let n = if species == Species::Empty { samples_per_species / 2 } else { samples_per_species };
        println!("Generating {} frames for {}...", n, species.name());
        
        let mut sim = HeadlessSim::new(species, master_rng.gen(), difficulty);
        
        // Initial warm up (only once per species)
        let dt = 1.0 / 60.0;
        for step in 0..300 { 
            sim.step(dt, step as f32 * dt); 
        }

        let mut t = 300.0 * dt;

        for _ in 0..n {
            frame_count += 1;
            
            // Advance simulation a bit between samples to get variety
            for _ in 0..60 {
                sim.step(dt, t);
                t += dt;
            }

            let mut history = Vec::with_capacity(PING_HISTORY_LEN * ECHOGRAM_HEIGHT as usize * 3);
            for _ in 0..PING_HISTORY_LEN {
                history.extend_from_slice(&sim.generate_ping(t));
                sim.step(dt, t);
                t += dt;
            }

            let id = format!("{:04}", frame_count);
            let mut ac_img = RgbImage::new(PING_HISTORY_LEN as u32, ECHOGRAM_HEIGHT);
            for p in 0..PING_HISTORY_LEN {
                for y in 0..ECHOGRAM_HEIGHT {
                    let idx = (p * ECHOGRAM_HEIGHT as usize * 3) + (y as usize * 3);
                    ac_img.put_pixel(p as u32, y, Rgb([history[idx+0], history[idx+1], history[idx+2]]));
                }
            }
            
            ac_img.save(format!("{}/frame_{}_acoustic.png", &output_dir, id)).unwrap();
            fs::write(format!("{}/frame_{}_history.bin", &output_dir, id), &history).unwrap();
            sim.render_visual().save(format!("{}/frame_{}_visual.png", &output_dir, id)).unwrap();
            
            // Save metadata with multi-label and counting information
            let dominant = species.name();
            let species_array: Vec<String> = sim.species_in_beam.iter()
                .map(|s| format!("\"{}\"", s))
                .collect();
            
            // Build species_counts JSON object
            let counts_obj = if sim.species_counts.is_empty() {
                r#"{"Kingfish": 0, "Snapper": 0, "Cod": 0, "Empty": 0}"#.to_string()
            } else {
                let pairs: Vec<String> = sim.species_counts.iter()
                    .map(|(k, v)| format!("\"{}\": {}", k, v))
                    .collect();
                format!("{{{}}}", pairs.join(", "))
            };
            
            let meta = if species_array.is_empty() {
                // No fish in beam - empty frame
                format!(
                    r#"{{"dominant_species": "Empty", "species_present": ["Empty"], "species_counts": {}}}"#,
                    counts_obj
                )
            } else {
                format!(
                    r#"{{"dominant_species": "{}", "species_present": [{}], "species_counts": {}}}"#,
                    dominant,
                    species_array.join(", "),
                    counts_obj
                )
            };
            fs::write(format!("{}/frame_{}_meta.json", &output_dir, id), meta).unwrap();

            if frame_count % 50 == 0 { println!("  -> {} frames...", frame_count); }
        }
    }
    println!("\n✅ Bit-Fidelity dataset generation complete!");
}
