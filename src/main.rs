use bevy::{
    input::mouse::{MouseMotion, MouseWheel},
    prelude::*,
    render::{
        camera::RenderTarget,
        render_resource::{
            Extent3d, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
        },
    },
};
use bevy_egui::{EguiContexts, EguiPlugin, egui};
use rand::prelude::*;
use crossbeam_channel::{Receiver, Sender, unbounded};
use std::path::Path;
use std::thread;

// --- Constants & Config ---
const TANK_SIZE: Vec3 = Vec3::new(20.0, 10.0, 20.0);
const FISH_COUNT: usize = 400;
const SCHOOL_COUNT: usize = 16; // Number of fish schools (~25 fish/school)
const TRANSDUCER_CONE_ANGLE: f32 = 12.0;
const ECHOGRAM_WIDTH: u32 = 512;
const ECHOGRAM_HEIGHT: u32 = 256;
const BIRD_EYE_WIDTH: u32 = 512;
const BIRD_EYE_HEIGHT: u32 = 512;

// Dynamic population control
const POPULATION_CYCLE_PERIOD: f32 = 120.0; // Seconds for full cycle
const MIN_ACTIVE_RATIO: f32 = 0.3;  // Minimum % of schools active
const MAX_ACTIVE_RATIO: f32 = 1.0;  // Maximum % of schools active

#[derive(Resource)]
struct DynamicPopulation {
    pub active_schools: Vec<bool>,  // Which schools are currently active
    pub cycle_timer: f32,           // Time in current cycle
    pub species_activity: [f32; 3], // Activity level per species (0-1)
}

impl Default for DynamicPopulation {
    fn default() -> Self {
        Self {
            active_schools: vec![true; SCHOOL_COUNT],
            cycle_timer: 0.0,
            species_activity: [1.0, 1.0, 1.0],
        }
    }
}

#[derive(Resource)]
struct InferenceChannels {
    pub tx_image: Sender<Vec<u8>>,
    pub rx_preds: Receiver<PredictionResponse>,
}

#[derive(Resource)]
struct InferenceResult {
    pub predictions: Vec<(String, f32)>,
    pub ground_truth_dist: Vec<(String, f32)>, 
    pub correct_count: u32,
    pub total_count: u32,
    pub timer: Timer,
}

impl Default for InferenceResult {
    fn default() -> Self {
        Self {
            predictions: Vec::new(),
            ground_truth_dist: Vec::new(),
            correct_count: 0,
            total_count: 0,
            timer: Timer::from_seconds(0.5, TimerMode::Repeating),
        }
    }
}

// --- Components ---

#[derive(Component, Debug, Clone, Copy, PartialEq)]
enum Species {
    Snapper,
    Kingfish,
    Cod,
    Empty, 
}

impl Species {
    fn preferred_depth_range(&self) -> (f32, f32) {
        match self {
            Species::Kingfish => (7.0, 9.5),
            Species::Snapper => (3.0, 6.5),
            Species::Cod => (0.0, 2.5),
            Species::Empty => (0.0, 0.0),
        }
    }
    fn color(&self) -> Color {
        match self {
            Species::Kingfish => Color::srgb(0.2, 0.5, 1.0),
            Species::Snapper => Color::srgb(1.0, 0.3, 0.3),
            Species::Cod => Color::srgb(0.5, 0.5, 0.2),
            Species::Empty => Color::BLACK,
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
        // Target strengths at 3 frequencies: (38kHz, 120kHz, 200kHz)
        // Different species have different frequency responses
        match self {
            // Kingfish: stronger at lower frequencies (large swim bladder)
            Species::Kingfish => (-32.0, -35.0, -38.0),
            // Snapper: relatively flat response
            Species::Snapper => (-45.0, -43.0, -42.0),
            // Cod: stronger at higher frequencies (different body composition)
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
    fn jitter_intensity(&self) -> f32 {
        match self {
            Species::Kingfish => 0.2,
            Species::Snapper => 0.5,
            Species::Cod => 0.1,
            Species::Empty => 0.0,
        }
    }
}

#[derive(Component)]
struct AcousticProfile {
    target_strength: (f32, f32, f32),  // TS at 3 frequencies: (38kHz, 120kHz, 200kHz)
    ts_phase: f32, // Random phase for time-varying TS
}

#[derive(Component)]
struct Boid {
    velocity: Vec3,
    acceleration: Vec3,
    max_speed: f32,
    max_force: f32,
    phase_offset: f32,
    school_id: u32,
    is_active: bool,  // Whether this fish is currently active
    base_heading: Vec3,  // Base school heading (changes over time)
    heading_change_timer: f32,  // Timer for heading changes
}

#[derive(Component)]
struct MainCamera;

#[derive(Component)]
struct BirdEyeCamera;

#[derive(Component)]
struct Transducer;

#[derive(serde::Deserialize)]
struct PredictionResponse {
    pub predictions: Vec<(String, f32)>,
    pub generated_image: String, 
}

#[derive(Resource)]
struct UIState {
    bird_eye_texture: Handle<Image>,
    echogram_texture: Handle<Image>,
    generated_visual_texture: Handle<Image>,
}

#[derive(Resource)]
struct CameraSettings {
    pub radius: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub center: Vec3,
    pub target_radius: f32,
    pub target_pitch: f32,
    pub target_yaw: f32,
    pub target_center: Vec3,
}

impl Default for CameraSettings {
    fn default() -> Self {
        Self {
            radius: 30.0,
            pitch: 0.5,
            yaw: -0.5,
            center: Vec3::ZERO,
            target_radius: 30.0,
            target_pitch: 0.5,
            target_yaw: -0.5,
            target_center: Vec3::ZERO,
        }
    }
}

use std::collections::VecDeque;

#[derive(Resource)]
struct DatasetExporter {
    pub frame_count: u32,
    pub export_path: String,
    pub is_exporting: bool,
    pub ping_history: VecDeque<Vec<u8>>,
}

impl Default for DatasetExporter {
    fn default() -> Self {
        Self {
            frame_count: 0,
            export_path: "dataset".to_string(),
            is_exporting: false,
            ping_history: VecDeque::with_capacity(32),
        }
    }
}

fn main() {
    let (tx_img, rx_img) = unbounded::<Vec<u8>>();
    let (tx_preds, rx_preds) = unbounded::<PredictionResponse>();

    thread::spawn(move || {
        let client = reqwest::blocking::Client::new();
        while let Ok(png_bytes) = rx_img.recv() {
            let form = reqwest::blocking::multipart::Form::new().part(
                "file",
                reqwest::blocking::multipart::Part::bytes(png_bytes)
                    .file_name("ping.png")
                    .mime_str("image/png")
                    .unwrap(),
            );

            if let Ok(resp) = client
                .post("http://127.0.0.1:8000/predict_acoustic")
                .multipart(form)
                .send()
                .and_then(|resp| resp.json::<PredictionResponse>())
            {
                let _ = tx_preds.send(resp);
            }
        }
    });

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(EguiPlugin)
        .insert_resource(ClearColor(Color::BLACK))
        .insert_resource(InferenceChannels {
            tx_image: tx_img,
            rx_preds,
        })
        .init_resource::<CameraSettings>()
        .init_resource::<DatasetExporter>()
        .init_resource::<InferenceResult>()
        .init_resource::<DynamicPopulation>()
        .add_systems(
            Startup,
            (setup_render_textures, setup_scene, setup_cameras).chain(),
        )
        .add_systems(
            Update,
            (
                boid_movement_system,
                boundary_wrapping_system,
                echosounder_ping_system,
                camera_controller_system,
                ui_tiled_windows_system,
                dataset_exporter_system,
                model_inference_system,
                set_camera_viewports,
                tint_fish_system,
                sync_cpu_buffer_system,
                update_population_system,  // NEW: Dynamic population control
                update_school_headings_system,  // NEW: Gradual heading changes
            ),
        )
        .run();
}

fn setup_render_textures(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let echo_size = Extent3d {
        width: ECHOGRAM_WIDTH,
        height: ECHOGRAM_HEIGHT,
        ..default()
    };
    let mut echo_data = vec![0; (ECHOGRAM_WIDTH * ECHOGRAM_HEIGHT * 4) as usize];
    for y in 0..ECHOGRAM_HEIGHT {
        for x in 0..ECHOGRAM_WIDTH {
            let i = ((y * ECHOGRAM_WIDTH + x) * 4) as usize;
            echo_data[i] = 0;
            echo_data[i + 1] = 5;
            echo_data[i + 2] = 20;
            echo_data[i + 3] = 255;
        }
    }
    let echo_image = Image {
        texture_descriptor: TextureDescriptor {
            label: Some("Echogram Texture"),
            size: echo_size,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        },
        data: echo_data,
        ..default()
    };
    let echo_handle = images.add(echo_image);

    let bird_size = Extent3d {
        width: BIRD_EYE_WIDTH,
        height: BIRD_EYE_HEIGHT,
        ..default()
    };
    let bird_image = Image {
        texture_descriptor: TextureDescriptor {
            label: Some("Bird Eye Texture"),
            size: bird_size,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::COPY_SRC,
            view_formats: &[],
        },
        data: vec![0; (BIRD_EYE_WIDTH * BIRD_EYE_HEIGHT * 4) as usize],
        ..default()
    };
    let bird_handle = images.add(bird_image);

    let gen_image = Image::new_fill(
        Extent3d {
            width: 224,
            height: 224,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        bevy::render::render_asset::RenderAssetUsages::default(),
    );
    let gen_handle = images.add(gen_image);

    commands.insert_resource(UIState {
        bird_eye_texture: bird_handle,
        echogram_texture: echo_handle,
        generated_visual_texture: gen_handle,
    });
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    asset_server: Res<AssetServer>,
) {
    for i in -5..=5 {
        let x = i as f32 * 4.0;
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(0.05, 0.05, TANK_SIZE.z))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.2, 0.2, 0.4),
                unlit: true,
                ..default()
            })),
            Transform::from_xyz(x, -TANK_SIZE.y / 2.0, 0.0),
        ));
        let z = i as f32 * 4.0;
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(TANK_SIZE.x, 0.05, 0.05))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::srgb(0.2, 0.2, 0.4),
                unlit: true,
                ..default()
            })),
            Transform::from_xyz(0.0, -TANK_SIZE.y / 2.0, z),
        ));
    }

    let fish_handle = asset_server.load("models/fish.glb#Scene0");

    // Create schools with balanced species distribution
    let mut rng = thread_rng();
    let mut school_centers: Vec<(Vec3, Species)> = Vec::with_capacity(SCHOOL_COUNT);
    
    // Equal schools per species for balanced dataset
    let species_list = [Species::Kingfish, Species::Snapper, Species::Cod];
    for i in 0..SCHOOL_COUNT {
        let species = species_list[i % 3];  // Cycle through species
        let (min_y, max_y) = species.preferred_depth_range();
        let center = Vec3::new(
            rng.gen_range(-TANK_SIZE.x / 2.0 + 2.0..TANK_SIZE.x / 2.0 - 2.0),
            rng.gen_range(min_y..max_y) - TANK_SIZE.y / 2.0,
            rng.gen_range(-TANK_SIZE.z / 2.0 + 2.0..TANK_SIZE.z / 2.0 - 2.0),
        );
        school_centers.push((center, species));
    }
    school_centers.shuffle(&mut rng);

    // Distribute fish among schools
    let fish_per_school = FISH_COUNT / SCHOOL_COUNT;
    
    for (school_id, (school_center, species)) in school_centers.iter().enumerate() {
        for _ in 0..fish_per_school {
            // Cluster fish around school center
            let spread = 2.5;
            let pos = Vec3::new(
                school_center.x + rng.gen_range(-spread..spread),
                (school_center.y + rng.gen_range(-0.8..0.8)).clamp(-TANK_SIZE.y/2.0, TANK_SIZE.y/2.0),
                school_center.z + rng.gen_range(-spread..spread),
            );

            let speed = species.speed();
            let school_direction = Vec3::new(rng.gen_range(-1.0..1.0), 0.0, rng.gen_range(-1.0..1.0)).normalize();
            
            commands.spawn((
                SceneRoot(fish_handle.clone()),
                Transform::from_translation(pos).with_scale(Vec3::splat(species.scale())),
                *species,
                AcousticProfile {
                    target_strength: species.target_strength(),
                    ts_phase: rng.gen_range(0.0..std::f32::consts::TAU),
                },
                Boid {
                    velocity: school_direction * speed * rng.gen_range(0.9..1.1),
                    acceleration: Vec3::ZERO,
                    max_speed: speed * 1.2,
                    max_force: 0.1,
                    phase_offset: rng.gen_range(0.0..std::f32::consts::TAU),
                    school_id: school_id as u32,
                    is_active: true,
                    base_heading: school_direction,
                    heading_change_timer: rng.gen_range(0.0..30.0),  // Stagger heading changes
                },
            ));
        }
    }

    commands.spawn((
        SpotLight {
            intensity: 5_000_000.0,
            range: 50.0,
            inner_angle: 0.0,
            outer_angle: (TRANSDUCER_CONE_ANGLE / 2.0).to_radians(),
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 15.0, 0.0).looking_at(Vec3::ZERO, Vec3::NEG_Z),
    ));

    commands.insert_resource(AmbientLight {
        color: Color::srgb(0.0, 0.0, 0.05),
        brightness: 5.0,
    });
}

fn setup_cameras(mut commands: Commands, ui_state: Res<UIState>) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 10.0, 25.0).looking_at(Vec3::ZERO, Vec3::Y),
        MainCamera,
    ));

    commands.spawn((
        Camera3d::default(),
        Projection::Orthographic(OrthographicProjection {
            scale: 5.0,
            scaling_mode: bevy::render::camera::ScalingMode::FixedVertical {
                viewport_height: 1.0,
            },
            ..OrthographicProjection::default_3d()
        }),
        Transform::from_xyz(0.0, 20.0, 0.0).looking_at(Vec3::ZERO, Vec3::NEG_Z),
        Camera {
            target: RenderTarget::Image(ui_state.bird_eye_texture.clone()),
            ..default()
        },
        BirdEyeCamera,
        Transducer,
    ));
}

fn ui_tiled_windows_system(
    mut contexts: EguiContexts,
    ui_state: Res<UIState>,
    mut exporter: ResMut<DatasetExporter>,
    inference: Res<InferenceResult>,
) {
    let bird_eye_id = contexts.add_image(ui_state.bird_eye_texture.clone());
    let echogram_id = contexts.add_image(ui_state.echogram_texture.clone());
    let generated_id = contexts.add_image(ui_state.generated_visual_texture.clone());

    egui::TopBottomPanel::top("top_status").show(contexts.ctx_mut(), |ui| {
        ui.horizontal(|ui| {
            ui.heading("\u{1f41f} DeepFish Sim-In-Loop Analytics");
            ui.separator();
            ui.label(format!("Frames: {}", exporter.frame_count));
            ui.separator();
            let acc = if inference.total_count > 0 {
                (inference.correct_count as f32 / inference.total_count as f32) * 100.0
            } else {
                0.0
            };
            ui.label(
                egui::RichText::new(format!("Accuracy: {:.1}%", acc))
                    .color(egui::Color32::LIGHT_GREEN)
                    .strong(),
            );
        });
    });

    egui::TopBottomPanel::bottom("bottom_controls").show(contexts.ctx_mut(), |ui| {
        ui.add_space(5.0);
        ui.horizontal(|ui| {
            if ui
                .button(if exporter.is_exporting {
                    "\u{1f534} Stop Recording"
                } else {
                    "\u{23fa} Start Recording [R]"
                })
                .clicked()
            {
                exporter.is_exporting = !exporter.is_exporting;
            }
            ui.separator();
            ui.label(
                "Orbit: L-Click Drag | Pan: WASD | Zoom: Scroll | Reset: Space | Single Export: E",
            );
        });
        ui.add_space(5.0);
    });

    egui::CentralPanel::default()
        .frame(egui::Frame::none().inner_margin(0.0))
        .show(contexts.ctx_mut(), |ui| {
            ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);

            let total_h = ui.available_height();
            let half_h = total_h / 2.0;

            ui.columns(2, |cols| {
                cols[0].vertical(|ui| {
                    ui.group(|ui| {
                        ui.set_min_height(half_h);
                        ui.set_max_height(half_h);
                        ui.set_width(ui.available_width());
                        ui.vertical_centered(|ui| {
                            ui.label(egui::RichText::new("QUADRANT 1: 3D SIMULATION").strong());
                        });
                    });

                    ui.group(|ui| {
                        ui.set_min_height(half_h);
                        ui.set_max_height(half_h);
                        ui.set_width(ui.available_width());
                        ui.vertical_centered(|ui| {
                            ui.label(
                                egui::RichText::new("QUADRANT 3: AI ANALYSIS & RECONSTRUCTION").strong(),
                            );
                            ui.add_space(5.0);

                            ui.label("Model Prediction Confidence:");
                            if inference.predictions.is_empty() {
                                ui.label("Searching for inference server...");
                            } else {
                                let top_gt = inference.ground_truth_dist.first().map(|x| &x.0);
                                for (species, score) in &inference.predictions {
                                    let is_dominant = top_gt == Some(species);
                                    ui.horizontal(|ui| {
                                        ui.add_space(20.0);
                                        ui.label(format!("{:<10}", species));
                                        let color = if is_dominant {
                                            egui::Color32::from_rgb(0, 255, 100)
                                        } else {
                                            egui::Color32::WHITE
                                        };
                                        ui.add(
                                            egui::ProgressBar::new((score + 1.0) / 2.0)
                                                .text(format!("{:.2}", score))
                                                .fill(color)
                                                .desired_width(150.0),
                                        ); 
                                        if is_dominant {
                                            ui.label("\u{1f3af}");
                                        }
                                    });
                                }
                            }
                            
                            ui.separator();
                            ui.add_space(5.0);
                            ui.label(egui::RichText::new("AI IMAGE RECONSTRUCTION").strong());
                            ui.add_space(10.0);

                            ui.horizontal(|ui| {
                                let available = ui.available_width();
                                let content_w = 320.0;
                                let padding = ((available - content_w) / 2.0).max(0.0);
                                ui.add_space(padding);

                                ui.vertical(|ui| {
                                    ui.label("Sonar");
                                    ui.image(egui::load::SizedTexture::new(
                                        echogram_id,
                                        [40.0, 120.0],
                                    ));
                                });

                                ui.add_space(20.0);

                                ui.vertical(|ui| {
                                    ui.label("Generated");
                                    ui.image(egui::load::SizedTexture::new(
                                        generated_id,
                                        [120.0, 120.0],
                                    ));
                                });

                                ui.add_space(20.0);

                                ui.vertical(|ui| {
                                    ui.label("Ground Truth");
                                    ui.image(egui::load::SizedTexture::new(
                                        bird_eye_id,
                                        [120.0, 120.0],
                                    ));
                                });
                            });
                            ui.add_space(10.0);
                            ui.label(format!("Samples Analyzed: {}", inference.total_count));
                        });
                    });
                });

                cols[1].vertical(|ui| {
                    ui.group(|ui| {
                        ui.set_min_height(half_h);
                        ui.set_max_height(half_h);
                        ui.set_width(ui.available_width());
                        ui.vertical_centered(|ui| {
                            ui.label(egui::RichText::new("QUADRANT 2: BIRD'S EYE (GT)").strong());
                            ui.image(egui::load::SizedTexture::new(
                                bird_eye_id,
                                [ui.available_width() - 30.0, half_h - 100.0],
                            ));

                            ui.horizontal(|ui| {
                                ui.label("Composition:");
                                if inference.ground_truth_dist.is_empty() {
                                    ui.label("None");
                                } else {
                                    for (species, pct) in &inference.ground_truth_dist {
                                        ui.label(format!("{}: {:.0}%", species, pct * 100.0));
                                    }
                                }
                            });
                        });
                    });

                    ui.group(|ui| {
                        ui.set_min_height(half_h);
                        ui.set_max_height(half_h);
                        ui.set_width(ui.available_width());
                        ui.vertical_centered(|ui| {
                            ui.label(egui::RichText::new("QUADRANT 4: ECHOSOUNDER").strong());
                            ui.image(egui::load::SizedTexture::new(
                                echogram_id,
                                [ui.available_width() - 30.0, half_h - 70.0],
                            ));
                            ui.label("Acoustic Time-Series (120kHz)");
                        });
                    });
                });
            });
        });
}

fn model_inference_system(
    time: Res<Time>,
    mut inference: ResMut<InferenceResult>,
    ui_state: Res<UIState>,
    mut images: ResMut<Assets<Image>>,
    channels: Res<InferenceChannels>,
    transducer_query: Query<&Transform, With<Transducer>>,
    fish_query: Query<(&Transform, &Species)>,
) {
    while let Ok(resp) = channels.rx_preds.try_recv() {
        inference.predictions = resp.predictions.clone();
        
        if let Some((top_gt, _)) = inference.ground_truth_dist.first() {
            let top_gt_name = top_gt.clone();
            if let Some((top_pred, _)) = resp.predictions.first() {
                inference.total_count += 1;
                if top_pred == &top_gt_name {
                    inference.correct_count += 1;
                }
            }
        }

        if let Ok(bytes) = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &resp.generated_image) {
            if let Ok(decoded) = image::load_from_memory(&bytes) {
                let rgba = decoded.to_rgba8();
                if let Some(target_img) = images.get_mut(&ui_state.generated_visual_texture) {
                    target_img.data = rgba.into_raw();
                }
            }
        }
    }

    inference.timer.tick(time.delta());
    if !inference.timer.finished() {
        return;
    }

    let Ok(transducer_tf) = transducer_query.get_single() else {
        return;
    };
    let t_pos = transducer_tf.translation;
    let half_angle_rad = (TRANSDUCER_CONE_ANGLE / 2.0).to_radians();

    let mut species_counts = std::collections::HashMap::new();
    let mut total_fish_in_beam = 0;

    for (fish_tf, species) in fish_query.iter() {
        let to_fish = fish_tf.translation - t_pos;
        let distance = to_fish.length();
        let direction = to_fish / distance;
        let angle = direction.dot(Vec3::NEG_Y).acos();

        if angle < half_angle_rad {
            let s_name = format!("{:?}", species);
            *species_counts.entry(s_name).or_insert(0) += 1;
            total_fish_in_beam += 1;
        }
    }

    let mut gt_dist = Vec::new();
    if total_fish_in_beam == 0 {
        gt_dist.push(("Empty".to_string(), 1.0));
    } else {
        for (name, count) in species_counts {
            gt_dist.push((name, count as f32 / total_fish_in_beam as f32));
        }
        gt_dist.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    }
    inference.ground_truth_dist = gt_dist;

    let Some(img) = images.get(&ui_state.echogram_texture) else {
        return;
    };
    let Ok(dynamic_img) = img.clone().try_into_dynamic() else {
        return;
    };
    
    let mut buffer = std::io::Cursor::new(Vec::new());
    if dynamic_img
        .to_rgba8()
        .write_to(&mut buffer, image::ImageFormat::Png)
        .is_ok()
    {
        let _ = channels.tx_image.send(buffer.into_inner());
    }
}

use bevy::render::view::screenshot::{Screenshot, save_to_disk};

fn dataset_exporter_system(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    mut exporter: ResMut<DatasetExporter>,
    ui_state: Res<UIState>,
    images: Res<Assets<Image>>,
    inference: Res<InferenceResult>,
    _bird_eye_camera: Query<Entity, With<BirdEyeCamera>>,
) {
    if keys.just_pressed(KeyCode::KeyR) {
        exporter.is_exporting = !exporter.is_exporting;
        if exporter.is_exporting {
            info!("Recording Started...");
        } else {
            info!("Recording Stopped. Total frames: {}", exporter.frame_count);
        }
    }

    let mut trigger_manual = false;
    if keys.just_pressed(KeyCode::KeyE) {
        trigger_manual = true;
    }

    if let Some(img) = images.get(&ui_state.echogram_texture) {
        let mut latest_ping_rgb = Vec::new();
        for y in 0..ECHOGRAM_HEIGHT {
            let i = ((y * ECHOGRAM_WIDTH + (ECHOGRAM_WIDTH - 1)) * 4) as usize;
            latest_ping_rgb.push(img.data[i]);
            latest_ping_rgb.push(img.data[i + 1]);
            latest_ping_rgb.push(img.data[i + 2]);
        }
        exporter.ping_history.push_back(latest_ping_rgb);
        if exporter.ping_history.len() > 32 {
            exporter.ping_history.pop_front();
        }
    }

    if (exporter.is_exporting && inference.timer.just_finished()) || trigger_manual {
        exporter.frame_count += 1;
        let frame = exporter.frame_count;
        let export_path = exporter.export_path.clone();

        if !Path::new(&export_path).exists() {
            let _ = std::fs::create_dir_all(&export_path);
        }

        let vis_path = format!("{}/frame_{:04}_visual.png", export_path, frame);
        commands.spawn(Screenshot::image(ui_state.bird_eye_texture.clone())).observe(save_to_disk(vis_path));

        let ac_img = images.get(&ui_state.echogram_texture).and_then(|img| {
            img.clone().try_into_dynamic().ok().map(|d| d.to_rgba8())
        });
        
        let mut full_history = Vec::new();
        for ping in &exporter.ping_history {
            full_history.extend_from_slice(ping);
        }
        while full_history.len() < 32 * ECHOGRAM_HEIGHT as usize * 3 {
            full_history.push(0);
        }

        let top_species = inference.ground_truth_dist.first().map(|(s, _)| s.clone());

        thread::spawn(move || {
            if let Some(data) = ac_img {
                let path = format!("{}/frame_{:04}_acoustic.png", export_path, frame);
                let _ = data.save(path);
            }
            if !full_history.is_empty() {
                let path = format!("{}/frame_{:04}_history.bin", export_path, frame);
                let _ = std::fs::write(path, full_history);
            }
            if let Some(species) = top_species {
                let path = format!("{}/frame_{:04}_meta.json", export_path, frame);
                let meta = format!(r#"{{"dominant_species": "{}"}}"#, species);
                let _ = std::fs::write(path, meta);
            }
        });
        
        info!("Queued export for frame {} (Acoustic + History)", frame);
    }
}

fn camera_controller_system(
    time: Res<Time>,
    mut mouse_wheel_events: EventReader<MouseWheel>,
    mut mouse_motion_events: EventReader<MouseMotion>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    mut query: Query<&mut Transform, With<MainCamera>>,
    mut settings: ResMut<CameraSettings>,
) {
    let mut transform = query.single_mut();

    for event in mouse_wheel_events.read() {
        if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
            settings.target_center.x -= event.x * 0.5;
            settings.target_center.z -= event.y * 0.5;
        } else {
            let zoom_delta = event.y * 2.0;
            settings.target_radius -= zoom_delta;
            settings.target_radius = settings.target_radius.clamp(5.0, 100.0);
            settings.target_yaw -= event.x * 0.05;
        }
    }

    if mouse_buttons.pressed(MouseButton::Left) {
        for event in mouse_motion_events.read() {
            settings.target_yaw -= event.delta.x * 0.005;
            settings.target_pitch += event.delta.y * 0.005;
        }
    } else {
        mouse_motion_events.clear();
    }
    settings.target_pitch = settings.target_pitch.clamp(-1.5, 1.5);

    let mut pan_vec = Vec3::ZERO;
    let pan_speed = 20.0 * time.delta_secs();

    let forward = transform.forward().as_vec3();
    let right = transform.right().as_vec3();
    let forward_flat = Vec3::new(forward.x, 0.0, forward.z).normalize_or_zero();
    let right_flat = Vec3::new(right.x, 0.0, right.z).normalize_or_zero();

    if keys.pressed(KeyCode::KeyW) || keys.pressed(KeyCode::ArrowUp) {
        pan_vec += forward_flat;
    }
    if keys.pressed(KeyCode::KeyS) || keys.pressed(KeyCode::ArrowDown) {
        pan_vec -= forward_flat;
    }
    if keys.pressed(KeyCode::KeyA) || keys.pressed(KeyCode::ArrowLeft) {
        pan_vec -= right_flat;
    }
    if keys.pressed(KeyCode::KeyD) || keys.pressed(KeyCode::ArrowRight) {
        pan_vec += right_flat;
    }

    settings.target_center += pan_vec * pan_speed;

    if keys.just_pressed(KeyCode::Space) {
        settings.target_radius = 30.0;
        settings.target_pitch = 0.5;
        settings.target_yaw = -0.5;
        settings.target_center = Vec3::ZERO;
    }

    let smoothing = 10.0 * time.delta_secs();
    settings.radius = settings.radius.lerp(settings.target_radius, smoothing);
    settings.pitch = settings.pitch.lerp(settings.target_pitch, smoothing);
    settings.yaw = settings.yaw.lerp(settings.target_yaw, smoothing);
    settings.center = settings.center.lerp(settings.target_center, smoothing);

    let rotation = Quat::from_rotation_y(settings.yaw) * Quat::from_rotation_x(-settings.pitch);
    let offset = rotation.mul_vec3(Vec3::new(0.0, 0.0, settings.radius));

    transform.translation = settings.center + offset;
    transform.look_at(settings.center, Vec3::Y);
}

fn boid_movement_system(
    time: Res<Time>,
    mut query: Query<(&mut Transform, &mut Boid, &Species)>,
) {
    let dt = time.delta_secs();
    let t = time.elapsed_secs();

    let boid_data: Vec<(Vec3, Vec3, Species, u32)> = query
        .iter()
        .map(|(tf, b, s)| (tf.translation, b.velocity, *s, b.school_id))
        .collect();

    for (mut transform, mut boid, species) in query.iter_mut() {
        let pos = transform.translation;
        let mut separation = Vec3::ZERO;
        let mut alignment = Vec3::ZERO;
        let mut cohesion = Vec3::ZERO;
        let mut neighbor_count = 0;

        let perception_radius = 5.0;
        let separation_distance = 1.5;

        for (other_pos, other_vel, _other_species, other_school) in &boid_data {
            // Stronger interaction within same school
            let is_schoolmate = *other_school == boid.school_id;
            let interaction_radius = if is_schoolmate {
                perception_radius * 1.5
            } else {
                perception_radius
            };

            let dist = pos.distance(*other_pos);
            if dist > 0.0 && dist < interaction_radius {
                if dist < separation_distance {
                    let diff = pos - *other_pos;
                    separation += diff.normalize_or_zero() / dist;
                }
                // Stronger alignment/cohesion for schoolmates
                let weight = if is_schoolmate { 2.0 } else { 0.5 };
                alignment += *other_vel * weight;
                cohesion += *other_pos * weight;
                neighbor_count += 1;
            }
        }

        let mut acceleration = Vec3::ZERO;

        if neighbor_count > 0 {
            let count_f = neighbor_count as f32;
            alignment = (alignment / count_f).normalize_or_zero() * boid.max_speed;
            let steer_align = (alignment - boid.velocity).clamp_length_max(boid.max_force);
            acceleration += steer_align * 1.5;

            cohesion = (cohesion / count_f - pos).normalize_or_zero() * boid.max_speed;
            let steer_cohere = (cohesion - boid.velocity).clamp_length_max(boid.max_force);
            acceleration += steer_cohere * 1.0;

            if separation.length_squared() > 0.0 {
                separation = (separation / count_f).normalize_or_zero() * boid.max_speed;
                let steer_sep = (separation - boid.velocity).clamp_length_max(boid.max_force * 1.2);
                acceleration += steer_sep * 2.0;
            }
        }

        // School heading preference (gentle steering toward school direction)
        let heading_preference = (boid.base_heading - boid.velocity.normalize_or_zero()) * 0.08;
        acceleration += heading_preference * boid.max_speed;

        let forward_drive = boid.velocity.normalize_or_zero() * 0.1;
        acceleration += forward_drive;

        // Depth keeping with small migration
        let (min_y, max_y) = species.preferred_depth_range();
        let target_y = (min_y + max_y) / 2.0 - TANK_SIZE.y / 2.0;
        let depth_error = target_y - pos.y;
        acceleration.y += depth_error * 0.5;

        boid.acceleration = acceleration;
        boid.velocity += acceleration * dt;
        boid.velocity = boid.velocity.clamp_length_max(boid.max_speed);

        let jitter = (t * 2.0 + boid.phase_offset).sin() * species.jitter_intensity() * 0.02;

        transform.translation += boid.velocity * dt;
        transform.translation.y += jitter;

        if boid.velocity.length_squared() > 0.001 {
            let target = transform.translation + boid.velocity;
            transform.look_at(target, Vec3::Y);
            transform.rotate_local_y(std::f32::consts::PI);
        }
    }
}

fn boundary_wrapping_system(mut query: Query<&mut Transform, With<Boid>>) {
    for mut transform in query.iter_mut() {
        if transform.translation.x > TANK_SIZE.x / 2.0 {
            transform.translation.x = -TANK_SIZE.x / 2.0;
        }
        if transform.translation.x < -TANK_SIZE.x / 2.0 {
            transform.translation.x = TANK_SIZE.x / 2.0;
        }
        if transform.translation.z > TANK_SIZE.z / 2.0 {
            transform.translation.z = -TANK_SIZE.z / 2.0;
        }
        if transform.translation.z < -TANK_SIZE.z / 2.0 {
            transform.translation.z = TANK_SIZE.z / 2.0;
        }
    }
}

fn echosounder_ping_system(
    time: Res<Time>,
    transducer_query: Query<&Transform, With<Transducer>>,
    fish_query: Query<(&Transform, &AcousticProfile)>,
    ui_state: Res<UIState>,
    mut images: ResMut<Assets<Image>>,
) {
    let Ok(transducer_tf) = transducer_query.get_single() else {
        return;
    };
    let Some(image) = images.get_mut(&ui_state.echogram_texture) else {
        return;
    };

    let t = time.elapsed_secs();

    // Shift existing echogram data left (scrolling effect)
    let row_width = ECHOGRAM_WIDTH as usize * 4;
    for y in 0..ECHOGRAM_HEIGHT as usize {
        let row_start = y * row_width;
        let row_end = row_start + row_width;
        image.data.copy_within(row_start + 4..row_end, row_start);
    }

    // Initialize new ping column with depth-dependent noise
    let mut rng = thread_rng();
    for y in 0..ECHOGRAM_HEIGHT {
        let i = ((y * ECHOGRAM_WIDTH + (ECHOGRAM_WIDTH - 1)) * 4) as usize;
        let depth_ratio = y as f32 / ECHOGRAM_HEIGHT as f32;
        // More noise near surface and bottom
        let noise_mult = 1.0 + 0.3 * (depth_ratio * (1.0 - depth_ratio)).sin().abs();
        let noise = (rng.gen_range(0..12) as f32 * noise_mult) as u8;
        image.data[i] = 2 + noise;
        image.data[i + 1] = 5 + noise;
        image.data[i + 2] = 15 + noise;
        image.data[i + 3] = 255;
    }

    let t_pos = transducer_tf.translation;
    let half_angle_rad = (TRANSDUCER_CONE_ANGLE / 2.0).to_radians();
    let side_lobe_angle = (TRANSDUCER_CONE_ANGLE * 1.5).to_radians();

    for (fish_tf, profile) in fish_query.iter() {
        let to_fish = fish_tf.translation - t_pos;
        let distance = to_fish.length();
        let direction = to_fish / distance;
        let angle = direction.dot(Vec3::NEG_Y).acos();

        if angle < side_lobe_angle {
            // Main lobe and side lobe response
            let beam_factor = if angle < half_angle_rad {
                (1.0 - (angle / half_angle_rad)).powi(2)
            } else {
                (1.0 - (angle / side_lobe_angle)).powi(2) * 0.1
            };

            // Orientation-dependent scattering
            let fish_forward = fish_tf.forward().as_vec3();
            let tilt_factor = 1.0 - direction.dot(fish_forward).abs();

            // Time-varying TS (fish orientation changes, ±15% variation)
            let ts_variation = 0.85 + 0.15 * (t * 2.0 + profile.ts_phase).sin();

            // Depth-based position
            let depth_ratio = (distance / (TANK_SIZE.y * 3.0)).clamp(0.0, 1.0);
            let y_center = (depth_ratio * (ECHOGRAM_HEIGHT as f32 - 1.0)) as i32;

            // Pulse stretching (vertical blob)
            let pulse_half_width = 3;
            for dy in -pulse_half_width..=pulse_half_width {
                let y = y_center + dy;
                if y < 0 || y >= ECHOGRAM_HEIGHT as i32 {
                    continue;
                }

                let i = ((y as u32 * ECHOGRAM_WIDTH + (ECHOGRAM_WIDTH - 1)) * 4) as usize;
                let pulse_factor = 1.0 - (dy.abs() as f32 / pulse_half_width as f32);

                // Multi-frequency intensity calculation
                // Each frequency channel has different target strength
                let (ts_38, ts_120, ts_200) = profile.target_strength;
                
                // Time-varying TS with frequency-dependent variation
                // Lower frequencies vary more (swim bladder resonance)
                let ts_variation_38 = 0.7 + 0.3 * (t * 2.0 + profile.ts_phase).sin();
                let ts_variation_120 = 0.8 + 0.2 * (t * 2.5 + profile.ts_phase * 1.2).sin();
                let ts_variation_200 = 0.85 + 0.15 * (t * 3.0 + profile.ts_phase * 1.4).sin();
                
                // Calculate intensity per frequency
                let intensity_38 = ((ts_38 + 65.0) / 45.0).clamp(0.0, 1.0)
                    * beam_factor * pulse_factor * tilt_factor * ts_variation_38;
                let intensity_120 = ((ts_120 + 65.0) / 45.0).clamp(0.0, 1.0)
                    * beam_factor * pulse_factor * tilt_factor * ts_variation_120;
                let intensity_200 = ((ts_200 + 65.0) / 45.0).clamp(0.0, 1.0)
                    * beam_factor * pulse_factor * tilt_factor * ts_variation_200;

                // Map frequencies to RGB channels for visualization
                // R = 38kHz (low freq), G = 120kHz (mid freq), B = 200kHz (high freq)
                image.data[i] = image.data[i].saturating_add((255.0 * intensity_38) as u8);
                image.data[i + 1] = image.data[i + 1].saturating_add((255.0 * intensity_120) as u8);
                image.data[i + 2] = image.data[i + 2].saturating_add((255.0 * intensity_200) as u8);
            }
        }
    }
}
fn set_camera_viewports(
    windows: Query<&Window>,
    mut cameras: Query<&mut Camera, With<MainCamera>>,
) {
    let Ok(window) = windows.get_single() else {
        return;
    };
    let Ok(mut camera) = cameras.get_single_mut() else {
        return;
    };

    let width = window.resolution.physical_width();
    let height = window.resolution.physical_height();
    let scale = window.resolution.scale_factor();

    let top_bar_h = (32.0 * scale) as u32;
    let bottom_bar_h = (40.0 * scale) as u32;

    let available_h = height
        .saturating_sub(top_bar_h)
        .saturating_sub(bottom_bar_h);

    camera.viewport = Some(bevy::render::camera::Viewport {
        physical_position: UVec2::new(0, top_bar_h),
        physical_size: UVec2::new(width / 2, available_h / 2),
        ..default()
    });
}
fn tint_fish_system(
    query: Query<(Entity, &Children, &Species)>,
    children_query: Query<&Children>,
    mut mesh_query: Query<&mut MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut already_tinted: Local<std::collections::HashSet<Entity>>,
) {
    for (entity, children, species) in query.iter() {
        if already_tinted.contains(&entity) {
            continue;
        }

        let mut tinted_something = false;
        let mut stack: Vec<Entity> = children.iter().copied().collect();
        while let Some(child_entity) = stack.pop() {
            if let Ok(mut mat_comp) = mesh_query.get_mut(child_entity) {
                let current_handle = mat_comp.0.clone();
                if let Some(shared_mat) = materials.get(&current_handle) {
                    let mut unique_mat = shared_mat.clone();
                    unique_mat.base_color_texture = None;
                    unique_mat.base_color = species.color();
                    unique_mat.emissive = species.color().into();
                    unique_mat.metallic = 0.1;
                    unique_mat.perceptual_roughness = 0.5;

                    mat_comp.0 = materials.add(unique_mat);
                    tinted_something = true;
                }
            }
            if let Ok(next_children) = children_query.get(child_entity) {
                stack.extend(next_children.iter().copied());
            }
        }

        if tinted_something {
            already_tinted.insert(entity);
        }
    }
}

fn sync_cpu_buffer_system(
    _ui_state: Res<UIState>,
    _images: ResMut<Assets<Image>>,
) {
}

/// Update which schools are active based on time-varying population dynamics
fn update_population_system(
    time: Res<Time>,
    mut population: ResMut<DynamicPopulation>,
    mut fish_query: Query<(&mut Visibility, &Boid)>,
) {
    // Update cycle timer
    population.cycle_timer += time.delta_secs();
    if population.cycle_timer > POPULATION_CYCLE_PERIOD {
        population.cycle_timer -= POPULATION_CYCLE_PERIOD;
    }
    
    // Calculate phase in cycle (0 to 1)
    let cycle_phase = population.cycle_timer / POPULATION_CYCLE_PERIOD;
    
    // Generate species activity levels using sine waves with different phases
    // This creates a rotating dominance pattern
    for i in 0..3 {
        // Each species has a different phase offset (0, 1/3, 2/3 of cycle)
        let phase_offset = (i as f32) / 3.0;
        let activity = 0.5 + 0.5 * ((cycle_phase + phase_offset) * std::f32::consts::PI * 2.0).sin();
        population.species_activity[i] = activity.clamp(MIN_ACTIVE_RATIO, MAX_ACTIVE_RATIO);
    }
    
    // Determine which schools should be active based on species activity
    // Schools are assigned to species in round-robin fashion (species_list[i % 3])
    let species_activity_copy = population.species_activity;  // Copy to avoid borrow conflict
    for (school_idx, active) in population.active_schools.iter_mut().enumerate() {
        let species_idx = school_idx % 3;  // 0=Kingfish, 1=Snapper, 2=Cod
        let activity = species_activity_copy[species_idx];
        
        // Schools have a probability of being active based on species activity
        // Add some randomness for natural variation
        let random_factor = (school_idx as f32 * 0.7 + cycle_phase * 10.0).sin() * 0.5 + 0.5;
        let effective_activity = activity * 0.7 + random_factor * 0.3;
        
        *active = effective_activity > 0.5;
    }
    
    // Update fish visibility based on school activity
    for (mut visibility, boid) in fish_query.iter_mut() {
        let school_active = population.active_schools.get(boid.school_id as usize)
            .copied()
            .unwrap_or(true);
        
        if school_active {
            *visibility = Visibility::Visible;
        } else {
            *visibility = Visibility::Hidden;
        }
    }
}

/// Gradually change school headings over time for direction invariance
/// Only updates base_heading - actual velocity is handled by boid_movement_system
fn update_school_headings_system(
    time: Res<Time>,
    mut fish_query: Query<(&mut Boid, &Species)>,
) {
    let dt = time.delta_secs();
    
    // Collect unique school IDs and their current state
    let mut school_data: std::collections::HashMap<u32, (Vec3, f32)> = std::collections::HashMap::new();
    
    for (boid, _species) in fish_query.iter() {
        if !school_data.contains_key(&boid.school_id) {
            school_data.insert(
                boid.school_id,
                (boid.base_heading, boid.heading_change_timer),
            );
        }
    }
    
    // Update headings per school
    for (school_id, data) in school_data.iter_mut() {
        let (heading, timer) = data;
        let mut new_heading = *heading;
        let mut new_timer = *timer;
        
        new_timer += dt;
        
        // Change heading every 30-90 seconds (randomized per school)
        let change_interval = 30.0 + (*school_id as f32 * 7.5);
        
        if new_timer > change_interval {
            new_timer = 0.0;
            
            // Gradually rotate heading by 15-45 degrees
            let rotation_angle = (*school_id as f32 * 0.3 + 0.5).sin() * 0.5;  // -0.5 to 0.5 radians
            
            // Rotate around Y axis (horizontal plane)
            let sin_a = rotation_angle.sin();
            let cos_a = rotation_angle.cos();
            
            let old_x = heading.x;
            let old_z = heading.z;
            
            new_heading.x = old_x * cos_a - old_z * sin_a;
            new_heading.z = old_x * sin_a + old_z * cos_a;
            new_heading = new_heading.normalize_or_zero();
        }
        
        *data = (new_heading, new_timer);
    }
    
    // Apply updated headings to all fish
    for (mut boid, _species) in fish_query.iter_mut() {
        if let Some((new_heading, new_timer)) = school_data.get(&boid.school_id) {
            boid.base_heading = *new_heading;
            boid.heading_change_timer = *new_timer;
        }
    }
}
