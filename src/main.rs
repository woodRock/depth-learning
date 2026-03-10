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
use std::fs;
use std::path::Path;

// --- Constants & Config ---
const TANK_SIZE: Vec3 = Vec3::new(20.0, 10.0, 20.0);
const FISH_COUNT: usize = 120;
const TRANSDUCER_CONE_ANGLE: f32 = 12.0;
const ECHOGRAM_WIDTH: u32 = 512;
const ECHOGRAM_HEIGHT: u32 = 256;
const BIRD_EYE_WIDTH: u32 = 512;
const BIRD_EYE_HEIGHT: u32 = 512;

#[derive(Resource)]
struct InferenceResult {
    pub predictions: Vec<(String, f32)>,
    pub ground_truth_dist: Vec<(String, f32)>, // (Species, Percentage)
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
}

impl Species {
    fn preferred_depth_range(&self) -> (f32, f32) {
        match self {
            Species::Kingfish => (7.0, 9.5),
            Species::Snapper => (3.0, 6.5),
            Species::Cod => (0.0, 2.5),
        }
    }
    fn color(&self) -> Srgba {
        match self {
            Species::Kingfish => Srgba::new(0.2, 0.5, 1.0, 1.0),
            Species::Snapper => Srgba::new(1.0, 0.3, 0.3, 1.0),
            Species::Cod => Srgba::new(0.5, 0.5, 0.2, 1.0),
        }
    }
    fn target_strength(&self) -> f32 {
        match self {
            Species::Kingfish => -35.0,
            Species::Snapper => -42.0,
            Species::Cod => -38.0,
        }
    }
}

#[derive(Component)]
struct AcousticProfile {
    target_strength: f32,
}

#[derive(Component)]
struct Boid {
    velocity: Vec3,
}

#[derive(Component)]
struct MainCamera;

#[derive(Component)]
struct BirdEyeCamera;

#[derive(Component)]
struct Transducer;

#[derive(Resource)]
struct UIState {
    bird_eye_texture: Handle<Image>,
    echogram_texture: Handle<Image>,
}

#[derive(Resource)]
struct CameraSettings {
    pub radius: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub center: Vec3,
    // Target values for smoothing
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

#[derive(Resource)]
struct DatasetExporter {
    pub frame_count: u32,
    pub export_path: String,
    pub is_exporting: bool,
}

impl Default for DatasetExporter {
    fn default() -> Self {
        Self {
            frame_count: 0,
            export_path: "dataset".to_string(),
            is_exporting: false,
        }
    }
}

// --- Main App ---

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(EguiPlugin)
        .insert_resource(ClearColor(Color::BLACK))
        .init_resource::<CameraSettings>()
        .init_resource::<DatasetExporter>()
        .init_resource::<InferenceResult>()
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
            ),
        )
        .run();
}

// --- Setup Systems ---

fn setup_render_textures(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    // 1. Echogram Texture
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

    // 2. Bird's Eye Render Target
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

    commands.insert_resource(UIState {
        bird_eye_texture: bird_handle,
        echogram_texture: echo_handle,
    });
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
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

    let mut rng = thread_rng();
    for _ in 0..FISH_COUNT {
        let species = match rng.gen_range(0..3) {
            0 => Species::Kingfish,
            1 => Species::Snapper,
            _ => Species::Cod,
        };
        let (min_y, max_y) = species.preferred_depth_range();
        let pos = Vec3::new(
            rng.gen_range(-TANK_SIZE.x / 2.0..TANK_SIZE.x / 2.0),
            rng.gen_range(min_y..max_y) - TANK_SIZE.y / 2.0,
            rng.gen_range(-TANK_SIZE.z / 2.0..TANK_SIZE.z / 2.0),
        );

        commands.spawn((
            Mesh3d(meshes.add(Capsule3d::new(0.1, 0.3))),
            MeshMaterial3d(materials.add(StandardMaterial {
                base_color: species.color().into(),
                unlit: false,
                emissive: species.color().into(),
                ..default()
            })),
            Transform::from_translation(pos),
            species,
            AcousticProfile {
                target_strength: species.target_strength(),
            },
            Boid {
                velocity: Vec3::new(rng.gen_range(-1.0..1.0), 0.0, rng.gen_range(-1.0..1.0))
                    .normalize()
                    * 2.0,
            },
        ));
    }

    commands.spawn((
        PointLight {
            intensity: 1_000_000.0,
            range: 100.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 20.0, 0.0),
    ));
    commands.insert_resource(AmbientLight {
        color: Color::srgb(0.1, 0.1, 0.2),
        brightness: 200.0,
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
            scale: 0.1,
            ..OrthographicProjection::default_3d()
        }),
        Transform::from_xyz(0.0, 30.0, 0.0).looking_at(Vec3::ZERO, Vec3::NEG_Z),
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

    let ctx = contexts.ctx_mut();

    // 1. TOP STATUS BAR
    egui::TopBottomPanel::top("top_status").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.heading("🐟 DeepFish Sim-In-Loop Analytics");
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

    // 2. BOTTOM CONTROL TOOLBAR
    egui::TopBottomPanel::bottom("bottom_controls").show(ctx, |ui| {
        ui.add_space(5.0);
        ui.horizontal(|ui| {
            if ui
                .button(if exporter.is_exporting {
                    "🔴 Stop Recording"
                } else {
                    "⏺ Start Recording [R]"
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

    // 3. CENTRAL QUADRANT GRID
    egui::CentralPanel::default()
        .frame(egui::Frame::none().inner_margin(0.0))
        .show(ctx, |ui| {
            // Remove spacing between rows/cols
            ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);

            let total_h = ui.available_height();
            let half_h = total_h / 2.0;

            ui.columns(2, |cols| {
                // --- LEFT COLUMN (Simulation & AI) ---
                cols[0].vertical(|ui| {
                    // Q1: 3D SIMULATION SPACE
                    ui.group(|ui| {
                        ui.set_min_height(half_h);
                        ui.set_max_height(half_h);
                        ui.set_width(ui.available_width());
                        ui.vertical_centered(|ui| {
                            ui.label(egui::RichText::new("QUADRANT 1: 3D SIMULATION").strong());
                        });
                    });

                    // Q3: AI ANALYSIS & METRICS
                    ui.group(|ui| {
                        ui.set_min_height(half_h);
                        ui.set_max_height(half_h);
                        ui.set_width(ui.available_width());
                        ui.vertical_centered(|ui| {
                            ui.label(
                                egui::RichText::new("QUADRANT 3: AI ANALYSIS & METRICS").strong(),
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
                                        ); // Fixed width to prevent quadrant expansion
                                        if is_dominant {
                                            ui.label("🎯");
                                        }
                                    });
                                }
                            }
                            ui.add_space(5.0);
                            ui.label(format!("Samples Analyzed: {}", inference.total_count));
                        });
                    });
                });

                // --- RIGHT COLUMN (Visual GT & Echosounder) ---
                cols[1].vertical(|ui| {
                    // Q2: BIRD'S EYE (GROUND TRUTH)
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

                    // Q4: ECHOSOUNDER (AI INPUT)
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
    images: Res<Assets<Image>>,
    transducer_query: Query<&Transform, With<Transducer>>,
    fish_query: Query<(&Transform, &Species)>,
) {
    inference.timer.tick(time.delta());
    if !inference.timer.finished() {
        return;
    }

    // 1. Calculate Biological Composition (Ground Truth Distribution)
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

    // Convert counts to percentages
    let mut gt_dist = Vec::new();
    for (name, count) in species_counts {
        gt_dist.push((name, count as f32 / total_fish_in_beam as f32));
    }
    gt_dist.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    inference.ground_truth_dist = gt_dist;

    // 2. Perform Inference
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
        .is_err()
    {
        return;
    }
    let png_bytes = buffer.into_inner();

    let client = reqwest::blocking::Client::new();
    let form = reqwest::blocking::multipart::Form::new().part(
        "file",
        reqwest::blocking::multipart::Part::bytes(png_bytes)
            .file_name("ping.png")
            .mime_str("image/png")
            .unwrap(),
    );

    if let Ok(preds) = client
        .post("http://127.0.0.1:8000/predict_acoustic")
        .multipart(form)
        .send()
        .and_then(|resp| resp.json::<Vec<(String, f32)>>())
    {
        inference.predictions = preds.clone();

        // 3. Update Accuracy (Top-1 dominant species match)
        if let Some((top_gt, _)) = inference.ground_truth_dist.first() {
            let top_gt_name = top_gt.clone(); // Clone to avoid borrow conflict
            if let Some((top_pred, _)) = preds.first() {
                inference.total_count += 1;
                if top_pred == &top_gt_name {
                    inference.correct_count += 1;
                }
            }
        }
    }
}

fn dataset_exporter_system(
    keys: Res<ButtonInput<KeyCode>>,
    mut exporter: ResMut<DatasetExporter>,
    ui_state: Res<UIState>,
    images: Res<Assets<Image>>,
    inference: Res<InferenceResult>,
) {
    // 1. Handle Toggles
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

    // 2. Only export if recording is active AND the inference timer just finished
    // This synchronizes our dataset (Visual, Acoustic, and Metadata labels)
    // and prevents saving 60 identical frames per second.
    if (exporter.is_exporting && inference.timer.just_finished()) || trigger_manual {
        exporter.frame_count += 1;
        let frame = exporter.frame_count;

        if !Path::new(&exporter.export_path).exists() {
            let _ = fs::create_dir_all(&exporter.export_path);
        }

        // --- A. Export Visual ---
        if let Some(img) = images.get(&ui_state.bird_eye_texture) {
            let path = format!("{}/frame_{:04}_visual.png", exporter.export_path, frame);
            // NOTE: If this image is black, it's because Bevy 0.15 RenderTargets 
            // aren't automatically copied back to CPU.
            if let Ok(dyn_img) = img.clone().try_into_dynamic() {
                if let Err(e) = dyn_img.to_rgba8().save(&path) {
                    warn!("Failed to save visual frame: {}", e);
                }
            }
        }

        // --- B. Export Acoustic ---
        if let Some(img) = images.get(&ui_state.echogram_texture) {
            let path_png = format!("{}/frame_{:04}_acoustic.png", exporter.export_path, frame);
            if let Ok(dyn_img) = img.clone().try_into_dynamic() {
                let _ = dyn_img.to_rgba8().save(path_png);
            }

            let path_bin = format!("{}/frame_{:04}_ping.bin", exporter.export_path, frame);
            let mut latest_ping = Vec::new();
            for y in 0..ECHOGRAM_HEIGHT {
                let i = ((y * ECHOGRAM_WIDTH + (ECHOGRAM_WIDTH - 1)) * 4) as usize;
                latest_ping.push(img.data[i]);
            }
            if let Err(e) = fs::write(&path_bin, latest_ping) {
                warn!("Failed to save ping binary: {}", e);
            }
        }

        // --- C. Export Metadata ---
        if let Some((top_species, _)) = inference.ground_truth_dist.first() {
            let path_meta = format!("{}/frame_{:04}_meta.json", exporter.export_path, frame);
            let meta = format!(r#"{{"dominant_species": "{}"}}"#, top_species);
            let _ = fs::write(path_meta, meta);
        }
        
        info!("Exported frame {} to {}", frame, exporter.export_path);
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

    // 1. Zoom (Scroll / Pinch)
    for event in mouse_wheel_events.read() {
        if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
            // Side scroll/trackpad horizontal can pan
            settings.target_center.x -= event.x * 0.5;
            settings.target_center.z -= event.y * 0.5;
        } else {
            let zoom_delta = event.y * 2.0;
            settings.target_radius -= zoom_delta;
            settings.target_radius = settings.target_radius.clamp(5.0, 100.0);

            // Trackpad horizontal scroll = Yaw orbit
            settings.target_yaw -= event.x * 0.05;
        }
    }

    // 2. Rotation (Left Click + Drag)
    if mouse_buttons.pressed(MouseButton::Left) {
        for event in mouse_motion_events.read() {
            settings.target_yaw -= event.delta.x * 0.005;
            settings.target_pitch += event.delta.y * 0.005;
        }
    } else {
        mouse_motion_events.clear();
    }
    settings.target_pitch = settings.target_pitch.clamp(-1.5, 1.5);

    // 3. Panning (WASD / Arrows)
    let mut pan_vec = Vec3::ZERO;
    let pan_speed = 20.0 * time.delta_secs();

    // Get camera relative axes
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

    // 4. Reset View
    if keys.just_pressed(KeyCode::Space) {
        settings.target_radius = 30.0;
        settings.target_pitch = 0.5;
        settings.target_yaw = -0.5;
        settings.target_center = Vec3::ZERO;
    }

    // 5. Apply Smoothing (Lerp)
    let smoothing = 10.0 * time.delta_secs();
    settings.radius = settings.radius.lerp(settings.target_radius, smoothing);
    settings.pitch = settings.pitch.lerp(settings.target_pitch, smoothing);
    settings.yaw = settings.yaw.lerp(settings.target_yaw, smoothing);
    settings.center = settings.center.lerp(settings.target_center, smoothing);

    // Calculate final transform
    let rotation = Quat::from_rotation_y(settings.yaw) * Quat::from_rotation_x(-settings.pitch);
    let offset = rotation.mul_vec3(Vec3::new(0.0, 0.0, settings.radius));

    transform.translation = settings.center + offset;
    transform.look_at(settings.center, Vec3::Y);
}

fn boid_movement_system(time: Res<Time>, mut query: Query<(&mut Transform, &Boid, &Species)>) {
    for (mut transform, boid, species) in query.iter_mut() {
        transform.translation += boid.velocity * time.delta_secs();
        if boid.velocity.length_squared() > 0.001 {
            let target = transform.translation + boid.velocity;
            transform.look_at(target, Vec3::Y);
        }
        let (min_y, max_y) = species.preferred_depth_range();
        let current_y = transform.translation.y + TANK_SIZE.y / 2.0;
        if current_y < min_y {
            transform.translation.y += 0.05;
        } else if current_y > max_y {
            transform.translation.y -= 0.05;
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

    // 1. Shift left
    for y in 0..ECHOGRAM_HEIGHT {
        for x in 0..(ECHOGRAM_WIDTH - 1) {
            let dst = ((y * ECHOGRAM_WIDTH + x) * 4) as usize;
            let src = ((y * ECHOGRAM_WIDTH + x + 1) * 4) as usize;
            image.data.copy_within(src..src + 4, dst);
        }
    }

    // 2. Clear right column (Dark deep blue)
    for y in 0..ECHOGRAM_HEIGHT {
        let i = ((y * ECHOGRAM_WIDTH + (ECHOGRAM_WIDTH - 1)) * 4) as usize;
        image.data[i] = 2;
        image.data[i + 1] = 5;
        image.data[i + 2] = 15;
        image.data[i + 3] = 255;
    }

    let t_pos = transducer_tf.translation;
    let half_angle_rad = (TRANSDUCER_CONE_ANGLE / 2.0).to_radians();

    for (fish_tf, profile) in fish_query.iter() {
        let to_fish = fish_tf.translation - t_pos;
        let distance = to_fish.length();
        let direction = to_fish / distance;
        let angle = direction.dot(Vec3::NEG_Y).acos();

        if angle < half_angle_rad {
            // BEAM PATTERN: Intensity drops as we move off-axis (Gaussian-ish)
            let beam_factor = (1.0 - (angle / half_angle_rad)).powi(2);

            let depth_ratio = (distance / (TANK_SIZE.y * 3.0)).clamp(0.0, 1.0);
            let y_center = (depth_ratio * (ECHOGRAM_HEIGHT as f32 - 1.0)) as i32;

            // PULSE STRETCHING: Draw a small vertical "blob" instead of a pixel
            let pulse_half_width = 3;
            for dy in -pulse_half_width..=pulse_half_width {
                let y = y_center + dy;
                if y < 0 || y >= ECHOGRAM_HEIGHT as i32 {
                    continue;
                }

                let i = ((y as u32 * ECHOGRAM_WIDTH + (ECHOGRAM_WIDTH - 1)) * 4) as usize;

                // FALLOFF within the pulse
                let pulse_factor = 1.0 - (dy.abs() as f32 / pulse_half_width as f32);
                let intensity = ((profile.target_strength + 65.0) / 45.0).clamp(0.0, 1.0)
                    * beam_factor
                    * pulse_factor;

                // ADDITIVE blending (Multiple fish make a brighter blob)
                image.data[i] = image.data[i].saturating_add((255.0 * intensity) as u8);
                image.data[i + 1] = image.data[i + 1].saturating_add((200.0 * intensity) as u8);
                image.data[i + 2] = image.data[i + 2].saturating_add((50.0 * intensity) as u8);
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

    // The central grid occupies the space between the top status bar and bottom toolbar
    let top_bar_h = (32.0 * scale) as u32;
    let bottom_bar_h = (40.0 * scale) as u32;

    let available_h = height
        .saturating_sub(top_bar_h)
        .saturating_sub(bottom_bar_h);

    // Top-Left Quadrant
    camera.viewport = Some(bevy::render::camera::Viewport {
        physical_position: UVec2::new(0, top_bar_h),
        physical_size: UVec2::new(width / 2, available_h / 2),
        ..default()
    });
}
