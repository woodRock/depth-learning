// Echosounder realism improvements
// Add these functions to main.rs and update echosounder_ping_system

// 1. Rayleigh noise for realistic speckle
fn rayleigh_noise(rng: &mut ThreadRng) -> f32 {
    let u1 = rng.gen_range(0.001..1.0);
    let u2 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

// 2. Gaussian beam pattern (more realistic than linear)
fn gaussian_beam_response(angle: f32, half_angle: f32) -> f32 {
    let normalized_angle = angle / half_angle;
    (-2.77 * normalized_angle * normalized_angle).exp()
}

// 3. Transmission Loss Compensation (TVG)
fn tvg_gain(distance: f32) -> f32 {
    1.0 + TVG_GAIN * distance
}

// 4. Frequency-dependent absorption
fn absorption_coefficient(frequency_khz: f32, distance: f32) -> f32 {
    let coeff = match frequency_khz as i32 {
        38 => 0.008,
        120 => 0.04,
        200 => 0.08,
        _ => 0.05,
    };
    (-coeff * distance).exp()
}

// Updated echosounder_ping_system with all improvements
fn echosounder_ping_system_improved(
    time: Res<Time>,
    transducer_query: Query<&Transform, With<Transducer>>,
    fish_query: Query<(&Transform, &AcousticProfile)>,
    ui_state: Res<UIState>,
    mut images: ResMut<Assets<Image>>,
) {
    // ... (keep setup code the same)
    
    // Initialize new ping column with Rayleigh noise
    let mut rng = thread_rng();
    for y in 0..ECHOGRAM_HEIGHT {
        let i = ((y * ECHOGRAM_WIDTH + (ECHOGRAM_WIDTH - 1)) * 4) as usize;
        
        // Rayleigh-distributed speckle noise
        let rayleigh = rayleigh_noise(&mut rng);
        let depth_ratio = y as f32 / ECHOGRAM_HEIGHT as f32;
        let background = 5.0 + 8.0 * depth_ratio + rayleigh.abs() * 3.0;
        
        image.data[i] = background as u8;
        image.data[i + 1] = (background * 1.2) as u8;
        image.data[i + 2] = (background * 1.5) as u8;
    }
    
    // In fish rendering loop, replace beam_factor calculation:
    let beam_factor = gaussian_beam_response(angle, half_angle_rad);
    
    // Add TVG and absorption to intensity calculation:
    let tvg = tvg_gain(distance);
    let abs_38 = absorption_coefficient(38.0, distance);
    let abs_120 = absorption_coefficient(120.0, distance);
    let abs_200 = absorption_coefficient(200.0, distance);
    
    let intensity_38 = base_intensity * tvg * abs_38;
    let intensity_120 = base_intensity * tvg * abs_120;
    let intensity_200 = base_intensity * tvg * abs_200;
}
