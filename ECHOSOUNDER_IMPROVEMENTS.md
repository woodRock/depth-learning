# Echosounder Realism Improvements

## Current Implementation Analysis

### What's Already Good ✅
1. **Multi-frequency simulation** (38, 120, 200 kHz) with species-specific TS
2. **Beam pattern** with main lobe and side lobes
3. **Fish orientation effects** (tilt factor)
4. **Time-varying TS** from fish movement
5. **Depth-dependent noise**
6. **Pulse stretching** (vertical resolution)

### What's Missing for Realism ⚠️

## Recommended Improvements

### 1. **Transmission Loss (TVG - Time Varied Gain)**
Real echosounders compensate for signal loss over distance.

```rust
// Add to echosounder_ping_system
fn transmission_loss(distance: f32, frequency: f32) -> f32 {
    // Spreading loss + absorption
    let spreading = 20.0 * distance.log10();
    let absorption = 0.001 * frequency * distance; // dB per meter
    (spreading + absorption) / 100.0 // Normalize
}
```

**Impact**: Fish at different depths will have more realistic intensity profiles.

---

### 2. **Beam Pattern Shaping**
Current: Simple linear falloff
Real: Gaussian or sinc function beam pattern

```rust
// Replace simple beam_factor with realistic beam pattern
fn beam_pattern_response(angle: f32, half_angle: f32) -> f32 {
    // Gaussian beam pattern (more realistic)
    let normalized_angle = angle / half_angle;
    (-2.77 * normalized_angle * normalized_angle).exp()
}
```

**Impact**: Smoother transitions at beam edges, more realistic "fuzzy" detections.

---

### 3. **Range Gating / Pulse Length**
Current: Fixed pulse width
Real: Pulse length affects vertical resolution

```rust
// Add pulse length parameter
const PULSE_LENGTH_M: f32 = 0.5; // meters

// Calculate range gate based on pulse length
let range_gate = (PULSE_LENGTH_M / 2.0) / (distance_resolution);
let pulse_half_width = (range_gate * ECHOGRAM_HEIGHT as f32 / TANK_SIZE.y) as i32;
```

**Impact**: Deeper fish appear more stretched (realistic pulse compression).

---

### 4. **Speckle Noise (Rayleigh Distribution)**
Current: Uniform random noise
Real: Speckle follows Rayleigh distribution

```rust
fn rayleigh_noise(scale: f32) -> f32 {
    let u1 = thread_rng().gen_range(0.0..1.0);
    let u2 = thread_rng().gen_range(0.0..1.0);
    scale * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

// Apply to background
let background_noise = rayleigh_noise(0.05);
```

**Impact**: More realistic "grainy" appearance in echogram.

---

### 5. **Bottom Return Simulation**
Add tank bottom reflection (strong horizontal line at max depth).

```rust
// In echosounder_ping_system, after fish loop:
let bottom_depth = TANK_SIZE.y;
let bottom_y = ((bottom_depth / (TANK_SIZE.y * 3.0)) * ECHOGRAM_HEIGHT as f32) as i32;

// Draw strong bottom return
for x in (ECHOGRAM_WIDTH - 5)..ECHOGRAM_WIDTH {
    let i = ((bottom_y as u32 * ECHOGRAM_WIDTH + x) * 4) as usize;
    image.data[i] = 200;  // Strong red
    image.data[i + 1] = 50;
    image.data[i + 2] = 50;
}
```

**Impact**: Provides depth reference, helps model learn depth context.

---

### 6. **Multiple Echoes / Reverberation**
Real echosounders get multiple bounces from strong targets.

```rust
// After drawing primary echo, add secondary echoes
if intensity_120 > 0.5 {
    // Secondary echo (weaker, delayed)
    let secondary_y = y_center + 5;
    if secondary_y < ECHOGRAM_HEIGHT as i32 {
        let i = ((secondary_y as u32 * ECHOGRAM_WIDTH + (ECHOGRAM_WIDTH - 1)) * 4) as usize;
        image.data[i] = image.data[i].saturating_add((100.0 * intensity_120 * 0.3) as u8);
    }
}
```

**Impact**: Strong targets show "tails" - more realistic appearance.

---

### 7. **Frequency-Dependent Absorption**
Higher frequencies attenuate faster in water.

```rust
fn absorption_coefficient(frequency_khz: f32) -> f32 {
    // Approximate absorption in seawater at 10°C
    // dB per km, convert to dB per meter
    match frequency_khz as i32 {
        38 => 0.008,    // 38 kHz
        120 => 0.04,    // 120 kHz
        200 => 0.08,    // 200 kHz
        _ => 0.05,
    } / 1000.0
}

// Apply in intensity calculation
let absorption = absorption_coefficient(frequency) * distance;
let intensity = base_intensity * (-absorption).exp();
```

**Impact**: 200kHz signals weaker at depth - realistic frequency response.

---

### 8. **Fish School Clustering**
Current: Each fish independent
Real: Schools create merged echoes

```rust
// Group fish by proximity before rendering
fn cluster_fish(fish_positions: &[Vec3], threshold: f32) -> Vec<Vec3>> {
    // Merge fish within threshold distance
    // Render clusters as single larger echo
}

// For clusters, increase TS based on N fish
let cluster_ts = individual_ts + 10.0 * (cluster_size as f32).log10();
```

**Impact**: Schools appear as single large blobs (more realistic).

---

### 9. **Shadow Zones**
Fish block sound, creating shadows below them.

```rust
// Track which columns have strong echoes
let mut shadow_mask = vec![false; ECHOGRAM_WIDTH as usize];

for fish in fish_query.iter() {
    // Mark column as shadowed
    shadow_mask[fish_column] = true;
}

// Reduce intensity in shadowed columns below fish
if shadow_mask[column] && y > fish_y {
    intensity *= 0.5; // 50% reduction in shadow
}
```

**Impact**: Dark regions below strong targets - realistic acoustic shadowing.

---

### 10. **Noise Floor & Dynamic Range**
Current: No explicit noise floor
Real: Echosounders have noise floor (~-100 dB)

```rust
const NOISE_FLOOR_DB: f32 = -100.0;
const DYNAMIC_RANGE_DB: f32 = 50.0;

fn db_to_color(db: f32) -> (u8, u8, u8) {
    let normalized = ((db - NOISE_FLOOR_DB) / DYNAMIC_RANGE_DB).clamp(0.0, 1.0);
    
    // Logarithmic mapping (human perception)
    let log_normalized = normalized.powf(0.5);
    
    // Color map (blue → green → red for increasing intensity)
    if log_normalized < 0.33 {
        (0, (log_normalized * 3.0 * 255.0) as u8, 255)
    } else if log_normalized < 0.66 {
        (0, 255, (255.0 - (log_normalized - 0.33) * 3.0 * 255.0) as u8)
    } else {
        (((log_normalized - 0.66) * 3.0 * 255.0) as u8, 255, 0)
    }
}
```

**Impact**: Better contrast, more professional echogram appearance.

---

## Priority Recommendations

### High Priority (Big Impact, Easy to Implement)
1. **Transmission Loss (TVG)** - 10 lines of code
2. **Rayleigh Speckle Noise** - 5 lines
3. **Bottom Return** - 10 lines
4. **Frequency-Dependent Absorption** - 10 lines

### Medium Priority (Moderate Impact)
5. **Gaussian Beam Pattern** - 5 lines
6. **Noise Floor & Dynamic Range** - 15 lines
7. **Multiple Echoes** - 10 lines

### Lower Priority (Nice to Have)
8. **Shadow Zones** - 20 lines
9. **Fish School Clustering** - 30 lines
10. **Range Gating** - 15 lines

---

## Example: Quick Realism Boost (30 lines)

Add these improvements to `echosounder_ping_system`:

```rust
// 1. Transmission Loss
fn tvg(distance: f32) -> f32 {
    1.0 + 0.15 * distance  // Simple linear TVG
}

// 2. Rayleigh Noise
fn rayleigh() -> f32 {
    let u1 = thread_rng().gen_range(0.001..1.0);
    let u2 = thread_rng().gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

// 3. Gaussian Beam
fn beam_gaussian(angle: f32, half_angle: f32) -> f32 {
    let norm = angle / half_angle;
    (-2.77 * norm * norm).exp()
}

// In the fish rendering loop, replace:
let beam_factor = if angle < half_angle_rad {
    (1.0 - (angle / half_angle_rad)).powi(2)
} else {
    (1.0 - (angle / side_lobe_angle)).powi(2) * 0.1
};

// With:
let beam_factor = beam_gaussian(angle, half_angle_rad);

// Add TVG and noise to intensity:
let tvg_gain = tvg(distance);
let noise = rayleigh() * 0.05;
let intensity_120 = ((ts_120 + 65.0) / 45.0).clamp(0.0, 1.0)
    * beam_factor * pulse_factor * tilt_factor * ts_variation_120
    * tvg_gain + noise;
```

**Result**: Immediately more realistic echogram with minimal code changes.

---

## Validation: Compare to Real Echosounder

After implementing improvements, compare your echogram to real data:

1. **EK80** (commercial scientific echosounder) example images
2. **Simrad** echograms from fisheries research
3. Look for:
   - Similar noise texture
   - Realistic beam width
   - Proper depth attenuation
   - Bottom return characteristics

---

## Impact on Model Accuracy

Better realism should **improve sim-to-real transfer**:

- Model learns features that generalize to real echosounders
- Reduces "synthetic gap" between simulation and reality
- Better preparation for field deployment

**Trade-off**: Too much realism might make training harder. Start with high-priority items and test model accuracy after each change.
