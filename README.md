# SAIREN Guardian

**Non-invasive pipe pressure monitoring via Acoustic Resonance Spectroscopy, Physics-Informed Spiking Neural Networks, gossip consensus, and a Closed-form Continuous-time (CfC) judge -- all in a single pure-NumPy file.**

SAIREN Guardian is a self-contained prototype for edge-native acoustic pressure estimation and anomaly detection on fluid-filled pipe systems. It was designed for offshore fire ring mains but the physics and architecture generalise to any fluid-filled piping (drilling risers, process lines, subsea flowlines).

Everything -- data generation, model training, online learning, six-node gossip mesh, CfC temporal arbiter, pump health monitoring, plotting, and serialisation -- lives in one file (`ars_pisnn_prototype.py`, ~3,800 lines) with zero GPU dependencies. It runs on a Raspberry Pi 5.

---

## Table of Contents

- [Quick Start](#quick-start)
- [What It Does](#what-it-does)
- [The Physics](#the-physics)
- [Architecture Overview](#architecture-overview)
  - [Layer 0 -- Physics-Informed SNN (PI-SNN)](#layer-0----physics-informed-snn-pi-snn)
  - [Layer 1 -- Gossip Micro-Mesh](#layer-1----gossip-micro-mesh)
  - [Layer 2 -- CfC Judge](#layer-2----cfc-judge)
  - [Pump Health Monitoring](#pump-health-monitoring)
- [Demo Scenarios](#demo-scenarios)
  - [Single-Node Demo](#single-node-demo)
  - [Mesh Demo](#mesh-demo)
  - [Pump Health Demo](#pump-health-demo)
- [Results](#results)
- [File Structure](#file-structure)
- [Configuration](#configuration)
- [Key Design Decisions](#key-design-decisions)
- [Dependencies](#dependencies)
- [Related Documents](#related-documents)

---

## Quick Start

```bash
# Install dependencies (no GPU required)
pip install numpy scipy matplotlib rich

# Run the single-node demo (drilling pipe, 0-15,000 PSI)
python ars_pisnn_prototype.py

# Run the 6-node gossip mesh demo (fire ring main, 120-220 PSI)
python ars_pisnn_prototype.py --mesh

# Run the pump health monitoring demo (jockey pump degradation + mesh)
python ars_pisnn_prototype.py --pump
```

All demos are fully self-contained. They generate synthetic data, train models, stream inference with anomaly injection, and produce plots in `output/`.

---

## What It Does

Clamp an accelerometer to a pipe. The fluid inside has acoustic resonances whose frequencies shift with internal pressure. SAIREN Guardian:

1. **Captures** the acoustic resonance spectrum (simulated as synthetic FFT magnitude vectors).
2. **Infers** internal pressure from the spectrum using a physics-informed neural network -- no pipe penetration, no pressure tapping, no hot work.
3. **Detects** anomalies via dual signals: spike-rate deviation and physics residual violation.
4. **Confirms** anomalies across a 6-node gossip mesh -- nodes vote to distinguish local leaks from sensor faults.
5. **Filters** false positives with a CfC temporal arbiter that learns the mesh's temporal dynamics and overrides gossip verdicts during recovery transitions.

The value proposition for offshore: zero system impairment during installation, spatial leak localisation (not just "there's a problem" but "between frame 12 and 18, port side"), and continuous monitoring that replaces periodic flow tests.

---

## The Physics

A fluid-filled cylindrical pipe has longitudinal acoustic resonances at:

```
f_n = (n / 2L) * sqrt(B_eff / rho_eff)
```

| Symbol | Meaning |
|--------|---------|
| `f_n` | Frequency of the nth resonance mode (Hz) |
| `n` | Mode number (1, 2, 3, ...) |
| `L` | Pipe section length between reflectors (m) |
| `B_eff` | Effective bulk modulus of the pipe-fluid system (Pa) |
| `rho_eff` | Effective density accounting for pipe wall inertia (kg/m^3) |

Internal pressure changes `B_eff` -- higher pressure stiffens the fluid, shifting resonance peaks upward. The PI-SNN encodes this relationship as a physics loss that penalises predictions violating the resonance equation.

**Effective bulk modulus** combines fluid compressibility and pipe wall compliance (thin-shell approximation):

```
1/B_eff = 1/B_fluid(P) + D / (E * t)
```

where `D` is the pipe inner diameter, `E` is Young's modulus, and `t` is wall thickness. `B_fluid` increases linearly with pressure, accounting for dissolved gas compressibility.

**Effective density** adds pipe wall inertia to fluid density, weighted by the ratio of wall to fluid cross-sectional areas.

### Fire Ring Main Parameters

| Parameter | Value |
|-----------|-------|
| Pipe | 6" Sch40 carbon steel (OD 168.3 mm, wall 7.11 mm) |
| Medium | Seawater (rho = 1025 kg/m^3, B = 2.34 GPa) |
| Operating pressure | 8-15 bar (120-220 PSI) |
| Sound speed | ~1480 m/s |
| Sensor spacing | ~15-30 m around ring |
| Section resonance | ~90-370 Hz fundamental (depends on flange spacing) |

---

## Architecture Overview

The system has three layers, each adding capability on top of the previous:

```
Layer 2:  CfC Judge (temporal arbiter)
              |
              | reviews gossip verdicts using prediction surprise
              |
Layer 1:  Gossip Micro-Mesh (6 nodes, consensus protocol)
              |
              | nodes broadcast anomalies, neighbors vote CONFIRM/DENY
              |
Layer 0:  PI-SNN Nodes (one per sensor, physics-informed inference)
              |
              | spectrum -> pressure estimate + anomaly score
              |
          Accelerometers (clamp-on, one per pipe section)
```

### Layer 0 -- Physics-Informed SNN (PI-SNN)

Each sensor node runs a Physics-Informed Spiking Neural Network that maps an acoustic spectrum to a pressure estimate.

**Architecture:** `512 -> 256 (ReLU) -> 128 (ReLU) -> 1 (linear)`

**SNN equivalence:** For rate-coded inputs (each sample is a static spectrum), a recurrent LIF network's steady-state spike rates are functionally equivalent to a feedforward network with sigmoid/ReLU activations. The prototype trains using the equilibrium-equivalent ReLU formulation for reliable gradient flow, with explicit LIF simulation at inference time for spike-rate anomaly detection. This is standard ANN-to-SNN conversion practice (Diehl et al. 2015, Sengupta et al. 2019). For neuromorphic hardware deployment, the trained weights map directly to LIF synapse weights.

**Training loss:**

```
L = MSE(pred, target) + lambda * physics_residual
```

The physics residual finds the dominant spectral peak, computes the expected frequency from the predicted pressure via the resonance equation, and penalises the squared relative error. This regularises the network to respect the known physics even when data is noisy.

**Online learning:** An OTTT-inspired (Online Training Through Time) loop runs continuously after initial batch training. A fixed-size circular buffer (500 samples) stores recent spectra and pressures. Every 50 steps, a mini-batch is sampled for one gradient update. Memory footprint is fixed -- the buffer overwrites oldest samples, preventing growth on resource-constrained edge devices.

**Anomaly detection (dual signal):**

| Signal | What it detects | How |
|--------|----------------|-----|
| Spike-rate z-score | Unusual neural activation patterns | EMA of per-neuron activation rates; z-score against running statistics |
| Physics residual | Pressure predictions violating acoustics | Dominant spectral peak vs resonance equation prediction |

Both signals are combined into a scalar anomaly score (0-1). A physics violation flag fires independently when the residual exceeds a configurable threshold.

**Columnar output weights:** Each node's output layer specialises to its pipe section geometry (different flange spacing = different resonance modes). Hidden layers (the "encoder") are shared across the mesh via federated averaging. Output columns are JSON-serialisable for P2P gossip distribution.

### Layer 1 -- Gossip Micro-Mesh

Six PI-SNN nodes are deployed around a fire ring main loop. Each node runs independently, but when one detects a pressure anomaly, it broadcasts to the mesh. Neighbors vote to confirm or deny.

**Sensor layout (typical offshore rig):**

```
                    HELIDECK
                   [Node 4]
                       |
          .-----------[ ]-----------.
         |         (branch)          |
    [Node 3]                    [Node 5]
         |                           |
    PORT SIDE                   STARBOARD
         |                           |
    [Node 2]                    [Node 0]
         |                           |
          '-----------[ ]-----------.
                       |
                  [Node 1]
               PUMP DISCHARGE
```

**Gossip trigger:** A dual-EMA (Exponential Moving Average) approach on the node's pressure reading. A fast EMA (alpha=0.3, ~3-sample window) tracks current pressure; a slow EMA (alpha=0.02, ~50-sample window) holds the baseline. When the deviation between them exceeds a threshold for 3 consecutive samples, the node initiates a gossip round. This is decoupled from the SNN anomaly score to avoid false triggers from model prediction noise.

**Voting logic:** Each node votes based on its distance from the origin and its own pressure readings:

| Distance | Own reading elevated | Own reading normal |
|----------|---------------------|-------------------|
| Nearby (< 30m) | CONFIRM | DENY |
| Medium (30-60m) | CONFIRM (lower weight) | ABSTAIN |
| Far (> 60m) | CONFIRM (lowest weight) | ABSTAIN |

Votes are weighted by the voter's confidence (its own gossip score). A quorum among non-abstaining voters is required to confirm. If 4+ of 5 voters confirm, the event is flagged **SYSTEM-WIDE** (catastrophic failure vs local leak).

**Post-confirmation cooldown:** After any confirmation (genuine or overridden), gossip is suppressed mesh-wide for 20 timesteps. This prevents the slow EMA's lag from re-triggering stale events during recovery.

**Federated encoder sync:** Every 100 timesteps, hidden-layer weights are averaged across all nodes (federated averaging). Output columns remain node-specific.

### Layer 2 -- CfC Judge

A Closed-form Continuous-time (CfC) neural network (Hasani et al. 2022) sits above the gossip mesh as a temporal arbiter. It receives the full 6-node mesh state every timestep, learns to predict the next state (self-supervised), and uses prediction surprise to override gossip verdicts.

**Why CfC:** The time-dependent forget gate (`f = sigmoid(-(dt * tau) * input)`) gives the network natural temporal context. It can distinguish a genuine anomaly onset (the mesh state is surprising -- high prediction error) from a recovery transition (the CfC has been tracking the recovery trajectory -- low surprise, pressures rising). Individual gossip nodes lack this mesh-wide temporal perspective.

**Architecture:** Neural Circuit Policy (NCP) wiring with sparse connectivity:

```
Sensory (24) -> Inter (12) -> Command (8) -> Motor (4) -> Output (24)
    ^                ^
    |                |
  24 features      recurrent
  from mesh        connections
```

| Feature block | Indices | Content |
|--------------|---------|---------|
| Pressures | 0-5 | True pressure at each node |
| Gossip scores | 6-11 | Dual-EMA deviation per node |
| Anomaly scores | 12-17 | SNN anomaly score per node |
| Delta-pressures | 18-23 | Pressure change from previous step |

**Self-supervised learning:** The CfC predicts the next timestep's 24-feature vector. At each step, the prediction error from the previous step becomes the training signal. No labels required -- the network learns "what normal mesh dynamics look like" purely from temporal self-prediction. Training uses truncated BPTT (depth=4) with exponential decay, Adam optimiser, and gradient clipping. Learning rate decays from 0.001 to a floor of 0.0001.

**Surprise-based verdict override:**

```
current_surprise = mean(|prediction - actual|)
```

The surprise score is tracked in a rolling buffer (500 samples). Percentile thresholds define "low" (25th percentile) and "high" (90th percentile) surprise.

| Gossip says | CfC observes | CfC verdict | Reason |
|-------------|-------------|-------------|--------|
| CONFIRMED | High surprise | CONFIRMED | Genuine anomaly -- mesh state is unexpected |
| CONFIRMED | Low/moderate surprise + majority of nodes show rising pressure | **DENIED** | Recovery transition -- pressures recovering, not failing |
| CONFIRMED | Low/moderate surprise + recent prior confirmation (< 100 steps) | **DENIED** | Confirmation fatigue -- slow EMA hasn't settled yet |
| CONFIRMED | Very low surprise (< 50% of low threshold) | **DENIED** | CfC has fully learned this pattern -- it's normal |
| DENIED | Very high surprise (> 1.5x high threshold) | **CONFIRMED** | Potential missed event -- CfC sees something the gossip doesn't |
| DENIED | Normal surprise | DENIED | Agrees with gossip |

### Pump Health Monitoring

A tri-modal condition monitoring system for the jockey pump (Node 1, pump discharge). It fuses vibration analysis, temperature trending, and Motor Current Signature Analysis (MCSA) into a composite health index that feeds into the gossip mesh.

**Why the jockey pump matters:** On an offshore fire ring main, the jockey pump runs almost continuously to maintain static pressure and compensate for minor leaks. It's the most mechanically stressed component in the loop. If it fails silently, the ring main depressurises gradually -- a condition the pressure sensors will eventually detect, but only after the situation has progressed. Monitoring the pump directly catches degradation weeks before it becomes a pressure event.

**Sensor suite (all non-invasive):**

| Sensor | Mounting | Sample rate | Measures |
|--------|----------|-------------|----------|
| Accelerometer | Bearing housing (DE/NDE) | 2 kHz | Vibration spectrum |
| RTD/thermocouple | Bearing housing, motor winding, seal | 1 Hz | Temperature trends |
| Current clamp | Motor supply cable | 2 kHz | Motor current spectrum |

**Vibration analysis:**

The vibration analyzer computes bearing defect frequencies from first principles using the 6205 bearing geometry (9 balls, 7.938 mm ball diameter, 38.5 mm pitch diameter, 0° contact angle):

| Frequency | Formula | Value (2950 RPM) |
|-----------|---------|-------------------|
| BPFO (outer race) | `(n/2) * f_s * (1 - Bd/Pd * cos(α))` | 175.6 Hz |
| BPFI (inner race) | `(n/2) * f_s * (1 + Bd/Pd * cos(α))` | 266.9 Hz |
| BSF (ball spin) | `(Pd/2Bd) * f_s * (1 - (Bd/Pd)^2 * cos²(α))` | 114.2 Hz |
| FTF (cage) | `(f_s/2) * (1 - Bd/Pd * cos(α))` | 19.5 Hz |

BPFO peaks are monitored for signal-to-noise ratio against the local noise floor. Shaft-frequency harmonics (1x, 2x) indicate imbalance -- but since 1x is always present in a running pump (healthy SNR ~5-8), the threshold is set high (SNR > 8.0) to avoid false alarms. Overall vibration severity follows ISO 10816-3 (Group 2, rigid mounting): alarm at 4.5 mm/s RMS, trip at 7.1 mm/s.

**Temperature analysis:**

Four temperature channels (DE bearing, NDE bearing, motor winding, seal) are compared against nominal baselines. Scoring uses delta-from-nominal with alarm (+15°C) and trip (+25°C) thresholds. Temperature evolution includes thermal time constants (~200 s for bearings, ~150 s for winding) to model realistic lag.

**Motor Current Signature Analysis (MCSA):**

The motor current spectrum is analyzed for:

- **Broken rotor bars:** Sidebands at `f_line ± 2*s*f_line` (48.3/51.7 Hz for 50 Hz supply, slip=0.017). Sideband-to-fundamental ratio > -40 dB indicates fault.
- **Eccentricity:** Sidebands at `f_line ± f_shaft` (1.3/98.7 Hz). Indicates air gap non-uniformity from bearing wear.

A noise-floor correction subtracts the median amplitude in a band around each sideband frequency (excluding the peak itself) before computing ratios. This eliminates false positives from the 50 Hz fundamental's spectral leakage into nearby bins.

**Composite health index:**

```
health = 0.4 * vibration + 0.3 * temperature + 0.3 * mcsa
```

With a critical-override rule: if any single modality score drops below 0.2, the composite is capped at 0.3 regardless of the weighted sum. This prevents a critically failed subsystem from being masked by healthy readings on the other two channels.

| Health range | Status | Action |
|-------------|--------|--------|
| 0.65 - 1.0 | Healthy | Normal operation |
| 0.40 - 0.65 | Alarm | Schedule inspection |
| 0.20 - 0.40 | Critical | Plan maintenance |
| 0.00 - 0.20 | Trip | Auto-shutdown, start fire pump |

**Mesh integration:** When the pump trips, the pressure dynamics at Node 1 change -- discharge pressure drops to ambient, and the fire pump auto-starts 20 steps later with a different pressure profile. These pressure transients propagate through the gossip mesh and CfC judge, testing the full system's ability to distinguish pump failure from pipe failure.

---

## Demo Scenarios

### Single-Node Demo

```bash
python ars_pisnn_prototype.py
```

Simulates a single ARS sensor on a drilling pipe (0-15,000 PSI range):

| Phase | Time steps | What happens |
|-------|-----------|--------------|
| Ramp | 0-500 | Pressure ramps 5,000 -> 10,000 PSI |
| Anomaly | 500-530 | Spike to 18,000 PSI (simulated kick) |
| Recovery | 530-580 | Pressure returns to 10,000 PSI |
| Stable | 580-800 | Steady state |

**Outputs** (in `output/`):
- `ars_pisnn_results.png` -- 3-panel plot: pressure tracking, spike-rate heatmap, physics residual
- `training_curve.png` -- test RMSE over training epochs
- `column_weights.json` -- serialised output-layer weights for gossip distribution

### Mesh Demo

```bash
python ars_pisnn_prototype.py --mesh
```

Simulates 6 ARS sensors on a fire ring main (120-220 PSI, seawater):

| Phase | Time steps | What happens |
|-------|-----------|--------------|
| Jockey fill | 0-200 | Pressure ramps 130 -> 150 PSI |
| Normal static | 200-400 | Stable at 150 PSI |
| **Local leak** | 400-450 | 35 PSI drop near Node 3 (port-side fwd), attenuating with distance |
| Leak recovery | 450-550 | Jockey pump restores pressure |
| Normal static | 550-700 | Stable at 150 PSI |
| **Catastrophic** | 700-740 | 70 PSI drop across all nodes (main pipe rupture) |
| Fire pump recovery | 740-850 | Partial recovery to 130 PSI (fire pump auto-start) |
| Post-event | 850-1000 | Stable at 130 PSI (lower setpoint) |

**Spatial pressure propagation:** The local leak drops pressure at the leak node by the full amount, with exponential attenuation along the ring (`exp(-0.05 * distance_m)`). Nearby nodes see a large drop; far nodes see almost nothing. This is what allows the gossip protocol to distinguish local leaks from catastrophic events.

**Outputs** (in `output/`):
- `mesh_results.png` -- 5-panel plot (see below)
- `mesh_state.json` -- full event log with gossip verdicts and CfC override details

**Plot panels:**

| Panel | Title | Shows |
|-------|-------|-------|
| (a) | Spatial Pressure Distribution | True pressure at all 6 nodes -- leak localisation visible as differential drop |
| (b) | True vs Predicted | PI-SNN tracking accuracy at leak node (N3) and far node (N0) |
| (c) | Per-Node Anomaly Scores | Dual-signal anomaly scores; gossip trigger threshold shown |
| (d) | Gossip Consensus Timeline | Orange circles = gossip start, red triangles = confirmed, green = denied. Magenta edge = CfC override |
| (e) | CfC Judge Surprise | CfC prediction surprise over time. Green diamonds = CONFIRMED->DENIED overrides. Red/green dashed lines = high/low surprise thresholds |

### Pump Health Demo

```bash
python ars_pisnn_prototype.py --pump
```

Simulates the jockey pump degrading over 1,200 timesteps while the full 6-node mesh runs concurrently:

| Phase | Time steps | What happens |
|-------|-----------|--------------|
| Jockey fill | 0-200 | Pressure ramps 130 -> 150 PSI, pump healthy |
| Normal static | 200-300 | Stable at 150 PSI, all systems nominal |
| **Degradation onset** | 300-900 | Linear severity ramp 0.0 -> 1.0 (bearing wear, winding heat, rotor bar cracking) |
| Alarm threshold | ~657 | Health index crosses 0.65 -- bearing BPFO peaks visible |
| Critical threshold | ~819 | Health index crosses 0.40 -- MCSA broken bar sidebands prominent |
| **Pump trip** | 900 | Health < 0.30 -- auto-shutdown. Discharge pressure drops to ambient |
| **Fire pump start** | 920 | Fire pump auto-starts, pressure recovers to ~130 PSI |
| Post-event | 920-1200 | Stable on fire pump, jockey pump offline |

**Outputs** (in `output/`):
- `pump_health_results.png` -- 10-panel plot (5x2 grid, see below)
- `pump_health_state.json` -- event log with health scores, fault diagnoses, and gossip verdicts

**Plot panels (5x2 grid):**

| Panel | Title | Shows |
|-------|-------|-------|
| (a) | Vibration Spectrum | BPFO peak growth over time as bearing degrades |
| (b) | Vibration Health | RMS velocity, ISO 10816-3 thresholds (alarm/trip), bearing SNR |
| (c) | Temperature Trends | DE/NDE bearing, winding, seal temperatures vs alarm/trip deltas |
| (d) | Temperature Health | Per-channel and composite temperature health score |
| (e) | Motor Current Spectrum | 50 Hz fundamental with broken rotor bar sidebands emerging |
| (f) | MCSA Health | Sideband-to-fundamental ratio, broken bar and eccentricity scores |
| (g) | Composite Health Index | Weighted fusion of all three modalities with alarm/critical/trip thresholds |
| (h) | Fault Diagnosis | Active fault flags over time (bearing, imbalance, thermal, broken bar, eccentricity) |
| (i) | Spatial Pressure | 6-node pressure distribution including pump trip transient |
| (j) | Gossip + CfC | Gossip verdicts and CfC overrides during pump failure event |

---

## Results

### Single-Node Performance

| Metric | Value | Target |
|--------|-------|--------|
| Batch test RMSE | 14.8 PSI | < 500 PSI |
| Anomaly detection delay | 0 samples | < 5 samples |
| Physics violation flagged | Yes | Yes |
| Buffer memory | Fixed (~2 KB) | No growth |

### Mesh Performance

| Metric | Value |
|--------|-------|
| Local leak confirmed | Yes, t=402 (2-step delay from onset at t=400) |
| Catastrophic confirmed | Yes, t=706, flagged SYSTEM-WIDE (6-step delay) |
| Total gossip rounds | 45 |
| Raw gossip confirmations | 44 (many false positives during recovery) |
| **After CfC judge** | **14 confirmed, 31 denied** |
| CfC overrides | 30 (all CONFIRMED -> DENIED during recovery) |
| False confirmations during normal operation | 0 |
| Per-node RMSE (normal regime) | 12.9 - 78.1 PSI |
| Buffer memory per node | ~1 KB (fixed) |

The CfC judge reduces the false alarm rate by **68%** (44 -> 14 confirmed events) while preserving 100% true positive detection. All overrides are during recovery transitions where the gossip protocol's slow EMA hasn't yet settled.

### Pump Health Performance

| Metric | Value |
|--------|-------|
| Initial health (t=0) | 0.85 (healthy) |
| Alarm threshold crossed | t=657 (health=0.65) |
| Critical threshold crossed | t=819 (health=0.39) |
| Pump trip | t=900 (health=0.29) |
| Fire pump auto-start | t=920 |
| First fault detected | Bearing defect (BPFO SNR rise) |
| MCSA broken bar detection | Sideband ratio > -40 dB at ~t=750 |
| False positives at severity=0 | 0 (noise-floor subtraction eliminates spectral leakage artifacts) |

The degradation arc tracks realistically: vibration (bearing wear) leads, followed by temperature (thermal lag from time constants), then MCSA (broken bar sidebands grow slowly). The critical-override rule ensures that even if two modalities read healthy, a single critically failed channel caps the composite score.

---

## File Structure

```
sairen-guardian/
|-- ars_pisnn_prototype.py          # Everything (~3,800 lines, single file)
|-- ARS-FIREWATER-CONCEPT.md        # 6-sensor deployment concept note
|-- SAIREN-firewater-pitch.md       # Business pitch for fire ring main monitoring
|-- SAIREN-FireMain-Pilot-Spec.md   # Pilot installation specification
|-- Physics-Based Spiking Neural    # Research note on PI-SNN replacing KNN
|   Networks as a Drop-In...md
|-- output/
|   |-- ars_pisnn_results.png       # Single-node demo plots
|   |-- training_curve.png          # Training convergence
|   |-- column_weights.json         # Serialised output weights
|   |-- mesh_results.png            # 5-panel mesh demo plots
|   |-- mesh_state.json             # Full mesh event log + CfC verdicts
|   |-- pump_health_results.png     # 10-panel pump health demo plots
|   |-- pump_health_state.json      # Pump health event log + fault diagnoses
```

### Code Organisation (within `ars_pisnn_prototype.py`)

| Section | Lines | Classes / Functions |
|---------|-------|-------------------|
| 1. Configuration | ~80 | `Config`, `MeshConfig` |
| 1b. Pump Health Config | ~80 | `PumpHealthConfig` (mechanical, bearing, MCSA, thermal, fusion params) |
| 2. Data Simulator | ~100 | `generate_spectrum()`, `generate_dataset()`, physics functions |
| 3. PI-SNN Model | ~250 | `PISNN` (forward, backward, LIF simulation, physics residual) |
| 4. Training Loop | ~60 | `train_model()` |
| 5. Online Learning | ~70 | `CircularBuffer`, `AnomalyDetector` |
| 6. Column Manager | ~60 | `ColumnManager` (JSON-serialisable output weights) |
| 7. Single-Node Demo | ~220 | `run_demo()` |
| 8. Gossip Mesh | ~450 | `SensorNode`, `GossipProtocol`, `GossipMessage/Vote/Round` |
| 8b. CfC Judge | ~400 | `NcpWiringCfC`, `CfcCell`, `CfcJudge`, `CfcForwardCache` |
| 8c. Pump Health | ~550 | `PumpPhysics`, `PumpSensorSimulator`, `PumpHealthAnalyzer` |
| 9. Mesh Orchestration | ~200 | `MeshNetwork`, `SpatialPressureSimulator` |
| 10. Mesh Demo | ~300 | `run_mesh_demo()` |
| 11. Pump Health Demo | ~250 | `run_pump_health_demo()` |

---

## Configuration

All parameters are centralised in three dataclasses. Nothing is hardcoded elsewhere.

### `Config` (single-node)

| Group | Key parameters | Defaults |
|-------|---------------|----------|
| Pipe geometry | `pipe_length`, `pipe_outer_diameter`, `pipe_wall_thickness` | 1.0 m, 4.5" OD, 0.37" wall |
| Fluid | `fluid_base_density`, `fluid_base_bulk_modulus`, `fluid_bulk_modulus_pressure_coeff` | 1200 kg/m^3, 2.2 GPa, 12.0 |
| Spectrum | `n_fft_bins`, `max_freq_hz`, `n_resonance_modes` | 512, 50 kHz, 20 |
| SNN | `snn_layers`, `snn_beta`, `snn_threshold` | [512,256,128,1], 0.80, 1.0 |
| Training | `learning_rate`, `n_epochs`, `physics_lambda` | 2e-3, 50, 0.1 |
| Online | `online_buffer_size`, `online_update_interval` | 500, 50 |
| Anomaly | `anomaly_zscore_threshold`, `physics_violation_threshold` | 3.0, 0.10 |

### `MeshConfig` (6-node mesh, extends `Config`)

| Group | Key parameters | Defaults |
|-------|---------------|----------|
| Ring geometry | `ring_main_length`, `section_lengths[]`, `node_positions[]` | 120 m loop, 2.8-6.5 m sections |
| Firewater pipe | `pipe_outer_diameter`, `fluid_base_density` | 6" Sch40, 1025 kg/m^3 seawater |
| Gossip | `gossip_trigger_threshold`, `gossip_trigger_consecutive`, `gossip_quorum` | 0.5, 3, 0.5 |
| Spatial | `leak_attenuation_per_m` | 0.05 /m |
| CfC Judge | `cfc_n_sensory/inter/command/motor`, `cfc_bptt_depth`, `cfc_surprise_*_pct` | 24/12/8/4, depth=4, 25th/90th |

### `PumpHealthConfig` (pump health monitoring)

| Group | Key parameters | Defaults |
|-------|---------------|----------|
| Motor | `pump_power_kw`, `pump_rpm`, `motor_poles`, `supply_freq_hz` | 7.5 kW, 2950 RPM, 2-pole, 50 Hz |
| Bearing (6205) | `bearing_n_balls`, `bearing_ball_dia_mm`, `bearing_pitch_dia_mm`, `bearing_contact_angle` | 9, 7.938 mm, 38.5 mm, 0° |
| MCSA | `motor_slip`, `mcsa_fft_bins`, `mcsa_max_freq_hz`, `broken_bar_threshold_db` | 0.017, 2048, 200 Hz, -40 dB |
| Temperature | `temp_nominal_*`, `temp_alarm_delta`, `temp_trip_delta` | DE=45°C, NDE=40°C, winding=65°C, seal=35°C; +15/+25°C |
| Vibration | `vib_alarm_rms`, `vib_trip_rms` | 4.5 mm/s, 7.1 mm/s (ISO 10816-3) |
| Fusion | `health_weight_vib/temp/mcsa`, `critical_override_threshold` | 0.4/0.3/0.3, 0.2 |
| Demo scenario | `pump_degradation_start/end`, `pump_trip_step`, `fire_pump_start_step` | 300/900, 900, 920 |

---

## Key Design Decisions

### Pure NumPy -- no PyTorch, no snnTorch

Every component -- forward/backward passes, Adam optimiser, LIF simulation, CfC gated dynamics, truncated BPTT, Welford normalisation -- is implemented from scratch in NumPy. This eliminates heavy dependencies for edge deployment on Raspberry Pi 5 and makes the entire signal processing chain auditable line by line.

### Equilibrium-equivalent SNN

Rather than training through LIF dynamics (which requires surrogate gradients and is notoriously unstable), the prototype trains a ReLU MLP that is mathematically equivalent to the LIF network's steady-state spike rates. LIF simulation runs at inference time only, for spike-rate anomaly detection. This gives reliable training with clean gradients while preserving the SNN's neuromorphic deployment path.

### Dual-EMA gossip trigger (not SNN anomaly score)

Early iterations triggered gossip from the SNN's spike-rate z-score anomaly detector. This caused false trigger floods because model prediction noise (especially on nodes with higher RMSE) constantly elevated z-scores. The solution decouples the gossip trigger: it uses a dedicated dual-EMA on the raw pressure reading (fast EMA vs slow EMA), which only fires on genuine pressure transients. The SNN anomaly score is still computed and logged but doesn't drive gossip initiation.

### CfC as a temporal arbiter, not a replacement

The CfC doesn't replace the gossip protocol -- it sits above it as a judge. The gossip mesh handles spatial consensus (which nodes confirm?). The CfC adds temporal context (is this a new event or a recovery?). This separation of concerns means the gossip protocol can run even if the CfC is in warmup, and the CfC doesn't need to understand the spatial topology.

### Self-supervised CfC training

The CfC learns by predicting the next mesh state from the current one. No labels, no anomaly annotations. It naturally develops a model of "what normal dynamics look like" and flags deviations as surprising. During recovery transitions, the CfC quickly adapts to the new trajectory (pressure rising, gossip scores decaying), so surprise drops -- enabling it to override false confirmations. During genuine events, the mesh state is genuinely unprecedented, so surprise spikes.

### Cooldown on original verdict, not CfC override

When the CfC overrides a CONFIRMED to DENIED, the mesh cooldown still activates (based on the original gossip verdict). Without this, overridden rounds don't suppress subsequent gossip, causing a flood of rounds during recovery. The EMA deviation that triggered gossip is real (it just isn't a new event), and the slow EMA needs time to settle regardless of the CfC's assessment.

### Noise-floor subtraction for MCSA

The 50 Hz fundamental's spectral leakage (Gaussian tail with width=0.5 Hz) produces non-zero amplitude at broken bar sideband frequencies (48.3 Hz), which naive peak detection picks up as a fault. The fix measures the median amplitude in a band around each sideband (excluding the peak itself) as the local noise floor, then scores only the excess above noise. This eliminated false positive MCSA detections at zero fault severity.

### Tri-modal fusion with critical override

A weighted average of three health scores (vibration 40%, temperature 30%, MCSA 30%) can mask a critically failed subsystem. If a bearing is disintegrating (vibration health = 0.1) but the motor current and temperature look fine, the weighted average might still read 0.65 (healthy). The critical-override rule caps the composite at 0.3 whenever any single modality drops below 0.2 -- ensuring that catastrophic failure in any channel triggers an alarm regardless of the others.

### Imbalance threshold calibration

A running pump always has some 1x shaft-frequency vibration component (healthy SNR ~5-8). Early iterations flagged imbalance from step 0 because the threshold was too low (`SNR > 3.0`). The corrected threshold (`SNR > 8.0`) only triggers when shaft vibration grows well above the healthy baseline, matching real-world practice where 1x is monitored for trend changes, not absolute presence.

### Single file by design

For a prototype targeting edge deployment, having everything in one file means:
- One `scp` to deploy
- No import chains to debug on a headless Pi
- The entire system is auditable by reading one file top to bottom
- No risk of version mismatches between modules

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | >= 1.24 | All computation, neural networks, linear algebra |
| `scipy` | >= 1.10 | `find_peaks` for spectral peak detection |
| `matplotlib` | >= 3.7 | Plot generation (Agg backend, no display required) |
| `rich` | >= 13.0 | Terminal output formatting (tables, progress bars) |

No GPU. No PyTorch. No TensorFlow. No CUDA. Runs on any system with Python 3.10+.

---

## Related Documents

| Document | Description |
|----------|-------------|
| `ARS-FIREWATER-CONCEPT.md` | Detailed concept note for 6-sensor deployment on fire ring main: what the sensors detect, comparison with conventional instrumentation, physics parameters, integration with SAIREN multi-agent pipeline |
| `SAIREN-firewater-pitch.md` | Business pitch for fire ring main monitoring: jockey pump cycling, leak detection, pressure gradient analysis |
| `SAIREN-FireMain-Pilot-Spec.md` | Pilot installation specification: hardware, mounting, networking, acceptance criteria |
| `Physics-Based Spiking Neural Networks...md` | Research note on using PI-SNNs as a drop-in replacement for KNN in self-supervised online learning |
