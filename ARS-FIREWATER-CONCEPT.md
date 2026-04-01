# ARS Sensor Network on Fire Ring Main — Concept Note

## Deployment: 6x Clamp-On ARS Sensors

### Sensor Placement (typical offshore rig ring main)

```
                    HELIDECK
                   [Sensor 5]
                       |
          .-----------[ ]-----------.
         |         (branch)          |
    [Sensor 4]                  [Sensor 6]
         |                           |
    PORT SIDE                   STARBOARD
    (monitors,                  (monitors,
     hydrants)                   hydrants)
         |                           |
    [Sensor 3]                  [Sensor 1]
         |                           |
          '-----------[ ]-----------.
                       |
                  [Sensor 2]
               PUMP DISCHARGE
            (jockey + main FW pumps)
```

Each sensor: Pi 5 + ATEX-rated accelerometer + clamp mount on ring main pipe.
Pipe: 6" / 8" carbon steel Sch40, seawater at 8-15 bar static.

---

## What 6 Sensors Detect (that individual pressure transmitters cannot)

### 1. SPATIAL PRESSURE MAPPING (leak localisation)

With 6 sensors around the loop, each measuring local pipe pressure
non-invasively, SAIREN can compute a **pressure gradient map** of the
entire ring main.

- Normal (static, no flow): all 6 sensors read ≈ same pressure.
  Any deviation > noise floor indicates a local anomaly.

- Leak present: the two sensors closest to the leak will see the
  fastest pressure decay. By comparing decay rates across all 6,
  SAIREN triangulates the leak to a specific pipe segment.

  Example: If Sensor 3 and Sensor 4 both show fast decay but
  Sensor 1 and Sensor 6 show slow decay, the leak is on the
  port-side run between those sensor pairs.

- This replaces the current pitch approach of "pressure gradient
  between two header points during a flow test reveals restrictions"
  (section 3.5) — except now it works CONTINUOUSLY, not just during
  scheduled flow tests.

### 2. VALVE STATUS INFERENCE (closed/partial isolation detection)

A closed or partially-closed isolation valve between two adjacent
sensors creates a measurable acoustic impedance discontinuity:

- Fully open valve: acoustic resonance propagates freely; both
  sensors see the same spectral signature.
- Partially closed valve: reflected acoustic energy creates a
  standing wave pattern detectable as spectral peak splitting or
  new resonance modes between the two sensors.
- Fully closed valve: the pipe section becomes acoustically isolated;
  its resonance spectrum shifts (shorter effective cavity length).

This directly addresses the pitch scenario: "partially closed valve
seen as normal in DCS but detected by pressure differential in SAIREN."
ARS detects it WITHOUT requiring a pressure differential (which only
appears during flow) — it sees the acoustic impedance change even
under static conditions.

### 3. JOCKEY PUMP CYCLING CORRELATION

The jockey pump cycling analysis (pitch section 2) currently uses a
single pressure transmitter on the jockey sensing line. With 6 ARS
sensors:

- Pressure decay rate is measured at 6 points simultaneously.
- Jockey pump discharge pulse propagation is visible as a travelling
  pressure wave: Sensor 2 (nearest pump) sees it first, then 1&3,
  then 4&6, then 5. The wave speed confirms the pipe is full of
  liquid (not air-locked).
- If one sensor sees delayed or attenuated response to jockey pulse,
  there is a restriction or partial blockage between it and the pump.

### 4. DEADLEG DETECTION

Deadlegs (stagnant pipe sections where flow doesn't reach) are a
serious corrosion and MIC (microbiologically influenced corrosion)
risk on seawater systems. They also freeze in cold weather.

ARS detects deadlegs because:
- Stagnant water has different dissolved gas content → different
  effective bulk modulus → different resonance spectrum.
- No flow-induced vibration signature (while adjacent sensors on
  the main loop show flow signatures during pump test).
- Temperature-driven spectral drift (if the deadleg heats up or
  cools down differently from the flowing main).

### 5. WATER HAMMER AND TRANSIENT DETECTION

Fast valve closures or pump trips create pressure transients
(water hammer) that propagate around the ring main at ~1400 m/s.

With 6 sensors sampling at high rate (accelerometers can do 10 kHz+):
- Time-of-arrival differences pinpoint the SOURCE of the transient.
- Peak pressure at each sensor maps the severity distribution.
- Repeated transients from the same source indicate a sticky valve
  or control issue.

This is a bonus capability not in the current pitch — it's free
because the accelerometers are already there.

### 6. PIPE WALL THICKNESS TRENDING

As a secondary measurement: the pipe wall itself has resonance modes
(radial/circumferential) in the ultrasonic band. Wall thinning from
corrosion or erosion shifts these modes to lower frequencies.

Over months/years, the ARS sensors can trend wall thickness at each
sensor location — non-invasive, no NDT crew required, continuous
rather than periodic.

---

## Comparison: ARS vs Conventional Instrumentation

| Capability | Conventional (pitch v1) | 6x ARS Sensors |
|---|---|---|
| Pressure measurement | Pipe tapping required | Clamp-on, no penetration |
| System impairment | Required for installation | None |
| Leak detection | Yes (single point) | Yes + spatial localisation |
| Leak localisation | Manual inspection | Automated (triangulation) |
| Valve status | Position switch on each valve | Acoustic inference (no wiring to valve) |
| Blockage detection | During flow test only | Continuous (acoustic impedance) |
| Deadleg detection | Not available | Yes (spectral signature) |
| Water hammer | Not detected | Source localisation + severity |
| Wall thickness | Periodic NDT campaign | Continuous trending |
| Sensors needed | ~5 pressure + valve switches | 6 accelerometers (one type) |
| Install complexity | Pipe tapping, hot work, isolation | Clamp + cable, cold work only |

---

## Integration with SAIREN Multi-Agent Pipeline

```
6x ARS-PISNN Edge Nodes (Pi 5)
    |
    | P2P gossip mesh (model weights, anomaly alerts)
    |
    v
SAIREN Tactical Agent (Layer 1)
    - Continuous pressure map from 6 ARS readings
    - Decay rate analysis per segment (not per point)
    - Acoustic impedance change → valve status inference
    - Transient detection + source localisation
    |
    v
SAIREN Strategic Agent (Layer 2)
    - Cross-reference: ARS pressure map vs jockey cycling vs pump current
    - Eliminate false positives (e.g., planned flow test vs real leak)
    - Fault localisation: which pipe segment, which branch
    |
    v
SAIREN LLM Diagnostic Agent (Layer 3)
    - "Ring main pressure is decaying at 0.4 bar/min between
       Sensor 3 and Sensor 4 (port side, frames 12-18).
       All other segments stable. Jockey pump current normal.
       Probable leak on port-side header between frame 12 and 18.
       Recommend visual inspection of hydrant H-14 isolation valve
       and test drain V-FW-034."
```

---

## Physics: ARS on a Firewater Ring Main

| Parameter | Value |
|---|---|
| Pipe | 6" Sch40 carbon steel (OD 168.3 mm, wall 7.11 mm) |
| Medium | Seawater (ρ ≈ 1025 kg/m³, B ≈ 2.34 GPa) |
| Operating pressure | 8-15 bar (120-220 PSI) |
| Effective sound speed | ~1480 m/s (seawater at ambient) |
| Sensor spacing | ~15-30 m (depends on rig layout) |
| Resonance length | Distance between flanges/branches (~2-8 m) |
| Fundamental frequency | ~90-370 Hz (depends on section length) |
| Pressure sensitivity | ΔB/ΔP ≈ 5-12, giving Δf/f ≈ 0.01% per bar |

The pressure range (8-15 bar) is much smaller than drilling (0-15000 PSI),
so the frequency shift per bar is smaller. However:
- We don't need absolute pressure — we need RELATIVE changes and SPATIAL
  gradients between sensors. These are much easier to detect.
- The temporal signature (decay rate) is the primary diagnostic, not
  absolute pressure accuracy.
- For leak localisation, we're comparing 6 sensors against each other
  (differential measurement), which cancels systematic errors.

---

## Business Case

The ARS approach turns what was a "we need to isolate the ring main to
install pressure tappings" conversation into a "we'll clamp these on
during the next shift, no isolation needed, you'll have a spatial
pressure map by tomorrow."

For the OIM: zero system impairment during install.
For the mechanic: tells them WHERE to look, not just THAT there's a problem.
For the safety case: adds monitoring without touching the certified system.
