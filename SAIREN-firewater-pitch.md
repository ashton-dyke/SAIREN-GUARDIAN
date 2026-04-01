# SAIREN — Firewater & Foam System Intelligent Monitoring
### Technical Instrument and Advisory Brief — For Review with Mechanical Team

---

## 1. Purpose

SAIREN is an edge-deployed, physics-based monitoring platform. It ingests continuous sensor data from your firewater and foam system, builds a statistical baseline of normal system behaviour, and issues **plain-language diagnostic advisories** that state **what is wrong, where the fault is located, and why SAIREN believes that to be the case** — before the problem is discovered on a flow test or, worse, during a real fire event.

SAIREN does **not** replace the certified fire and gas system, deluge controls, or manual override functions. It operates as a continuous readiness intelligence layer running in parallel on an isolated edge node.

---

## 2. Current Issue: Ring Main Leak — A Worked Example

**Observed behaviour on this rig:** Jockey pump runs more frequently than expected; ring main pressure does not hold between jockey cycles.

**What SAIREN would do with this:**

SAIREN establishes a baseline profile covering three parameters simultaneously:

| Parameter | Sensor Type | Location |
|---|---|---|
| Ring main static pressure | 4–20 mA pressure transmitter (0–25 bar range) | Tapping on jockey pump sensing line, between discharge NRV and main header isolation |
| Jockey pump run status | Digital input (volt-free contact) from MCC starter | Jockey MCC panel |
| Jockey pump motor current | Split-core current transformer (CT) on one supply phase | Jockey MCC, clamp around line conductor — no wiring modification required |

**SAIREN's reasoning process (three-layer pipeline):**

**Layer 1 — Tactical Physics Agent (runs every second):**
Monitors the rate of pressure decay between jockey pump starts. Computes ΔP/min on a rolling 5-minute window. Compares against trained baseline decay curve. If decay rate exceeds baseline by more than 2 standard deviations for longer than 10 minutes, a diagnostic ticket is raised.

**Layer 2 — Strategic Verification Agent:**
Cross-references the ticket against correlated signals. Asks: Is the jockey starting at its correct low setpoint? Is it restoring the correct high setpoint? Is run duration normal? Is motor current consistent with a healthy pump at this duty, or does current suggest reduced head (pump fault) vs. normal current suggesting ring main loss?

**Layer 3 — LLM Diagnostic Agent (Qwen / DeepSeek):**
Generates a plain-language advisory combining all verified signals. Example output:

> **ADVISORY — AMBER | Firewater Ring Main | Probable Leak**
>
> **WHAT:** Ring main pressure decay rate is 340% above the 30-day baseline. Jockey pump is cycling every 4.2 minutes (baseline: 18.7 minutes) and restoring pressure to the high setpoint on each start. Motor current on jockey start is 8.4 A (baseline: 8.5 A), consistent with a healthy pump at normal duty.
>
> **WHY:** Jockey pump performance is normal. The pressure loss is occurring in the ring main itself, not through pump degradation. The rapid repeat cycling confirms a continuous leak, not a one-off pressure transient. Estimated leak rate based on decay gradient: approximately 0.3–0.5 bar/min under static conditions.
>
> **WHERE:** Because jockey current is normal and the pump is restoring pressure correctly, the fault is downstream of the jockey discharge NRV. Recommend physical inspection starting with: (1) most recently disturbed pipework or connections on the ring main, (2) hydrant and monitor isolation valves for passing seats, (3) drain or test valves not fully closed.
>
> **CONSEQUENCE IF UNRESOLVED:** At current leak rate, the jockey will be unable to maintain ring main pressure if a second demand (e.g. manual hose station) is opened simultaneously. Main fire pumps will auto-start at the ring main low setpoint. If main pumps also fail to start or have degraded performance, ring main pressure will not be restored under fire demand conditions.

---

## 3. Full Sensor Schedule — All Monitored Subsystems

### 3.1 Surge Tank and Deepwell Pump

| Sensor | Type | Mounting Location | Diagnostic Purpose |
|---|---|---|---|
| Surge tank level | Ultrasonic level transmitter or submersible hydrostatic pressure sensor (0–5 m WG) | Top of surge tank (ultrasonic) or bottom drain port (hydrostatic) | Detects unexplained level loss (leak), failure of deepwell to refill, or water ingress if level rises unexpectedly |
| Deepwell pump motor current | Split-core CT on MCC supply conductor | Deepwell pump MCC | Tracks run frequency and duration; increasing run time to maintain level indicates degraded deepwell output or increased tank consumption |
| Surge tank outlet pressure | 4–20 mA pressure transmitter (0–10 bar) | Outlet manifold of surge tank | Confirms tank is pressurised and available to supply fire pump suctions on demand |

### 3.2 Jockey Pump

| Sensor | Type | Mounting Location | Diagnostic Purpose |
|---|---|---|---|
| Ring main static pressure | 4–20 mA pressure transmitter (0–25 bar) | Jockey sensing line tapping between discharge NRV and main header | Continuous pressure trending; decay rate analysis; leak signature detection |
| Jockey run status | Digital input (volt-free contact or current threshold from CT) | Jockey MCC starter | Start frequency and run duration profiling; abnormal cycling pattern detection |
| Jockey motor current | Split-core CT (clamp-on, single phase) | MCC enclosure, clamped on supply conductor | Distinguishes pump fault (current change) from ring main leak (normal current); confirms pump is loading correctly |

### 3.3 Main Fire Pumps (x2)

| Sensor | Type | Mounting Location | Diagnostic Purpose |
|---|---|---|---|
| Suction pressure | 4–20 mA pressure transmitter (0–10 bar) | Suction flange inlet spool or suction header tapping | Detects strainer blockage, low surge tank supply, NPSH risk; separates suction-side from discharge-side faults |
| Discharge pressure | 4–20 mA pressure transmitter (0–25 bar) | Pump discharge manifold before NRV | Tracks pressure rise curve and peak head on each test; degrading head at constant speed = internal hydraulic wear |
| Motor current | Split-core CT (clamp-on, all three phases) | Main pump MCC, clamped on supply conductors | Signature analysis across test cycles; elevated current at reduced head = mechanical friction; reduced current at reduced head = hydraulic loss |
| Pump run status | Digital input from MCC starter auxiliary contact | MCC panel | Confirms auto-start event; SAIREN measures delay between ring main setpoint trip and pump run confirmation |
| Vibration (optional — phase 2) | ATEX-rated triaxial accelerometer | Pump bearing housing | Bearing health monitoring; cavitation detection from vibration signature changes |

### 3.4 Foam Pumps and Proportioner (x2 foam pumps)

| Sensor | Type | Mounting Location | Diagnostic Purpose |
|---|---|---|---|
| Foam tank level | Ultrasonic level transmitter or float-type level switch with 4–20 mA output | Foam concentrate storage tank, top mounting for ultrasonic | Unexplained level drop = concentrate leak or valve not seating; level rise = water ingress through proportioner backflow |
| Foam tank temperature | RTD (Pt100) or Type-K thermocouple with transmitter | Tank wall, mid-height immersion | Tracks temperature history; extreme cycles indicate storage conditions outside concentrate specification |
| Foam pump suction pressure | 4–20 mA pressure transmitter (0–10 bar) | Suction inlet spool | Strainer blockage on concentrate side; distinguishes concentrate-supply fault from pump fault |
| Foam pump discharge pressure | 4–20 mA pressure transmitter (0–25 bar) | Discharge manifold before proportioner inlet | Pump head trend across test cycles; degrading head indicates pump wear |
| Proportioner inlet pressure (water side) | 4–20 mA pressure transmitter (0–25 bar) | Water inlet to proportioner | Confirms water-side pressure available to drive proportioning |
| Proportioner outlet pressure (mixed solution) | 4–20 mA pressure transmitter (0–25 bar) | Mixed foam-solution outlet from proportioner | ΔP across proportioner (inlet minus outlet) at a known test flow is the key indicator of proportioner wear or mis-setting |

**Key diagnostic logic for proportioner:**
SAIREN computes ΔP = P_water_inlet − P_mixed_outlet during each flow test and compares against historical values. A drift of >10% from baseline at the same test flow rate indicates the proportioner is not operating at its designed differential — the first early warning that the mixing ratio is drifting out of specification before it fails a foam quality test.

### 3.5 Ring Main and Monitor/Hydrant Branches

| Sensor | Type | Mounting Location | Diagnostic Purpose |
|---|---|---|---|
| Ring main pressure (main header) | 4–20 mA pressure transmitter (0–25 bar) | Two points on main ring header — port side and starboard (or forward/aft) | Pressure gradient between two header points during a flow test reveals restrictions or closed isolations on specific loops |
| Helideck branch pressure | 4–20 mA pressure transmitter (0–25 bar) | Helideck branch tapping upstream of branch isolation valve | Compares branch supply pressure to main header pressure; significant differential = restriction between header and helideck |
| Critical isolation valve position | Digital input (valve actuator position switch — open/closed) | Helideck branch isolation, main riser isolation | Confirms isolations are fully open; partially closed valve seen as normal in DCS but detected by pressure differential in SAIREN |
| Helideck monitor flow (test header) | Clamp-on ultrasonic flow transmitter (non-intrusive) | On the test drain or discharge line from helideck monitor during flow tests | Absolute flow measurement on each helideck test; trend of flow at constant pump speed reveals blockage, nozzle degradation, or monitor restriction |

---

## 4. SAIREN's Reasoning Process — How It Goes From Signal to Advisory

```
Raw sensor data (continuous, 1–10 Hz sampling)
          │
          ▼
┌─────────────────────────────────────────┐
│ LAYER 1: TACTICAL PHYSICS AGENT         │
│ • Computes rolling statistics per       │
│   sensor: mean, std dev, rate-of-change │
│ • Applies physics rules:                │
│   – ΔP/min > 2σ baseline → leak flag   │
│   – Pump ΔP at test flow down >10% →   │
│     hydraulic degradation flag          │
│   – Proportioner ΔP drift >10% →       │
│     proportioner flag                   │
│ • Fast, deterministic — no AI needed   │
│ • Raises a diagnostic ticket if rules   │
│   are satisfied                         │
└───────────────┬─────────────────────────┘
                │ Ticket raised
                ▼
┌─────────────────────────────────────────┐
│ LAYER 2: STRATEGIC VERIFICATION AGENT   │
│ • Cross-references ticket against all   │
│   correlated sensors                    │
│ • Eliminates false positives:           │
│   – Is this consistent with test mode?  │
│   – Does the jockey current confirm     │
│     pump health (separating pump fault  │
│     from ring main leak)?               │
│   – Does suction pressure explain the   │
│     discharge drop, or is the pump sick?│
│ • Assigns fault location from sensor    │
│   logic tree                            │
│ • Confirms or rejects ticket            │
└───────────────┬─────────────────────────┘
                │ Confirmed ticket
                ▼
┌─────────────────────────────────────────┐
│ LAYER 3: LLM DIAGNOSTIC AGENT           │
│ • Generates plain-language advisory     │
│   structured as WHAT / WHY / WHERE      │
│ • Cites specific sensor values vs       │
│   baseline (not just "anomaly detected")│
│ • States consequences if unresolved     │
│ • Recommends specific inspection steps  │
│ • Runs on-device — no cloud required    │
└─────────────────────────────────────────┘
                │
                ▼
     SAIREN Dashboard Advisory
     (HLO / OIM / Mechanic screen)
```

---

## 5. What SAIREN Does NOT Do

- It does not actuate valves, start pumps, or trigger releases.
- It does not replace the certified fire panel, F&G system, or any SIL-rated safety function.
- It does not require any modification to existing safety systems.
- All sensors are passive monitoring instruments only — wired to the SAIREN edge node, completely isolated from control circuits.

---

## 6. Sensor Summary — Installation Reference

| Subsystem | No. of Sensors | Sensor Types | Typical Installation Method |
|---|---|---|---|
| Surge tank / deepwell | 3 | Ultrasonic level, hydrostatic pressure, CT | Tank boss fitting (level), pipe tapping (pressure), MCC clamp (CT) |
| Jockey pump | 3 | Pressure transmitter, digital input, CT | Pipe tapping, MCC aux contact, MCC clamp |
| Main fire pumps (x2) | 8 (4 each) | Pressure transmitters x2, CT x3 phases, digital run status | Pipe tappings, MCC enclosure clamp, MCC aux contact |
| Foam pumps + proportioner (x2) | 10 | Pressure transmitters x4, level, temperature, CT | Pipe tappings, tank boss, tank wall |
| Ring main + branches | 4–5 | Pressure transmitters, valve position switches, flow (optional) | Pipe tappings, valve actuator terminal |
| **Total (minimum viable)** | **~28–29 sensors** | 4–20 mA analogue + digital inputs + CTs | Non-intrusive — no cutting of firewater pipe required |

All sensors are read-only, passive instruments. No modification to existing pipework beyond installation of pressure tapping bosses (standard half-inch NPT or BSP), which can be hot-tapped or installed during the next scheduled maintenance isolation.

---

*SAIREN Ltd | Edge-native industrial intelligence | All data processed and stored on-site — no cloud dependency*
