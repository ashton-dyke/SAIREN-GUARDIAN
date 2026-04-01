# SAIREN Fire Main Continuous Integrity Monitoring: Pilot Specification

## 1. Executive Summary (For the OIM & Chief Mechanic)

**The Problem:** The rig's primary firefighting system (water/foam ring main) must maintain pressure integrity 24/7. However, faults like slow leaks, partially stuck valves, or internal pipe corrosion often go undetected until a mandatory flow test—or worse, an actual incident. Conventional intrusive sensors are difficult to retrofit and introduce new leak paths.

**The Solution:** This pilot deploys SAIREN's **non-invasive Acoustic Resonance Spectroscopy (ARS)** technology. By clamping industrial accelerometers to the outside of the pipe, we can passively monitor the "acoustic signature" of the fire main. Combined with our Physics-Informed AI on the edge, this allows us to continuously infer internal pressure, detect slow leaks, spot stuck isolation valves, and monitor long-term wall thinning—without drilling a single hole in the pressure boundary.

**The Benefit:** Continuous, automated evidence of fire main integrity for the safety case, early warning of leaks before they require emergency maintenance, and reduced reliance on manual pressure checks.

---

## 2. Sensor Specification & Layout

This pilot covers the primary water/foam ring main, driven by two main fire pumps and one jockey pump.

### 2.1 Primary Sensing (Non-Invasive)
- **Sensor Type:** Industrial IEPE Piezoelectric Accelerometers
- **Certification:** ATEX / IECEx certified for relevant hazardous zones (e.g., Ex ia IIC T4)
- **Frequency Range:** 1 Hz to 10 kHz (minimum)
- **Mounting:** High-strength structural epoxy or welded stud on cleaned pipe wall (no penetration)
- **Quantity:** 6 to 8 sensors

### 2.2 Ground-Truth Instrumentation (Existing)
- **Primary:** Existing Fire Main Pressure Transmitter on the main pump discharge header.
- **Secondary:** Jockey pump run-state signals (if available via PLC/dry contact).
- **Purpose:** These existing signals provide the "ground truth" labels that automatically calibrate the AI model during normal operations and weekly tests.

### 2.3 Recommended Sensor Placement map
| Sensor ID | Location | Purpose |
| :--- | :--- | :--- |
| **A1** | Common discharge header (downstream of main pump check valves) | Master acoustic profile during main pump runs; surge detection. |
| **A2** | Jockey pump discharge tie-in | Detects jockey pump cycling and baseline leak make-up signatures. |
| **A3** | Main header exit (leaving pump room) | Baseline integrity before the ring splits. |
| **A4** | Ring main segment (Accommodation block) | Localised pressure/flow tracking; remote zone integrity. |
| **A5** | Ring main segment (Machinery space / Engine room) | Localised pressure/flow tracking; high-vibration noise filtering. |
| **A6** | Ring main segment (Drill floor / Open deck) | Localised pressure tracking; detects stuck local isolation valves. |
| **A7 (Opt)** | Downstream of foam proportioner | Detects acoustic shift when fluid changes from pure water to foam mix. |

---

## 3. Edge Compute & Data Acquisition

The sensors will wire back to a local SAIREN Edge Node located in a safe area (or certified Ex d enclosure).

### 3.1 Hardware
- **Edge Unit:** SAIREN Raspberry Pi 5 core.
- **Data Acquisition (DAQ):** 8-channel IEPE signal conditioner HAT/USB interface (minimum 24-bit resolution, capable of 10-20 kHz sampling per channel).
- **Integration:** 1x isolated 4-20mA or Modbus input to tap the existing pump discharge pressure transmitter.

### 3.2 Processing Pipeline
- **Sampling:** High-frequency burst sampling (e.g., 2-second windows at 10 kHz) triggered periodically and during pump starts.
- **Edge AI:** A Physics-Informed Spiking Neural Network (PI-SNN) running locally. It uses the acoustic wave equation for fluid-filled steel pipes to ensure all predictions obey the laws of physics.
- **Self-Calibration:** The system cross-references the acoustic spectra against the existing discharge pressure gauge, continuously refining its accuracy specific to this rig's pipe geometry.

---

## 4. Deliverables & Outputs

This system does not replace existing Fire & Gas (F&G) executive actions. It acts as an independent advisory and predictive maintenance overlay.

### 4.1 Real-Time Dashboard (SAIREN UI)
- **Virtual Pressure Map:** Estimated pressure trends across all sensored zones (A1 through A6), highlighting localized pressure drops that a single central gauge would miss.
- **System Health Score:** A 0-100 index of fire main integrity.

### 4.2 Automated Alarms & Advisories
- **Slow Leak Warning:** Detected via anomalous jockey pump signatures and shifting acoustic resonance during static periods.
- **Valve Anomaly Alert:** Flags if a specific ring main segment (e.g., A5) fails to show the correct acoustic response during a test flow, indicating a partially stuck or incorrectly closed isolation valve.
- **Corrosion Trend (Long-Term):** Tracks broadband resonance shifts over months, indicating potential pipe wall thinning in specific segments.

### 4.3 Regulatory Output
- **Continuous Integrity Log:** Automated digital log proving the fire main maintained required standby pressure and responded correctly during weekly drills, exportable for OIM / safety case audits.
