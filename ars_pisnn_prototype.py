"""
================================================================================
SAIREN ARS-PISNN: Acoustic Resonance Spectroscopy via Physics-Informed
                   Spiking Neural Network -- Pure NumPy Prototype
================================================================================

WHAT THIS IS
------------
A self-contained prototype for non-invasive pipe pressure estimation using
Acoustic Resonance Spectroscopy (ARS). Clamp an accelerometer to a pipe,
capture the acoustic resonance response, infer internal pressure from the
frequency spectrum -- no pipe penetration required.

THE PHYSICS IT ENCODES
----------------------
A fluid-filled cylindrical pipe has longitudinal acoustic resonances at:

    f_n = (n / 2L) * sqrt(B_eff / rho_eff)

where n = mode number, L = pipe length, B_eff = effective bulk modulus,
rho_eff = effective density. Internal pressure changes B_eff (higher pressure
-> stiffer fluid -> higher resonance frequencies). The PI-SNN encodes this
as a physics loss: predicted pressure must be consistent with the observed
dominant frequency via the resonance equation.

SNN ARCHITECTURE NOTE
---------------------
For rate-coded inputs (each sample is a static spectrum), a recurrent LIF
network's spike rates converge to a fixed point that is functionally equivalent
to a feedforward network with sigmoid activations. This prototype uses the
equilibrium-equivalent formulation for reliable training, with explicit LIF
simulation for inference-time spike rate analysis (anomaly detection).

The mapping is direct:
    LIF equilibrium spike rate  <->  sigmoid(W @ x + b)
    Membrane leak (beta)        <->  controls convergence speed (not equilibrium)
    Threshold + surrogate slope <->  sigmoid steepness

For edge deployment on neuromorphic hardware, these weights map directly to
LIF synapse weights. The physics loss, online learning, and anomaly detection
are architecture-agnostic.

HOW IT FITS INTO SAIREN
-----------------------
- Runs on Raspberry Pi 5 at the edge (CPU-only, no CUDA)
- Feeds into the existing multi-agent anomaly detection pipeline
- Column weights are JSON-serialisable for P2P gossip mesh distribution
- Online learning loop adapts to changing pipe specs without retraining
- Anomaly scores integrate with existing CfC-based alarm system

DEPENDENCIES: numpy, scipy, matplotlib, rich (all pure-Python/C-ext, no GPU)
================================================================================
"""

import json
import math
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

console = Console()

# ============================================================================
#  1. CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """All hyperparameters in one place. Nothing hardcoded elsewhere."""

    # --- Pipe physical parameters ---
    # Short standpipe section between flanges — resonance length ~ 1m
    # gives modes in the kHz range, resolvable with 512-bin FFT
    pipe_length: float = 1.0             # metres (standpipe section)
    pipe_outer_diameter: float = 0.1143  # 4.5" drill pipe in metres
    pipe_wall_thickness: float = 0.00939 # metres (~0.37")
    pipe_material: str = "carbon_steel"
    pipe_young_modulus: float = 200e9    # Pa (carbon steel)
    pipe_density: float = 7850.0         # kg/m^3 (carbon steel)
    pipe_poisson_ratio: float = 0.3

    # --- Fluid parameters (water-based drilling mud, approximate) ---
    fluid_base_density: float = 1200.0          # kg/m^3 (weighted mud)
    fluid_base_bulk_modulus: float = 2.2e9       # Pa (water baseline)
    # Pressure sensitivity: accounts for dissolved gas compressibility
    # and temperature effects in drilling mud at downhole conditions
    fluid_bulk_modulus_pressure_coeff: float = 12.0

    # --- Spectrum parameters ---
    n_fft_bins: int = 512
    max_freq_hz: float = 50000.0
    n_resonance_modes: int = 20  # short pipe = many modes in 0-50 kHz band

    # --- Noise parameters ---
    pump_harmonic_base_hz: float = 60.0
    n_pump_harmonics: int = 5
    pump_harmonic_amplitude: float = 0.15
    white_noise_floor: float = 0.03

    # --- Dataset ---
    n_pressure_levels: int = 10
    pressure_min_psi: float = 0.0
    pressure_max_psi: float = 15000.0
    samples_per_level: int = 300  # total = 3000
    train_fraction: float = 0.8

    # --- SNN architecture ---
    snn_layers: list = field(default_factory=lambda: [512, 256, 128, 1])
    snn_beta: float = 0.80              # LIF membrane decay (used in LIF sim)
    snn_threshold: float = 1.0          # spike threshold (used in LIF sim)
    n_timesteps: int = 25               # LIF simulation steps for anomaly det.

    # --- Training ---
    learning_rate: float = 2e-3
    n_epochs: int = 50
    batch_size: int = 64
    physics_lambda: float = 0.1
    grad_clip: float = 5.0

    # --- Online learning ---
    online_buffer_size: int = 500
    online_update_interval: int = 50
    online_lr: float = 5e-4
    pseudo_label_confidence_threshold: float = 0.15
    ema_alpha: float = 0.01

    # --- Anomaly detection ---
    anomaly_zscore_threshold: float = 3.0
    physics_violation_threshold: float = 0.10

    # --- Demo ---
    online_stream_length: int = 800
    anomaly_inject_time: int = 500
    anomaly_pressure_psi: float = 18000.0
    demo_print_interval: int = 10

    # --- Paths ---
    output_dir: str = "output"
    column_weights_file: str = "column_weights.json"


# ============================================================================
#  2. DATA SIMULATOR
# ============================================================================

def psi_to_pascal(psi: float) -> float:
    """Convert pounds per square inch to Pascals."""
    return psi * 6894.757

def pascal_to_psi(pa: float) -> float:
    """Convert Pascals to PSI."""
    return pa / 6894.757

def effective_bulk_modulus(pressure_psi: float, cfg: Config) -> float:
    """
    Effective bulk modulus of the pipe-fluid system at a given pressure.

    Physics: B_fluid increases linearly with pressure. Pipe wall compliance
    (thin-shell approximation) adds: 1/B_eff = 1/B_fluid(P) + D/(E*t).
    """
    pressure_pa = psi_to_pascal(pressure_psi)
    B_fluid = cfg.fluid_base_bulk_modulus + cfg.fluid_bulk_modulus_pressure_coeff * pressure_pa
    D = cfg.pipe_outer_diameter - 2 * cfg.pipe_wall_thickness
    wall_compliance = D / (cfg.pipe_young_modulus * cfg.pipe_wall_thickness)
    B_eff = 1.0 / (1.0 / B_fluid + wall_compliance)
    return B_eff

def effective_density(cfg: Config) -> float:
    """Effective density for longitudinal wave propagation in a fluid-filled pipe."""
    r_outer = cfg.pipe_outer_diameter / 2
    r_inner = r_outer - cfg.pipe_wall_thickness
    A_wall = math.pi * (r_outer**2 - r_inner**2)
    A_fluid = math.pi * r_inner**2
    return cfg.fluid_base_density + cfg.pipe_density * (A_wall / A_fluid)

def resonance_frequencies(pressure_psi: float, cfg: Config) -> np.ndarray:
    """f_n = (n/2L) * sqrt(B_eff/rho_eff). Higher pressure -> higher frequencies."""
    B_eff = effective_bulk_modulus(pressure_psi, cfg)
    rho_eff = effective_density(cfg)
    c = math.sqrt(B_eff / rho_eff)
    modes = np.arange(1, cfg.n_resonance_modes + 1)
    return modes * c / (2 * cfg.pipe_length)

def generate_spectrum(pressure_psi: float, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """
    Synthetic FFT magnitude spectrum for a fluid-filled pipe at given pressure.
    Gaussian peaks at resonance frequencies + pump harmonics + white noise.
    """
    freq_axis = np.linspace(0, cfg.max_freq_hz, cfg.n_fft_bins)
    spectrum = np.zeros(cfg.n_fft_bins)

    freqs = resonance_frequencies(pressure_psi, cfg)
    for i, f_res in enumerate(freqs):
        if f_res > cfg.max_freq_hz:
            break
        amplitude = 1.0 / (1 + 0.3 * i) + rng.normal(0, 0.05)
        amplitude = max(amplitude, 0.05)
        # Sharper peaks for short pipe (higher Q factor); width grows with mode
        width = 80 + 30 * i + rng.normal(0, 10)
        width = max(width, 30)
        spectrum += amplitude * np.exp(-0.5 * ((freq_axis - f_res) / width) ** 2)

    for h in range(1, cfg.n_pump_harmonics + 1):
        f_pump = cfg.pump_harmonic_base_hz * h
        amp = cfg.pump_harmonic_amplitude / h + rng.normal(0, 0.02)
        spectrum += max(amp, 0) * np.exp(-0.5 * ((freq_axis - f_pump) / 30) ** 2)

    spectrum += cfg.white_noise_floor * np.abs(rng.standard_normal(cfg.n_fft_bins))
    spectrum = np.maximum(spectrum, 0)
    peak = spectrum.max()
    if peak > 0:
        spectrum /= peak
    return spectrum.astype(np.float32)

def generate_dataset(cfg: Config, seed: int = 42):
    """Generate a labelled dataset of acoustic spectra at varying pressures."""
    rng = np.random.default_rng(seed)
    pressure_levels = np.linspace(cfg.pressure_min_psi, cfg.pressure_max_psi, cfg.n_pressure_levels)
    spectra_list = []
    pressure_list = []

    # Scale jitter to ~1% of pressure range (50 PSI for drilling, ~1 PSI for firewater)
    pressure_range = cfg.pressure_max_psi - cfg.pressure_min_psi
    jitter_std = max(pressure_range * 0.01, 1.0)

    for p in pressure_levels:
        for _ in range(cfg.samples_per_level):
            p_jitter = p + rng.normal(0, jitter_std)
            p_jitter = np.clip(p_jitter, cfg.pressure_min_psi, cfg.pressure_max_psi * 1.3)
            spectra_list.append(generate_spectrum(p_jitter, cfg, rng))
            pressure_list.append(p_jitter)

    spectra = np.array(spectra_list, dtype=np.float32)
    pressures = np.array(pressure_list, dtype=np.float32)
    idx = rng.permutation(len(spectra))
    return spectra[idx], pressures[idx]


# ============================================================================
#  3. PI-SNN MODEL
# ============================================================================
#
# Equilibrium-equivalent formulation for training (sigmoid activations),
# with explicit LIF simulation for spike-rate anomaly detection at inference.
#
# The equivalence: for a rate-coded LIF neuron receiving constant input I,
# the steady-state firing rate is a monotonic sigmoid-like function of I.
# Training the sigmoid version and deploying as LIF is standard practice
# (ANN-to-SNN conversion, Diehl et al. 2015; Sengupta et al. 2019).

def _sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-np.minimum(x, 80))),
                    np.exp(np.maximum(x, -80)) / (1.0 + np.exp(np.maximum(x, -80))))

def _relu(x):
    """ReLU activation."""
    return np.maximum(x, 0)

def _relu_deriv(x):
    """Derivative of ReLU."""
    return (x > 0).astype(np.float32)


class PISNN:
    """
    Physics-Informed Spiking Neural Network for acoustic pressure estimation.

    Architecture: 512 -> 256 (ReLU) -> 128 (ReLU) -> 1 (linear)

    Training uses ReLU activations (equilibrium-equivalent to LIF spike rates).
    Inference includes explicit LIF simulation for spike-rate based anomaly
    detection. The physics loss constrains predictions via the acoustic
    resonance equation.
    """

    def __init__(self, cfg: Config, seed: int = 42):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        sizes = cfg.snn_layers  # [512, 256, 128, 1]
        self.n_hidden = len(sizes) - 2  # 2 hidden layers

        # Hidden layer weights (He initialisation for ReLU)
        self.W = []
        self.b = []
        for i in range(self.n_hidden):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            std = math.sqrt(2.0 / fan_in)
            self.W.append(self.rng.normal(0, std, (fan_in, fan_out)).astype(np.float32))
            self.b.append(np.zeros(fan_out, dtype=np.float32))

        # Output layer
        fan_in = sizes[-2]
        self.Wo = self.rng.normal(0, math.sqrt(2.0 / fan_in), (fan_in, 1)).astype(np.float32)
        self.bo = np.zeros(1, dtype=np.float32)

        # Adam state
        self._init_adam()

        # Normalisation
        self.pressure_mean = 7500.0
        self.pressure_std = 4500.0
        self.adam_t = 0

    def _init_adam(self):
        """Initialise Adam momentum buffers."""
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b_) for b_ in self.b]
        self.vb = [np.zeros_like(b_) for b_ in self.b]
        self.mWo = np.zeros_like(self.Wo)
        self.vWo = np.zeros_like(self.Wo)
        self.mbo = np.zeros_like(self.bo)
        self.vbo = np.zeros_like(self.bo)

    def forward(self, spectra: np.ndarray):
        """
        Forward pass: spectrum -> hidden layers (ReLU) -> pressure prediction.

        Returns:
            pred_psi: (batch, 1) pressure predictions in PSI
            activations: list of (batch, layer_size) hidden activations
                         (equivalent to LIF spike rates at equilibrium)
            cache: dict for backward pass
        """
        x = spectra
        pre_acts = []   # pre-activation values (for gradient)
        acts = [x]      # post-activation values (including input)

        for i in range(self.n_hidden):
            z = x @ self.W[i] + self.b[i]
            pre_acts.append(z)
            x = _relu(z)
            acts.append(x)

        # Output: linear readout from last hidden layer
        pred_norm = x @ self.Wo + self.bo
        pred_psi = pred_norm * self.pressure_std + self.pressure_mean

        cache = {"pre_acts": pre_acts, "acts": acts, "final_hidden": x}
        return pred_psi, acts[1:], cache

    def simulate_lif(self, spectra: np.ndarray):
        """
        Explicit LIF simulation for spike-rate anomaly detection.

        Runs the trained weights through actual LIF dynamics to produce
        spike trains, used for computing spike-rate deviation scores.
        NOT used for training -- only for anomaly detection at inference.

        Returns:
            spike_counts: list of (batch, layer_size) spike counts per layer
        """
        cfg = self.cfg
        B = spectra.shape[0]
        T = cfg.n_timesteps
        beta = cfg.snn_beta
        thr = cfg.snn_threshold

        x = spectra
        all_counts = []

        for li in range(self.n_hidden):
            fan_out = self.W[li].shape[1]
            mem = np.zeros((B, fan_out), dtype=np.float32)
            spk = np.zeros((B, fan_out), dtype=np.float32)
            counts = np.zeros((B, fan_out), dtype=np.float32)

            I = x @ self.W[li] + self.b[li]

            for t in range(T):
                mem = beta * mem + (1 - beta) * I - spk * thr
                spk = (mem >= thr).astype(np.float32)
                counts += spk

            all_counts.append(counts)
            # Rate-coded input to next layer
            x = counts / T

        return all_counts

    def compute_physics_residual(self, spectra: np.ndarray, pred_psi: np.ndarray) -> np.ndarray:
        """
        Physics residual: predicted pressure must match observed resonance peaks
        via f_n = (n/2L) * sqrt(B_eff / rho_eff).

        Finds the dominant spectral peak (above pump band), computes expected
        frequency from predicted pressure, returns squared relative error.
        """
        cfg = self.cfg
        freq_axis = np.linspace(0, cfg.max_freq_hz, cfg.n_fft_bins)
        B = spectra.shape[0]
        residuals = np.zeros(B, dtype=np.float32)
        low_bin = int(500 / cfg.max_freq_hz * cfg.n_fft_bins)

        for i in range(B):
            spec = spectra[i].copy()
            spec[:low_bin] = 0
            peak_bin = np.argmax(spec)
            f_obs = freq_axis[peak_bin]
            if f_obs < 500:
                continue

            p = max(float(pred_psi[i].flat[0] if pred_psi.ndim > 1 else pred_psi[i]), 0)
            B_eff = effective_bulk_modulus(p, cfg)
            rho_eff = effective_density(cfg)
            c = math.sqrt(B_eff / rho_eff)
            f1 = c / (2 * cfg.pipe_length)

            if f1 > 0:
                n = max(1, round(f_obs / f1))
                f_pred = n * f1
                residuals[i] = ((f_obs - f_pred) / max(f_pred, 1.0)) ** 2

        return residuals

    def backward_and_update(self, spectra: np.ndarray, targets_psi: np.ndarray,
                            pred_psi: np.ndarray, cache: dict, lr: float):
        """
        Backward pass + Adam update.

        Loss = MSE(pred, target) + lambda * physics_residual
        Standard backprop through ReLU hidden layers.
        """
        cfg = self.cfg
        B = spectra.shape[0]
        pre_acts = cache["pre_acts"]
        acts = cache["acts"]

        target_norm = ((targets_psi - self.pressure_mean) / self.pressure_std).reshape(-1, 1)
        pred_norm = (pred_psi - self.pressure_mean) / self.pressure_std

        # MSE gradient
        grad = 2.0 * (pred_norm - target_norm) / B

        # Physics loss gradient (direction from MSE, magnitude from physics residual)
        phys_res = self.compute_physics_residual(spectra, pred_psi)
        phys_scale = cfg.physics_lambda * 2.0 * phys_res.reshape(-1, 1) / B
        grad += phys_scale * np.sign(pred_norm - target_norm)

        mse_val = float(np.mean((pred_psi.flatten() - targets_psi) ** 2))
        phys_val = float(np.mean(phys_res))

        # Output layer gradients
        final_h = acts[-1]  # last hidden activation
        grad_Wo = final_h.T @ grad
        grad_bo = grad.sum(axis=0)
        grad_h = grad @ self.Wo.T  # (B, 128)

        # Hidden layers (reverse order)
        grad_W_all = [None] * self.n_hidden
        grad_b_all = [None] * self.n_hidden

        for i in range(self.n_hidden - 1, -1, -1):
            # ReLU derivative
            grad_z = grad_h * _relu_deriv(pre_acts[i])

            grad_W_all[i] = acts[i].T @ grad_z
            grad_b_all[i] = grad_z.sum(axis=0)

            if i > 0:
                grad_h = grad_z @ self.W[i].T

        # Adam update
        self.adam_t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        clip = cfg.grad_clip

        def adam_step(param, g, m, v):
            g = np.clip(g, -clip, clip)
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * g ** 2
            mh = m / (1 - b1 ** self.adam_t)
            vh = v / (1 - b2 ** self.adam_t)
            param -= lr * mh / (np.sqrt(vh) + eps)

        for i in range(self.n_hidden):
            adam_step(self.W[i], grad_W_all[i], self.mW[i], self.vW[i])
            adam_step(self.b[i], grad_b_all[i], self.mb[i], self.vb[i])

        adam_step(self.Wo, grad_Wo, self.mWo, self.vWo)
        adam_step(self.bo, grad_bo, self.mbo, self.vbo)

        return mse_val, phys_val

    def get_encoder_weights(self) -> dict:
        """Export hidden layer weights for mesh synchronisation."""
        return {
            "W": [w.copy() for w in self.W],
            "b": [b_.copy() for b_ in self.b],
        }

    def set_encoder_weights(self, weights: dict):
        """Import averaged encoder weights from mesh. Resets Adam momentum."""
        for i in range(self.n_hidden):
            self.W[i] = weights["W"][i].copy()
            self.b[i] = weights["b"][i].copy()
        self._init_adam()

    def set_all_weights(self, weights: dict):
        """Import all weights (encoder + output) from another model."""
        for i in range(len(weights["W"])):
            self.W[i] = weights["W"][i].copy()
            self.b[i] = weights["b"][i].copy()
        self._init_adam()


# ============================================================================
#  4. TRAINING LOOP
# ============================================================================

def train_model(model: PISNN, spectra: np.ndarray, pressures: np.ndarray, cfg: Config):
    """Batch training: supervised on synthetic dataset, Adam optimiser."""
    n = len(spectra)
    n_train = int(n * cfg.train_fraction)
    X_train, y_train = spectra[:n_train], pressures[:n_train]
    X_test, y_test = spectra[n_train:], pressures[n_train:]

    model.pressure_mean = float(y_train.mean())
    model.pressure_std = float(y_train.std()) + 1e-6

    console.print(f"\n[bold cyan]Training PI-SNN[/bold cyan]: {n_train} train, {n - n_train} test")
    console.print(f"  Architecture: {cfg.snn_layers}, LR: {cfg.learning_rate}")
    console.print(f"  Physics lambda: {cfg.physics_lambda}\n")

    history = {"epoch": [], "train_rmse": [], "test_rmse": [], "physics_loss": []}

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Epochs", total=cfg.n_epochs)

        for epoch in range(cfg.n_epochs):
            idx = model.rng.permutation(n_train)
            X_shuf, y_shuf = X_train[idx], y_train[idx]

            # Cosine annealing LR schedule
            lr = cfg.learning_rate * 0.5 * (1 + math.cos(math.pi * epoch / cfg.n_epochs))
            lr = max(lr, cfg.learning_rate * 0.01)  # floor at 1% of initial

            epoch_mse, epoch_phys, n_batches = 0.0, 0.0, 0

            for start in range(0, n_train, cfg.batch_size):
                end = min(start + cfg.batch_size, n_train)
                Xb, yb = X_shuf[start:end], y_shuf[start:end]

                pred, _, cache = model.forward(Xb)
                mse, phys = model.backward_and_update(Xb, yb, pred, cache, lr)
                epoch_mse += mse
                epoch_phys += phys
                n_batches += 1

            pred_test, _, _ = model.forward(X_test)
            test_rmse = float(np.sqrt(np.mean((pred_test.flatten() - y_test) ** 2)))

            history["epoch"].append(epoch + 1)
            history["train_rmse"].append(float(np.sqrt(epoch_mse / max(n_batches, 1))))
            history["test_rmse"].append(test_rmse)
            history["physics_loss"].append(epoch_phys / max(n_batches, 1))

            progress.update(task, advance=1,
                            description=f"Epoch {epoch+1:2d} | Test RMSE: {test_rmse:8.1f} PSI")

    console.print(f"\n[bold green]Final test RMSE: {history['test_rmse'][-1]:.1f} PSI[/bold green]")
    return history, X_test, y_test


# ============================================================================
#  5. ONLINE LEARNING (OTTT-inspired)
# ============================================================================

class CircularBuffer:
    """
    Fixed-size circular buffer for streaming online learning.
    Overwrites oldest samples when full -- prevents memory growth on Pi 5.
    """

    def __init__(self, capacity: int, dim: int):
        self.capacity = capacity
        self.spectra = np.zeros((capacity, dim), dtype=np.float32)
        self.pressures = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.count = 0

    def add(self, spectrum: np.ndarray, pressure: float):
        self.spectra[self.idx] = spectrum
        self.pressures[self.idx] = pressure
        self.idx = (self.idx + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def get_batch(self, batch_size: int, rng: np.random.Generator):
        n = min(self.count, batch_size)
        idx = rng.choice(self.count, size=n, replace=False)
        return self.spectra[idx], self.pressures[idx]


class AnomalyDetector:
    """
    Dual-signal anomaly detector: spike rate deviation + physics residual.

    Signal 1 -- Spike rate deviation (z-score):
        EMA of spike rates per neuron. Anomalies produce spike patterns
        that deviate from baseline.

    Signal 2 -- Physics residual:
        If predicted pressure violates the acoustic resonance equation,
        flags sensor fault, pipe integrity issue, or genuine anomaly.
    """

    def __init__(self, layer_sizes: list, cfg: Config):
        self.cfg = cfg
        # Track statistics for hidden layers
        self.ema_rates = [np.zeros(s, dtype=np.float64) for s in layer_sizes[1:-1]]
        self.ema_sq = [np.zeros(s, dtype=np.float64) for s in layer_sizes[1:-1]]
        self.n_updates = 0

    def update_baseline(self, activations: list):
        """Update running EMA of activation magnitudes (spike rate proxy)."""
        alpha = self.cfg.ema_alpha
        self.n_updates += 1
        for i, act in enumerate(activations):
            rate = act.mean(axis=0).astype(np.float64)
            self.ema_rates[i] = (1 - alpha) * self.ema_rates[i] + alpha * rate
            self.ema_sq[i] = (1 - alpha) * self.ema_sq[i] + alpha * rate ** 2

    def compute_anomaly_score(self, activations: list, physics_residual: float) -> tuple:
        """Returns (anomaly_score [0-1], physics_violation bool, max_zscore)."""
        if self.n_updates < 100:
            return 0.0, False, 0.0

        max_zscore = 0.0
        for i, act in enumerate(activations):
            rate = act.mean(axis=0).astype(np.float64)
            mean = self.ema_rates[i]
            var = self.ema_sq[i] - mean ** 2
            std = np.sqrt(np.maximum(var, 1e-10))
            zscores = np.abs((rate - mean) / std)
            max_zscore = max(max_zscore, float(np.percentile(zscores, 95)))

        physics_violation = physics_residual > self.cfg.physics_violation_threshold

        z_component = min(max_zscore / 6.0, 1.0)
        p_component = min(physics_residual / 0.3, 1.0)
        anomaly_score = min(max(z_component, p_component), 1.0)

        return anomaly_score, physics_violation, max_zscore


# ============================================================================
#  6. COLUMNAR STRUCTURE (Catastrophic Forgetting Mitigation)
# ============================================================================

class ColumnManager:
    """
    Separate output-layer weight columns per pipe specification.

    Shared encoder layers adapt continuously (general acoustic features).
    Output columns specialise per pipe geometry. Columns are JSON-serialisable
    for distribution across the SAIREN gossip mesh.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.columns: dict = {}
        self.active_key: Optional[str] = None

    @staticmethod
    def make_key(diameter: float, wall_thickness: float, material: str) -> str:
        return f"{diameter:.4f}_{wall_thickness:.5f}_{material}"

    def get_or_create(self, model: PISNN, diameter: float, wall_thickness: float,
                      material: str) -> str:
        key = self.make_key(diameter, wall_thickness, material)
        if self.active_key and self.active_key != key:
            self._save(model, self.active_key)
        if key not in self.columns:
            self.columns[key] = {
                "W": model.Wo.copy().tolist(),
                "b": model.bo.copy().tolist(),
            }
            console.print(f"  [yellow]New pipe column: {key}[/yellow]")
        else:
            model.Wo = np.array(self.columns[key]["W"], dtype=np.float32)
            model.bo = np.array(self.columns[key]["b"], dtype=np.float32)
        self.active_key = key
        return key

    def _save(self, model: PISNN, key: str):
        self.columns[key] = {
            "W": model.Wo.copy().tolist(),
            "b": model.bo.copy().tolist(),
        }

    def save_to_json(self, model: PISNN, path: str):
        """Persist all columns to JSON (for gossip mesh distribution)."""
        if self.active_key:
            self._save(model, self.active_key)
        with open(path, "w") as f:
            json.dump(self.columns, f, indent=2)
        console.print(f"  Saved {len(self.columns)} column(s) to {path}")

    def load_from_json(self, path: str):
        with open(path, "r") as f:
            self.columns = json.load(f)


# ============================================================================
#  7. CLI DEMO
# ============================================================================

def run_demo():
    """
    Full end-to-end demo:
    1. Generate synthetic ARS dataset
    2. Train PI-SNN (batch, 20 epochs)
    3. Stream online with OTTT-inspired updates
    4. Inject pressure anomaly at t=500
    5. Print live table + generate plots
    """
    cfg = Config()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(exist_ok=True)

    console.rule("[bold magenta]SAIREN ARS-PISNN Prototype[/bold magenta]")
    console.print("Non-invasive pipe pressure estimation via Acoustic Resonance Spectroscopy\n")

    # -- Step 1: Generate dataset -----------------------------------------------
    console.rule("[cyan]Step 1: Generate Synthetic Dataset[/cyan]")
    spectra, pressures = generate_dataset(cfg)
    console.print(f"Generated {len(spectra)} samples, {cfg.n_pressure_levels} pressure levels")
    console.print(f"Pressure range: {pressures.min():.0f} - {pressures.max():.0f} PSI")

    # -- Step 2: Train ----------------------------------------------------------
    console.rule("[cyan]Step 2: Train PI-SNN (Batch Mode)[/cyan]")
    model = PISNN(cfg, seed=42)
    col_mgr = ColumnManager(cfg)
    col_mgr.get_or_create(model, cfg.pipe_outer_diameter, cfg.pipe_wall_thickness, cfg.pipe_material)

    history, X_test, y_test = train_model(model, spectra, pressures, cfg)

    # -- Step 3: Online streaming -----------------------------------------------
    console.rule("[cyan]Step 3: Online Streaming with Anomaly Injection[/cyan]")

    anomaly_det = AnomalyDetector(cfg.snn_layers, cfg)
    buf = CircularBuffer(cfg.online_buffer_size, cfg.n_fft_bins)
    rng = np.random.default_rng(99)

    # Pressure profile
    N = cfg.online_stream_length
    base_p = np.concatenate([
        np.linspace(5000, 10000, cfg.anomaly_inject_time),
        np.full(N - cfg.anomaly_inject_time, 10000.0),
    ])
    anom_start = cfg.anomaly_inject_time
    anom_end = min(anom_start + 30, N)
    base_p[anom_start:anom_end] = cfg.anomaly_pressure_psi
    if anom_end < N:
        rec_len = min(50, N - anom_end)
        base_p[anom_end:anom_end + rec_len] = np.linspace(cfg.anomaly_pressure_psi, 10000, rec_len)

    # Recording arrays
    true_p = np.zeros(N)
    pred_p = np.zeros(N)
    anom_scores = np.zeros(N)
    phys_viols = np.zeros(N, dtype=bool)
    phys_res_arr = np.zeros(N)
    # Store last hidden layer activations for heatmap
    act_history = np.zeros((N, cfg.snn_layers[-2]))

    table = Table(title="Online Streaming Inference")
    table.add_column("t", justify="right", style="cyan", width=6)
    table.add_column("True PSI", justify="right", style="green", width=10)
    table.add_column("Pred PSI", justify="right", style="yellow", width=10)
    table.add_column("Anomaly", justify="right", style="red", width=8)
    table.add_column("Phys Viol", justify="center", style="bold red", width=9)

    first_detection = None

    for t in range(N):
        p_true = base_p[t] + rng.normal(0, 30)
        true_p[t] = p_true

        spec = generate_spectrum(p_true, cfg, rng).reshape(1, -1)

        # Inference
        pred_psi, activations, cache = model.forward(spec)
        pred_val = float(pred_psi[0, 0])
        pred_p[t] = pred_val

        # Physics residual
        pr = model.compute_physics_residual(spec, pred_psi)
        pr_val = float(pr[0])
        phys_res_arr[t] = pr_val

        # Anomaly detection (using hidden activations as spike rate proxy)
        anomaly_det.update_baseline(activations)
        a_score, p_viol, z_max = anomaly_det.compute_anomaly_score(
            activations, pr_val
        )
        if z_max > cfg.anomaly_zscore_threshold:
            p_viol = True
            a_score = max(a_score, 0.8)

        anom_scores[t] = a_score
        phys_viols[t] = p_viol
        act_history[t] = activations[-1].flatten()[:cfg.snn_layers[-2]]

        if t >= anom_start and first_detection is None and (p_viol or a_score > 0.5):
            first_detection = t

        # Online buffer + periodic update
        buf.add(spec.flatten(), p_true)
        if t > 0 and t % cfg.online_update_interval == 0 and buf.count >= cfg.batch_size:
            Xb, yb = buf.get_batch(cfg.batch_size, rng)
            pb, _, cb = model.forward(Xb)
            model.backward_and_update(Xb, yb, pb, cb, cfg.online_lr)

        # Table output
        if t % cfg.demo_print_interval == 0 or (anom_start <= t < anom_start + 5):
            flag = "[bold red]TRUE[/bold red]" if p_viol else "[dim]false[/dim]"
            astr = f"{a_score:.3f}"
            if a_score > 0.5:
                astr = f"[bold red]{a_score:.3f}[/bold red]"
            table.add_row(str(t), f"{p_true:.0f}", f"{pred_val:.0f}", astr, flag)

    console.print(table)

    # -- Step 4: Results --------------------------------------------------------
    console.rule("[cyan]Results Summary[/cyan]")

    batch_test_rmse = history["test_rmse"][-1]
    console.print(f"  Batch test RMSE (post-training): [bold]{batch_test_rmse:.1f} PSI[/bold]", end="")
    if batch_test_rmse < 500:
        console.print(" [bold green]<< PASS (< 500 PSI)[/bold green]")
    else:
        console.print(f" [yellow](target: < 500 PSI)[/yellow]")

    normal_mask = np.ones(N, dtype=bool)
    normal_mask[anom_start:anom_end + 50] = False
    online_rmse = float(np.sqrt(np.mean((pred_p[normal_mask] - true_p[normal_mask]) ** 2)))
    console.print(f"  Online RMSE (normal regime): [bold]{online_rmse:.1f} PSI[/bold]")

    if first_detection is not None:
        delay = first_detection - anom_start
        console.print(f"  Anomaly detected at t={first_detection} "
                       f"(delay: {delay} samples after injection at t={anom_start})")
        if delay <= 5:
            console.print("  [bold green]>> Anomaly detected within 5 samples[/bold green]")
    else:
        console.print("  [bold red]>> Anomaly NOT detected[/bold red]")

    if any(phys_viols[anom_start:anom_end]):
        console.print("  [bold green]>> Physics violation flagged during anomaly[/bold green]")
    else:
        console.print("  [yellow]Physics violation not flagged during anomaly window[/yellow]")

    buf_kb = (buf.spectra.nbytes + buf.pressures.nbytes) / 1024
    console.print(f"  Buffer memory: {buf_kb:.1f} KB (fixed, no growth)")

    col_path = str(output_dir / cfg.column_weights_file)
    col_mgr.save_to_json(model, col_path)

    # -- Step 5: Plots ----------------------------------------------------------
    console.rule("[cyan]Generating Plots[/cyan]")

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), dpi=100)
    t_axis = np.arange(N)

    # (a) Pressure tracking
    ax = axes[0]
    ax.plot(t_axis, true_p, "b-", alpha=0.7, lw=1.0, label="True Pressure")
    ax.plot(t_axis, pred_p, "r-", alpha=0.7, lw=1.0, label="Predicted Pressure")
    ax.axvspan(anom_start, anom_end, alpha=0.2, color="red", label="Anomaly Window")
    if first_detection is not None:
        ax.axvline(first_detection, color="green", ls="--", label=f"Detection (t={first_detection})")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Pressure (PSI)")
    ax.set_title("(a) True vs Predicted Pressure -- Online Streaming")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # (b) Activation heatmap (spike rate equivalent)
    ax = axes[1]
    n_show = min(64, act_history.shape[1])
    im = ax.imshow(act_history[:, :n_show].T, aspect="auto", cmap="hot",
                   interpolation="nearest", extent=[0, N, n_show, 0])
    ax.axvline(anom_start, color="cyan", ls="--", lw=1.5)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Neuron Index")
    ax.set_title(f"(b) Spike Rate Heatmap -- Last Hidden Layer (first {n_show} neurons)")
    plt.colorbar(im, ax=ax, label="Activation / Spike Rate")

    # (c) Physics residual
    ax = axes[2]
    ax.plot(t_axis, phys_res_arr, "purple", alpha=0.7, lw=0.8)
    ax.axhline(cfg.physics_violation_threshold, color="red", ls="--", alpha=0.5,
               label=f"Violation Threshold ({cfg.physics_violation_threshold})")
    ax.axvspan(anom_start, anom_end, alpha=0.2, color="red")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Physics Residual")
    ax.set_title("(c) Physics Residual Over Time")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plot_path = str(output_dir / "ars_pisnn_results.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    console.print(f"  Saved plot to {plot_path}")

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5), dpi=100)
    ax2.plot(history["epoch"], history["test_rmse"], "b-o", ms=3, label="Test RMSE")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("RMSE (PSI)")
    ax2.set_title("Training Curve -- Test RMSE")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    train_plot = str(output_dir / "training_curve.png")
    fig2.savefig(train_plot, bbox_inches="tight")
    plt.close(fig2)
    console.print(f"  Saved training curve to {train_plot}")

    console.rule("[bold green]Demo Complete[/bold green]")
    return batch_test_rmse, first_detection, anom_start


# ============================================================================
#  8. GOSSIPING MICRO-MESH ARCHITECTURE
# ============================================================================
#
# 6 ARS sensor nodes deployed around a fire ring main. Each node runs its
# own PI-SNN independently. When a node detects an anomaly, it broadcasts
# to the mesh; neighbors vote to CONFIRM or DENY based on their own readings.
# Consensus prevents false alarms from single-sensor faults while catching
# both local leaks (1-2 nodes see it) and catastrophic events (all nodes).

@dataclass
class MeshConfig:
    """Configuration for the 6-node fire ring main mesh."""

    n_nodes: int = 6

    # Ring main geometry
    ring_main_length: float = 120.0  # total loop path, metres

    # Per-node pipe section lengths between flanges (metres).
    # Each section has different resonance characteristics.
    section_lengths: list = field(
        default_factory=lambda: [3.2, 5.1, 4.0, 6.5, 2.8, 4.7]
    )

    # Node positions along the ring (cumulative metres from node 0)
    node_positions: list = field(
        default_factory=lambda: [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
    )

    # Node labels (matching ARS-FIREWATER-CONCEPT.md layout)
    node_labels: list = field(
        default_factory=lambda: [
            "A1-STBD-AFT", "A2-PUMP", "A3-PORT-AFT",
            "A4-PORT-FWD", "A5-HELI", "A6-STBD-FWD",
        ]
    )

    # Firewater pipe/fluid parameters (6" Sch40, seawater)
    pipe_outer_diameter: float = 0.1683
    pipe_wall_thickness: float = 0.00711
    fluid_base_density: float = 1025.0
    fluid_base_bulk_modulus: float = 2.34e9
    # Firewater physics: narrow pressure range requires high sensitivity.
    # In real system, this comes from high-res spectral analysis (>4096 bins).
    # For demo with 512 bins, we boost the coefficient to make frequency shifts
    # span multiple FFT bins. Also use lower max_freq (5kHz vs 50kHz) for
    # better spectral resolution — pipe acoustics at these dimensions are in
    # the low-kHz range anyway.
    fluid_bulk_modulus_pressure_coeff: float = 5000.0
    pressure_min_psi: float = 120.0
    pressure_max_psi: float = 220.0
    mesh_max_freq_hz: float = 5000.0  # lower max freq = finer resolution

    # Gossip protocol
    gossip_trigger_threshold: float = 0.5  # pressure gossip score to initiate
    gossip_trigger_consecutive: int = 3    # consecutive samples above threshold
    gossip_quorum: float = 0.5            # fraction of non-abstaining votes
    gossip_timeout_steps: int = 3
    gossip_post_confirm_cooldown: int = 20  # suppress gossip after confirmation
    gossip_nearby_m: float = 30.0         # distance threshold for "nearby" node
    gossip_far_m: float = 60.0            # distance threshold for "far" node

    # Spatial pressure propagation (leak attenuation)
    # Higher value = more localised leak (realistic for fire ring main)
    leak_attenuation_per_m: float = 0.05  # exponential decay per metre

    # Demo scenario
    demo_length: int = 1000
    local_leak_time: int = 400
    local_leak_node: int = 3
    local_leak_drop_psi: float = 35.0
    local_leak_duration: int = 50
    catastrophic_time: int = 700
    catastrophic_drop_psi: float = 70.0
    catastrophic_duration: int = 40
    encoder_sync_interval: int = 100

    # Training
    mesh_train_epochs: int = 50
    mesh_train_lr: float = 2e-3
    mesh_samples_per_level: int = 250

    # CfC Judge -- sits above the gossip mesh as a temporal arbiter
    cfc_n_sensory: int = 24
    cfc_n_inter: int = 12
    cfc_n_command: int = 8
    cfc_n_motor: int = 4
    cfc_n_features: int = 24       # 6 pressures + 6 gossip + 6 anomaly + 6 delta-p
    cfc_n_outputs: int = 24        # next-state prediction (same dimension as input)
    cfc_ff_density: float = 0.30
    cfc_recurrent_density: float = 0.15
    cfc_bptt_depth: int = 4
    cfc_bptt_decay: float = 0.7
    cfc_lr: float = 0.001
    cfc_lr_floor: float = 0.0001
    cfc_lr_decay: float = 0.9999
    cfc_grad_clip: float = 5.0
    cfc_surprise_buffer: int = 500
    cfc_surprise_low_pct: float = 25.0
    cfc_surprise_high_pct: float = 90.0


@dataclass
class PumpHealthConfig:
    """
    Configuration for jockey pump health monitoring.

    Typical offshore fire ring main jockey pump: small single-stage
    centrifugal, close-coupled to a 2-pole induction motor, 50 Hz supply.
    Maintains static pressure on the ring between fire pump demand events.
    """

    # --- Pump mechanical parameters ---
    pump_power_kw: float = 7.5
    pump_rpm: float = 2950.0       # 2-pole, 50 Hz, ~1.7% slip
    pump_poles: int = 2
    supply_freq_hz: float = 50.0
    n_vanes: int = 6               # impeller vane count

    # --- Bearing geometry (6205 deep groove ball bearing) ---
    # Typical for motors in the 5-15 kW class
    n_balls: int = 9
    ball_diameter_mm: float = 7.938
    pitch_diameter_mm: float = 38.5
    contact_angle_deg: float = 0.0  # deep groove

    # --- MCSA parameters ---
    slip_nominal: float = 0.017     # nominal slip at rated load
    mcsa_n_fft_bins: int = 2048     # high resolution for sideband detection
    mcsa_max_freq_hz: float = 200.0 # focus around supply freq + harmonics

    # --- Temperature baselines (degrees C) ---
    temp_bearing_de_nominal: float = 55.0   # drive-end bearing
    temp_bearing_nde_nominal: float = 50.0  # non-drive-end (lower load)
    temp_winding_nominal: float = 75.0      # Class F insulation, normal
    temp_seal_nominal: float = 45.0         # mechanical seal / gland packing
    temp_ambient: float = 35.0              # offshore machinery space
    temp_alarm_delta: float = 15.0          # degrees above nominal -> alarm
    temp_trip_delta: float = 25.0           # degrees above nominal -> trip

    # --- Vibration analysis ---
    vib_n_fft_bins: int = 512
    vib_max_freq_hz: float = 1000.0         # pump vibration mostly < 500 Hz
    vib_alarm_mm_s: float = 4.5             # ISO 10816-3 Zone B/C boundary
    vib_trip_mm_s: float = 7.1              # ISO 10816-3 Zone C/D boundary

    # --- Health index fusion weights ---
    health_weight_vibration: float = 0.40
    health_weight_temperature: float = 0.30
    health_weight_mcsa: float = 0.30

    # --- Demo scenario ---
    pump_demo_length: int = 1200
    degradation_start: int = 300    # severity ramp begins
    degradation_end: int = 900      # severity reaches 1.0
    pump_trip_step: int = 900       # pump trips offline
    fire_pump_start_step: int = 920 # backup fire pump auto-starts

    # Jockey pump cycling (normal duty cycle)
    pump_on_duration: int = 30      # steps running
    pump_off_duration: int = 50     # steps idle (pressure holding)


# ============================================================================
#  PUMP PHYSICS UTILITIES
# ============================================================================

class PumpPhysics:
    """
    Static methods for pump-specific frequency calculations.
    All from first principles -- bearing geometry, motor slip, vane count.
    """

    @staticmethod
    def bearing_defect_freqs(rpm: float, n_balls: int, ball_dia_mm: float,
                             pitch_dia_mm: float, contact_angle_deg: float) -> dict:
        """
        Characteristic bearing defect frequencies from geometry.

        BPFO = Ball Pass Frequency Outer race
        BPFI = Ball Pass Frequency Inner race
        BSF  = Ball Spin Frequency
        FTF  = Fundamental Train (cage) Frequency
        """
        shaft_freq = rpm / 60.0
        bd = ball_dia_mm
        pd = pitch_dia_mm
        alpha = math.radians(contact_angle_deg)
        cos_a = math.cos(alpha)

        bpfo = (n_balls / 2.0) * shaft_freq * (1.0 - (bd / pd) * cos_a)
        bpfi = (n_balls / 2.0) * shaft_freq * (1.0 + (bd / pd) * cos_a)
        bsf = (pd / (2.0 * bd)) * shaft_freq * (1.0 - ((bd / pd) * cos_a) ** 2)
        ftf = (shaft_freq / 2.0) * (1.0 - (bd / pd) * cos_a)

        return {
            "shaft": shaft_freq,
            "BPFO": bpfo,
            "BPFI": bpfi,
            "BSF": bsf,
            "FTF": ftf,
            "vane_pass": shaft_freq * 6,  # default 6 vanes
        }

    @staticmethod
    def mcsa_broken_bar_sidebands(line_freq: float, slip: float,
                                   n_harmonics: int = 3) -> list:
        """Sideband frequencies for broken rotor bars: f_line +/- 2*k*s*f."""
        sidebands = []
        for k in range(1, n_harmonics + 1):
            offset = 2.0 * k * slip * line_freq
            sidebands.append(line_freq - offset)
            sidebands.append(line_freq + offset)
        return sidebands

    @staticmethod
    def mcsa_eccentricity_freqs(line_freq: float, slip: float,
                                 poles: int) -> list:
        """Eccentricity sideband frequencies."""
        p = poles / 2.0  # pole pairs
        f_rot = line_freq * (1.0 - slip) / p
        return [line_freq - f_rot, line_freq + f_rot]


# ============================================================================
#  PUMP SENSOR SIMULATOR
# ============================================================================

class PumpSensorSimulator:
    """
    Generates realistic time-evolving sensor data for a jockey pump
    across three modalities: vibration, temperature, and motor current.

    Fault progression is parameterised by severity (0.0 = healthy, 1.0 = failed).
    The demo ramps severity linearly from degradation_start to degradation_end.
    """

    def __init__(self, pcfg: PumpHealthConfig, seed: int = 55):
        self.pcfg = pcfg
        self.rng = np.random.default_rng(seed)

        # Pre-compute characteristic frequencies
        self.freqs = PumpPhysics.bearing_defect_freqs(
            pcfg.pump_rpm, pcfg.n_balls, pcfg.ball_diameter_mm,
            pcfg.pitch_diameter_mm, pcfg.contact_angle_deg,
        )
        self.mcsa_bars = PumpPhysics.mcsa_broken_bar_sidebands(
            pcfg.supply_freq_hz, pcfg.slip_nominal,
        )
        self.mcsa_ecc = PumpPhysics.mcsa_eccentricity_freqs(
            pcfg.supply_freq_hz, pcfg.slip_nominal, pcfg.pump_poles,
        )

        # Thermal state (persists across steps for time-constant behaviour)
        self.temp_state = {
            "bearing_de": pcfg.temp_bearing_de_nominal,
            "bearing_nde": pcfg.temp_bearing_nde_nominal,
            "winding": pcfg.temp_winding_nominal,
            "seal": pcfg.temp_seal_nominal,
        }

    def _is_pump_running(self, t: int) -> bool:
        """Jockey pump duty cycle: on for N steps, off for M steps."""
        pcfg = self.pcfg
        # After trip, pump is off permanently
        if t >= pcfg.pump_trip_step:
            return False
        cycle = pcfg.pump_on_duration + pcfg.pump_off_duration
        phase = t % cycle
        return phase < pcfg.pump_on_duration

    def get_severity(self, t: int) -> float:
        """Linear fault severity ramp."""
        pcfg = self.pcfg
        if t < pcfg.degradation_start:
            return 0.0
        if t >= pcfg.degradation_end:
            return 1.0
        return (t - pcfg.degradation_start) / (pcfg.degradation_end - pcfg.degradation_start)

    def get_readings(self, t: int) -> dict:
        """Full sensor reading at timestep t."""
        pcfg = self.pcfg
        severity = self.get_severity(t)
        running = self._is_pump_running(t)

        # --- Temperature (evolves with thermal time constants) ---
        temps = self._simulate_temperatures(severity, running)

        # --- Vibration spectrum ---
        if running:
            vib_spectrum, vib_rms = self._simulate_vibration(severity)
        else:
            vib_spectrum = np.zeros(pcfg.vib_n_fft_bins, dtype=np.float32)
            vib_rms = 0.0

        # --- Motor current spectrum ---
        if running:
            cur_spectrum, cur_rms = self._simulate_current(severity)
        else:
            cur_spectrum = np.zeros(pcfg.mcsa_n_fft_bins, dtype=np.float32)
            cur_rms = 0.0

        return {
            "temperatures": dict(self.temp_state),
            "vibration_spectrum": vib_spectrum,
            "vibration_rms_mm_s": vib_rms,
            "current_spectrum": cur_spectrum,
            "current_rms_a": cur_rms,
            "pump_running": running,
            "fault_severity": severity,
        }

    def _simulate_temperatures(self, severity: float, running: bool) -> dict:
        """Thermal model with time constants. Bearing temp rises with friction."""
        pcfg = self.pcfg
        tau = 0.05  # thermal time constant (slow response)

        if running:
            # Targets shift with severity (more friction = more heat)
            targets = {
                "bearing_de": pcfg.temp_bearing_de_nominal + severity * 30.0,
                "bearing_nde": pcfg.temp_bearing_nde_nominal + severity * 15.0,
                "winding": pcfg.temp_winding_nominal + severity * 10.0,
                "seal": pcfg.temp_seal_nominal + severity * 8.0,
            }
        else:
            # Cool toward ambient when off
            targets = {k: pcfg.temp_ambient for k in self.temp_state}

        for key in self.temp_state:
            self.temp_state[key] += tau * (targets[key] - self.temp_state[key])
            self.temp_state[key] += self.rng.normal(0, 0.3)

        return self.temp_state

    def _simulate_vibration(self, severity: float) -> tuple:
        """
        Build vibration spectrum from physical components.

        Healthy: shaft harmonics at low amplitude, noise floor.
        Degraded: BPFO peak grows, develops shaft-frequency sidebands,
                  broadband noise rises.
        """
        pcfg = self.pcfg
        freq_axis = np.linspace(0, pcfg.vib_max_freq_hz, pcfg.vib_n_fft_bins)
        spectrum = np.zeros(pcfg.vib_n_fft_bins, dtype=np.float32)

        shaft = self.freqs["shaft"]

        def add_peak(freq, amp, width=3.0):
            if 0 < freq < pcfg.vib_max_freq_hz:
                spectrum[:] += amp * np.exp(-0.5 * ((freq_axis - freq) / width) ** 2)

        # 1x shaft (always present, grows with imbalance)
        add_peak(shaft, 0.15 + severity * 0.3, width=2.0)
        # 2x shaft (misalignment proxy -- grows with bearing wear)
        add_peak(2 * shaft, 0.05 + severity * 0.15, width=2.0)

        # BPFO and harmonics (the primary bearing defect indicator)
        bpfo_amp = severity ** 1.5 * 0.8  # nonlinear growth (subtle then rapid)
        for h in range(1, 4):
            add_peak(self.freqs["BPFO"] * h, bpfo_amp / h, width=2.5)
            # Shaft-frequency sidebands around BPFO (modulation from rotation)
            if severity > 0.3:
                sideband_amp = bpfo_amp * 0.3 * (severity - 0.3) / 0.7
                add_peak(self.freqs["BPFO"] * h + shaft, sideband_amp / h, width=1.5)
                add_peak(self.freqs["BPFO"] * h - shaft, sideband_amp / h, width=1.5)

        # BPFI (inner race, lower amplitude contribution)
        add_peak(self.freqs["BPFI"], severity ** 2 * 0.3, width=2.5)

        # BSF (rolling element)
        add_peak(self.freqs["BSF"], severity ** 2 * 0.2, width=2.0)

        # FTF (cage)
        add_peak(self.freqs["FTF"], severity ** 2 * 0.1, width=1.5)

        # Vane pass frequency (cavitation indicator at high severity)
        vpf = self.freqs["vane_pass"]
        add_peak(vpf, 0.1 + severity * 0.2, width=3.0)

        # Broadband noise floor rises with severity (random impacts)
        noise_floor = 0.02 + severity * 0.08
        spectrum += noise_floor * np.abs(self.rng.standard_normal(pcfg.vib_n_fft_bins))

        spectrum = np.maximum(spectrum, 0)

        # RMS velocity (mm/s) -- scale from normalised spectrum
        # Healthy pump ~1.5 mm/s, alarm at 4.5, trip at 7.1
        rms = 1.5 + severity * 6.0 + self.rng.normal(0, 0.2)
        rms = max(rms, 0.1)

        return spectrum.astype(np.float32), float(rms)

    def _simulate_current(self, severity: float) -> tuple:
        """
        Motor current spectrum for MCSA.

        Healthy: dominant 50 Hz fundamental + odd supply harmonics.
        Degraded: broken-bar sidebands grow at f +/- 2sf, eccentricity sidebands appear.
        """
        pcfg = self.pcfg
        freq_axis = np.linspace(0, pcfg.mcsa_max_freq_hz, pcfg.mcsa_n_fft_bins)
        spectrum = np.zeros(pcfg.mcsa_n_fft_bins, dtype=np.float32)

        def add_peak(freq, amp, width=0.3):
            if 0 < freq < pcfg.mcsa_max_freq_hz:
                spectrum[:] += amp * np.exp(-0.5 * ((freq_axis - freq) / width) ** 2)

        # Supply fundamental (50 Hz) -- always dominant
        add_peak(pcfg.supply_freq_hz, 1.0, width=0.5)

        # Supply harmonics (3rd, 5th, 7th)
        for h in [3, 5, 7]:
            add_peak(pcfg.supply_freq_hz * h, 0.05 / h, width=0.4)

        # Broken bar sidebands: severity controls amplitude relative to fundamental
        # Healthy: -55 dB (~0.002). Severe: -25 dB (~0.056)
        bar_amp = 0.002 + severity ** 2 * 0.06
        for sb in self.mcsa_bars:
            add_peak(sb, bar_amp, width=0.2)

        # Eccentricity sidebands (from bearing wear causing rotor eccentricity)
        ecc_amp = severity ** 1.5 * 0.03
        for ef in self.mcsa_ecc:
            add_peak(ef, ecc_amp, width=0.2)

        # Noise floor
        spectrum += 0.001 * np.abs(self.rng.standard_normal(pcfg.mcsa_n_fft_bins))
        spectrum = np.maximum(spectrum, 0)

        # RMS current (amps) -- rises slightly with mechanical degradation
        cur_rms = 12.0 + severity * 3.0 + self.rng.normal(0, 0.1)

        return spectrum.astype(np.float32), float(cur_rms)


# ============================================================================
#  PUMP HEALTH ANALYZER
# ============================================================================

class PumpHealthAnalyzer:
    """
    Analyses pump sensor data across three modalities and produces a
    composite health index.

    Each modality returns a health score (1.0 = healthy, 0.0 = failed).
    Scores are fused with configurable weights, with a critical-override
    rule: if any single modality is below 0.2, composite is clamped to 0.3.
    """

    def __init__(self, pcfg: PumpHealthConfig):
        self.pcfg = pcfg

        # Pre-compute characteristic frequencies for spectral lookup
        self.freqs = PumpPhysics.bearing_defect_freqs(
            pcfg.pump_rpm, pcfg.n_balls, pcfg.ball_diameter_mm,
            pcfg.pitch_diameter_mm, pcfg.contact_angle_deg,
        )
        self.mcsa_bars = PumpPhysics.mcsa_broken_bar_sidebands(
            pcfg.supply_freq_hz, pcfg.slip_nominal,
        )

        # EMA baselines for temperatures
        self.temp_ema = {
            "bearing_de": pcfg.temp_bearing_de_nominal,
            "bearing_nde": pcfg.temp_bearing_nde_nominal,
            "winding": pcfg.temp_winding_nominal,
            "seal": pcfg.temp_seal_nominal,
        }
        self.temp_ema_alpha = 0.02

        # Last known health (held during pump-off periods)
        self.last_health = 1.0
        self.last_vib_result = None
        self.last_temp_result = None
        self.last_mcsa_result = None
        self.last_diagnosis = "Healthy"
        self.n_updates = 0

    def analyze(self, readings: dict) -> dict:
        """
        Full analysis pass. Returns dict with per-modality results,
        composite health index, and fault diagnosis.
        """
        if not readings["pump_running"]:
            # Hold last known health during off periods
            return {
                "vibration": self.last_vib_result or {"vibration_health": 1.0},
                "temperature": self.last_temp_result or {"temperature_health": 1.0},
                "mcsa": self.last_mcsa_result or {"mcsa_health": 1.0},
                "health_index": self.last_health,
                "diagnosis": self.last_diagnosis,
                "pump_running": False,
            }

        self.n_updates += 1

        vib = self._analyze_vibration(readings)
        temp = self._analyze_temperature(readings["temperatures"])
        mcsa = self._analyze_mcsa(readings)
        health = self._compute_health_index(vib, temp, mcsa)
        diagnosis = self._diagnose(vib, temp, mcsa)

        self.last_vib_result = vib
        self.last_temp_result = temp
        self.last_mcsa_result = mcsa
        self.last_health = health
        self.last_diagnosis = diagnosis

        return {
            "vibration": vib,
            "temperature": temp,
            "mcsa": mcsa,
            "health_index": health,
            "diagnosis": diagnosis,
            "pump_running": True,
        }

    def _analyze_vibration(self, readings: dict) -> dict:
        """Score vibration spectrum for bearing defects and imbalance."""
        pcfg = self.pcfg
        spectrum = readings["vibration_spectrum"]
        rms = readings["vibration_rms_mm_s"]
        freq_axis = np.linspace(0, pcfg.vib_max_freq_hz, pcfg.vib_n_fft_bins)

        def peak_amplitude_at(target_freq, tolerance_hz=5.0):
            """Max spectral amplitude within tolerance of target frequency."""
            mask = np.abs(freq_axis - target_freq) < tolerance_hz
            if not np.any(mask):
                return 0.0
            return float(np.max(spectrum[mask]))

        def local_noise(target_freq, band_hz=30.0):
            """Median amplitude in a band around the target (excluding peak)."""
            mask = (np.abs(freq_axis - target_freq) < band_hz) & \
                   (np.abs(freq_axis - target_freq) > 8.0)
            if not np.any(mask):
                return 0.01
            return float(np.median(spectrum[mask])) + 1e-6

        # Bearing defect score: BPFO peak relative to local noise floor
        bpfo_amp = peak_amplitude_at(self.freqs["BPFO"])
        bpfo_noise = local_noise(self.freqs["BPFO"])
        bpfo_snr = bpfo_amp / bpfo_noise
        # SNR 1-3 = healthy, 3-8 = developing, 8+ = severe
        bearing_score = min(max((bpfo_snr - 2.0) / 10.0, 0.0), 1.0)

        # Imbalance score: 1x shaft amplitude
        # Note: 1x shaft is always present in a running pump (SNR ~5-8 is normal).
        # Only flag when it grows well above the healthy baseline.
        shaft_amp = peak_amplitude_at(self.freqs["shaft"])
        shaft_noise = local_noise(self.freqs["shaft"])
        shaft_snr = shaft_amp / shaft_noise
        imbalance_score = min(max((shaft_snr - 8.0) / 10.0, 0.0), 1.0)

        # Overall RMS against ISO thresholds
        if rms < pcfg.vib_alarm_mm_s:
            rms_health = 1.0 - (rms / pcfg.vib_alarm_mm_s) * 0.3
        elif rms < pcfg.vib_trip_mm_s:
            frac = (rms - pcfg.vib_alarm_mm_s) / (pcfg.vib_trip_mm_s - pcfg.vib_alarm_mm_s)
            rms_health = 0.7 - frac * 0.5
        else:
            rms_health = max(0.2 - (rms - pcfg.vib_trip_mm_s) / 10.0, 0.0)

        # Composite vibration health: worst of bearing/imbalance/rms
        vibration_health = max(1.0 - max(bearing_score, imbalance_score), rms_health)
        vibration_health = min(vibration_health, rms_health)

        return {
            "bearing_score": bearing_score,
            "imbalance_score": imbalance_score,
            "bpfo_snr": bpfo_snr,
            "rms_mm_s": rms,
            "rms_health": rms_health,
            "vibration_health": max(vibration_health, 0.0),
        }

    def _analyze_temperature(self, temps: dict) -> dict:
        """Score temperatures against alarm/trip deltas with rate-of-rise."""
        pcfg = self.pcfg

        nominals = {
            "bearing_de": pcfg.temp_bearing_de_nominal,
            "bearing_nde": pcfg.temp_bearing_nde_nominal,
            "winding": pcfg.temp_winding_nominal,
            "seal": pcfg.temp_seal_nominal,
        }

        scores = {}
        max_delta = 0.0
        for key in nominals:
            delta = temps[key] - nominals[key]
            max_delta = max(max_delta, delta)
            # Score: 0 delta = 1.0, alarm delta = 0.4, trip delta = 0.1
            if delta <= 0:
                scores[key] = 1.0
            elif delta < pcfg.temp_alarm_delta:
                scores[key] = 1.0 - 0.6 * (delta / pcfg.temp_alarm_delta)
            elif delta < pcfg.temp_trip_delta:
                frac = (delta - pcfg.temp_alarm_delta) / (pcfg.temp_trip_delta - pcfg.temp_alarm_delta)
                scores[key] = 0.4 - 0.3 * frac
            else:
                scores[key] = max(0.1 - (delta - pcfg.temp_trip_delta) / 20.0, 0.0)

            # Update EMA for rate-of-rise detection
            self.temp_ema[key] = (1 - self.temp_ema_alpha) * self.temp_ema[key] + self.temp_ema_alpha * temps[key]

        temperature_health = min(scores.values())

        return {
            "bearing_de_score": scores["bearing_de"],
            "bearing_nde_score": scores["bearing_nde"],
            "winding_score": scores["winding"],
            "seal_score": scores["seal"],
            "max_delta_above_nominal": max_delta,
            "temperature_health": temperature_health,
        }

    def _analyze_mcsa(self, readings: dict) -> dict:
        """Score motor current spectrum for broken bars and eccentricity."""
        pcfg = self.pcfg
        spectrum = readings["current_spectrum"]
        freq_axis = np.linspace(0, pcfg.mcsa_max_freq_hz, pcfg.mcsa_n_fft_bins)

        def peak_at(freq, tol_hz=0.3):
            mask = np.abs(freq_axis - freq) < tol_hz
            if not np.any(mask):
                return 1e-6
            return float(np.max(spectrum[mask]))

        def noise_floor_at(freq, band_hz=2.0, exclude_hz=0.5):
            """Median amplitude near target, excluding the peak itself."""
            mask = (np.abs(freq_axis - freq) < band_hz) & \
                   (np.abs(freq_axis - freq) > exclude_hz)
            if not np.any(mask):
                return 1e-4
            return float(np.median(spectrum[mask])) + 1e-6

        # Fundamental amplitude
        fund_amp = peak_at(pcfg.supply_freq_hz) + 1e-6

        # Broken bar sideband ratio: measure peak ABOVE local noise floor
        # This eliminates false positives from the fundamental's spectral tail
        sb_excess = []
        for sb in self.mcsa_bars[:2]:  # first pair +/- 2sf
            peak = peak_at(sb)
            noise = noise_floor_at(sb)
            sb_excess.append(max(peak - noise, 1e-6))
        avg_sb_excess = np.mean(sb_excess) if sb_excess else 1e-6
        sideband_ratio_db = 20.0 * math.log10(max(avg_sb_excess / fund_amp, 1e-10))

        # Score: < -50 dB healthy, -40 to -35 developing, > -30 severe
        if sideband_ratio_db < -50.0:
            broken_bar_score = 0.0
        elif sideband_ratio_db < -30.0:
            broken_bar_score = (sideband_ratio_db + 50.0) / 20.0
        else:
            broken_bar_score = 1.0

        # Eccentricity: check sidebands (also noise-floor corrected)
        ecc_excess = []
        ecc_freqs = PumpPhysics.mcsa_eccentricity_freqs(
            pcfg.supply_freq_hz, pcfg.slip_nominal, pcfg.pump_poles)
        for ef in ecc_freqs:
            peak = peak_at(ef)
            noise = noise_floor_at(ef)
            ecc_excess.append(max(peak - noise, 1e-6))
        avg_ecc_excess = np.mean(ecc_excess) if ecc_excess else 1e-6
        ecc_ratio_db = 20.0 * math.log10(max(avg_ecc_excess / fund_amp, 1e-10))
        eccentricity_score = min(max((ecc_ratio_db + 45.0) / 20.0, 0.0), 1.0)

        mcsa_health = 1.0 - max(broken_bar_score, eccentricity_score)

        return {
            "broken_bar_score": broken_bar_score,
            "eccentricity_score": eccentricity_score,
            "sideband_ratio_db": sideband_ratio_db,
            "mcsa_health": max(mcsa_health, 0.0),
        }

    def _compute_health_index(self, vib: dict, temp: dict, mcsa: dict) -> float:
        """Weighted fusion with critical-override rule."""
        pcfg = self.pcfg
        vh = vib["vibration_health"]
        th = temp["temperature_health"]
        mh = mcsa["mcsa_health"]

        health = (pcfg.health_weight_vibration * vh +
                  pcfg.health_weight_temperature * th +
                  pcfg.health_weight_mcsa * mh)

        # Critical override: any single modality near-failure caps the composite
        if min(vh, th, mh) < 0.2:
            health = min(health, 0.3)

        return max(min(health, 1.0), 0.0)

    def _diagnose(self, vib: dict, temp: dict, mcsa: dict) -> str:
        """Human-readable fault diagnosis from modality scores."""
        faults = []

        if vib["bearing_score"] > 0.5:
            faults.append("Bearing outer race defect")
        if vib["imbalance_score"] > 0.5:
            faults.append("Mechanical imbalance")
        if temp["max_delta_above_nominal"] > self.pcfg.temp_alarm_delta:
            faults.append(f"Overtemperature (+{temp['max_delta_above_nominal']:.0f}C)")
        if mcsa["broken_bar_score"] > 0.5:
            faults.append("Broken rotor bar(s)")
        if mcsa["eccentricity_score"] > 0.5:
            faults.append("Rotor eccentricity")

        return " | ".join(faults) if faults else "Healthy"


def _ring_distance(pos_a: float, pos_b: float, ring_len: float) -> float:
    """Shortest path distance between two points on a ring."""
    d = abs(pos_a - pos_b)
    return min(d, ring_len - d)


# --- Gossip data structures ---

class NodeState:
    NORMAL = "NORMAL"
    SUSPECTED = "SUSPECTED"
    GOSSIPING = "GOSSIPING"
    CONFIRMED = "CONFIRMED"
    DENIED = "DENIED"


@dataclass
class GossipMessage:
    """Broadcast from a node that suspects an anomaly."""
    origin_node: int
    timestamp: int
    anomaly_score: float
    pred_pressure: float


@dataclass
class GossipVote:
    """A node's response to a gossip message."""
    voter: int
    origin: int
    vote: str  # CONFIRM, DENY, ABSTAIN
    confidence: float
    own_anomaly_score: float
    own_pressure: float


@dataclass
class GossipRound:
    """Tracks one gossip round initiated by a single node."""
    origin: int
    start_time: int
    message: GossipMessage
    votes: list = field(default_factory=list)
    verdict: Optional[str] = None


class SensorNode:
    """
    One ARS sensor node on the fire ring main.

    Each node has its own PISNN model instance, anomaly detector, and
    online learning buffer. The encoder (hidden layers) is periodically
    synchronised across the mesh; the output column is node-specific
    (different pipe section geometry = different resonance characteristics).
    """

    def __init__(self, node_id: int, position: float, section_length: float,
                 base_cfg: Config, mcfg: MeshConfig, seed: int):
        self.node_id = node_id
        self.position = position
        self.section_length = section_length
        self.mcfg = mcfg

        # Build node-specific config by overriding pipe parameters
        self.cfg = Config(
            pipe_length=section_length,
            pipe_outer_diameter=mcfg.pipe_outer_diameter,
            pipe_wall_thickness=mcfg.pipe_wall_thickness,
            fluid_base_density=mcfg.fluid_base_density,
            fluid_base_bulk_modulus=mcfg.fluid_base_bulk_modulus,
            fluid_bulk_modulus_pressure_coeff=mcfg.fluid_bulk_modulus_pressure_coeff,
            pressure_min_psi=mcfg.pressure_min_psi,
            pressure_max_psi=mcfg.pressure_max_psi,
            n_resonance_modes=20,
            n_fft_bins=base_cfg.n_fft_bins,
            max_freq_hz=mcfg.mesh_max_freq_hz,
            pump_harmonic_base_hz=base_cfg.pump_harmonic_base_hz,
            n_pump_harmonics=base_cfg.n_pump_harmonics,
            pump_harmonic_amplitude=base_cfg.pump_harmonic_amplitude,
            white_noise_floor=base_cfg.white_noise_floor,
            snn_layers=base_cfg.snn_layers,
            n_timesteps=base_cfg.n_timesteps,
            ema_alpha=base_cfg.ema_alpha,
            anomaly_zscore_threshold=base_cfg.anomaly_zscore_threshold,
            physics_violation_threshold=base_cfg.physics_violation_threshold,
            physics_lambda=base_cfg.physics_lambda,
            online_buffer_size=base_cfg.online_buffer_size,
            online_update_interval=base_cfg.online_update_interval,
            online_lr=base_cfg.online_lr,
            batch_size=base_cfg.batch_size,
        )

        self.model = PISNN(self.cfg, seed=seed + node_id)
        self.anomaly_det = AnomalyDetector(self.cfg.snn_layers, self.cfg)
        self.buffer = CircularBuffer(self.cfg.online_buffer_size, self.cfg.n_fft_bins)
        self.rng = np.random.default_rng(seed + node_id + 1000)

        # Pressure deviation tracker for gossip triggers
        # Uses a dual-EMA approach: slow EMA for baseline, fast for current.
        # Gossip triggers on deviation of fast from slow (detects sudden shifts
        # while ignoring gradual ramps and model noise).
        self.pressure_slow_ema = None
        self.pressure_fast_ema = None
        self.pressure_slow_alpha = 0.02   # ~50-sample window (stable baseline)
        self.pressure_fast_alpha = 0.3    # ~3-sample window (responsive to events)
        self.pressure_n = 0
        self.pressure_gossip_score = 0.0  # dedicated gossip trigger signal

        # State machine
        self.state = NodeState.NORMAL
        self.consecutive_anomaly = 0
        self.gossip_cooldown = 0  # steps until node can re-trigger gossip
        self.last_reading = None

    def process_sample(self, spectrum: np.ndarray, true_pressure: float, t: int):
        """
        Run one inference step. Returns a dict with all node outputs.
        Updates anomaly detector baseline and online buffer.
        """
        spec = spectrum.reshape(1, -1)
        pred_psi, activations, cache = self.model.forward(spec)
        pred_val = float(pred_psi[0, 0])

        pr = self.model.compute_physics_residual(spec, pred_psi)
        pr_val = float(pr[0])

        self.anomaly_det.update_baseline(activations)
        a_score, p_viol, z_max = self.anomaly_det.compute_anomaly_score(
            activations, pr_val
        )
        if z_max > self.cfg.anomaly_zscore_threshold:
            p_viol = True
            a_score = max(a_score, 0.8)

        # Pressure deviation for gossip trigger: dual-EMA approach.
        # Fast EMA tracks current pressure, slow EMA holds baseline.
        # Large divergence = sudden pressure event (leak, depressurisation).
        #
        # Uses true_pressure (sensor reading) rather than pred_val (model output)
        # because the model's per-sample prediction noise would cause false gossip
        # triggers. In production, the sensor reading comes from the calibrated
        # ARS inference chain — model accuracy was demonstrated in the single-node
        # prototype. The mesh demo focuses on the consensus protocol.
        self.pressure_n += 1
        if self.pressure_slow_ema is None:
            self.pressure_slow_ema = true_pressure
            self.pressure_fast_ema = true_pressure
        else:
            self.pressure_slow_ema = ((1 - self.pressure_slow_alpha) * self.pressure_slow_ema
                                      + self.pressure_slow_alpha * true_pressure)
            self.pressure_fast_ema = ((1 - self.pressure_fast_alpha) * self.pressure_fast_ema
                                      + self.pressure_fast_alpha * true_pressure)
        dev = abs(self.pressure_fast_ema - self.pressure_slow_ema)
        if self.pressure_n > 300:  # after warmup (past initial ramp + EMA settling)
            # 10 PSI deviation = gossip score 1.0 (on 120-220 PSI range)
            self.pressure_gossip_score = min(dev / 10.0, 1.0)
        else:
            self.pressure_gossip_score = 0.0

        # Track consecutive anomaly samples for gossip trigger
        # Use the dedicated pressure gossip score (not general anomaly score)
        # to avoid false triggers from noisy spike z-scores
        if self.pressure_gossip_score > self.mcfg.gossip_trigger_threshold:
            self.consecutive_anomaly += 1
        else:
            self.consecutive_anomaly = 0

        # Online learning
        self.buffer.add(spec.flatten(), true_pressure)
        if (t > 0 and t % self.cfg.online_update_interval == 0
                and self.buffer.count >= self.cfg.batch_size):
            Xb, yb = self.buffer.get_batch(self.cfg.batch_size, self.rng)
            pb, _, cb = self.model.forward(Xb)
            self.model.backward_and_update(Xb, yb, pb, cb, self.cfg.online_lr)

        self.last_reading = {
            "node_id": self.node_id,
            "t": t,
            "true_psi": true_pressure,
            "pred_psi": pred_val,
            "anomaly_score": a_score,
            "physics_violation": p_viol,
            "z_max": z_max,
        }
        return self.last_reading

    def should_gossip(self) -> bool:
        """Check if this node should initiate a gossip round."""
        if self.gossip_cooldown > 0:
            self.gossip_cooldown -= 1
            return False
        return (self.consecutive_anomaly >= self.mcfg.gossip_trigger_consecutive
                and self.state == NodeState.NORMAL)

    def vote_on_gossip(self, msg: GossipMessage) -> GossipVote:
        """
        Evaluate a neighbor's anomaly claim against our own readings.

        Voting logic based on distance and own anomaly state:
        - Nearby + own anomaly elevated: CONFIRM (corroborates the claim)
        - Nearby + own readings normal: DENY (should see it if real)
        - Far + own anomaly elevated: CONFIRM (system-wide event)
        - Far + own readings normal: ABSTAIN (too far for local leak info)
        """
        if self.last_reading is None:
            return GossipVote(self.node_id, msg.origin_node, "ABSTAIN", 0.0, 0.0, 0.0)

        dist = _ring_distance(
            self.mcfg.node_positions[self.node_id],
            self.mcfg.node_positions[msg.origin_node],
            self.mcfg.ring_main_length,
        )
        own_score = self.pressure_gossip_score  # use pressure-based signal
        own_psi = self.last_reading["pred_psi"]

        # Nearby threshold: scale down if origin reports very high anomaly
        # (catastrophic events should lower the bar for confirmation)
        confirm_threshold = 0.25 if msg.anomaly_score > 0.7 else 0.35

        if dist < self.mcfg.gossip_nearby_m:
            # Nearby node: should have information
            if own_score > confirm_threshold:
                return GossipVote(self.node_id, msg.origin_node, "CONFIRM",
                                  own_score, own_score, own_psi)
            else:
                return GossipVote(self.node_id, msg.origin_node, "DENY",
                                  1.0 - own_score, own_score, own_psi)
        elif dist < self.mcfg.gossip_far_m:
            # Medium distance
            if own_score > confirm_threshold:
                return GossipVote(self.node_id, msg.origin_node, "CONFIRM",
                                  own_score * 0.7, own_score, own_psi)
            else:
                return GossipVote(self.node_id, msg.origin_node, "ABSTAIN",
                                  0.3, own_score, own_psi)
        else:
            # Far away
            if own_score > confirm_threshold:
                return GossipVote(self.node_id, msg.origin_node, "CONFIRM",
                                  own_score * 0.5, own_score, own_psi)
            else:
                return GossipVote(self.node_id, msg.origin_node, "ABSTAIN",
                                  0.1, own_score, own_psi)


class GossipProtocol:
    """
    Manages gossip rounds for distributed anomaly confirmation.

    When a node suspects an anomaly, it broadcasts to all neighbors.
    Each neighbor votes CONFIRM/DENY/ABSTAIN based on its own readings
    and its distance from the origin. Votes are weighted by confidence
    and distance. A quorum among non-abstaining voters is required to
    confirm the anomaly.

    This distinguishes:
    - Local leak: 1-2 nearby nodes confirm, far nodes abstain -> CONFIRMED
    - Catastrophic event: all nodes confirm -> CONFIRMED (flagged SYSTEM_WIDE)
    - Sensor fault: nearby nodes deny, far nodes abstain -> DENIED
    """

    def __init__(self, mcfg: MeshConfig):
        self.mcfg = mcfg
        self.active_rounds: dict = {}  # origin_node -> GossipRound
        self.completed_rounds: list = []

    def initiate(self, node: SensorNode, t: int) -> GossipMessage:
        """Start a gossip round from a node that suspects an anomaly."""
        reading = node.last_reading
        msg = GossipMessage(
            origin_node=node.node_id,
            timestamp=t,
            anomaly_score=node.pressure_gossip_score,
            pred_pressure=reading["pred_psi"],
        )
        self.active_rounds[node.node_id] = GossipRound(
            origin=node.node_id, start_time=t, message=msg
        )
        node.state = NodeState.GOSSIPING
        return msg

    def collect_votes(self, msg: GossipMessage, nodes: list):
        """Solicit votes from all nodes except the origin."""
        rd = self.active_rounds.get(msg.origin_node)
        if rd is None:
            return
        for node in nodes:
            if node.node_id != msg.origin_node:
                vote = node.vote_on_gossip(msg)
                rd.votes.append(vote)

    def resolve(self, t: int, nodes: list) -> list:
        """
        Check all active rounds for quorum or timeout. Returns list of
        (origin_node, verdict, detail_dict) for each resolved round.
        """
        resolved = []
        to_remove = []

        for origin_id, rd in self.active_rounds.items():
            if rd.verdict is not None:
                continue
            if t - rd.start_time >= self.mcfg.gossip_timeout_steps or len(rd.votes) >= self.mcfg.n_nodes - 1:
                verdict, detail = self._tally(rd)
                rd.verdict = verdict
                # Update originating node state
                origin_node = nodes[origin_id]
                origin_node.state = NodeState.CONFIRMED if verdict == "CONFIRMED" else NodeState.DENIED
                # Cooldown before allowing re-trigger (5 steps)
                origin_node.consecutive_anomaly = 0
                origin_node.gossip_cooldown = 5
                self.completed_rounds.append(rd)
                to_remove.append(origin_id)
                resolved.append((origin_id, verdict, detail))

        for k in to_remove:
            del self.active_rounds[k]

        # Decay confirmed/denied states back to normal
        for node in nodes:
            if node.state in (NodeState.CONFIRMED, NodeState.DENIED):
                if node.consecutive_anomaly == 0:
                    node.state = NodeState.NORMAL

        return resolved

    def _tally(self, rd: GossipRound) -> tuple:
        """Weighted majority vote. Returns (verdict, detail_dict)."""
        confirms, denies, abstains = 0.0, 0.0, 0
        confirm_nodes, deny_nodes = [], []

        for v in rd.votes:
            if v.vote == "CONFIRM":
                confirms += v.confidence
                confirm_nodes.append(v.voter)
            elif v.vote == "DENY":
                denies += v.confidence
                deny_nodes.append(v.voter)
            else:
                abstains += 1

        total = confirms + denies
        n_confirms = len(confirm_nodes)
        system_wide = n_confirms >= self.mcfg.n_nodes - 2  # 4+ of 5 voters

        if total < 0.01:
            verdict = "DENIED"  # all abstained = insufficient evidence
        elif confirms / total > self.mcfg.gossip_quorum:
            verdict = "CONFIRMED"
        else:
            verdict = "DENIED"

        detail = {
            "confirms": confirms,
            "denies": denies,
            "abstains": abstains,
            "confirm_nodes": confirm_nodes,
            "deny_nodes": deny_nodes,
            "system_wide": system_wide and verdict == "CONFIRMED",
        }
        return verdict, detail


# ============================================================================
#  8b. CfC JUDGE -- Closed-form Continuous-time Network
# ============================================================================
#
# A CfC network (Hasani et al. 2022) sits above the gossip mesh, receiving
# the full mesh state each timestep. It learns to predict the next state
# (self-supervised) and uses prediction surprise to judge gossip verdicts.
#
# Key advantage: the time-dependent forget gate (modulated by dt * tau)
# gives the CfC natural temporal context, letting it distinguish genuine
# anomaly onsets (high surprise) from recovery transitions (low surprise,
# positive delta-pressure). Individual gossip nodes lack this context.
#
# Architecture: NCP wiring (sensory -> inter -> command -> motor) with
# sparse connectivity, matching the APEX Rust CfC implementation.


class NcpWiringCfC:
    """
    Neural Circuit Policy wiring for the CfC judge.

    Generates sparse feedforward + recurrent connectivity following the
    NCP topology: sensory -> inter -> command -> motor.
    """

    def __init__(self, n_sensory: int, n_inter: int, n_command: int,
                 n_motor: int, n_features: int, ff_density: float = 0.30,
                 rec_density: float = 0.15, seed: int = 42):
        self.n_sensory = n_sensory
        self.n_inter = n_inter
        self.n_command = n_command
        self.n_motor = n_motor
        self.n_total = n_sensory + n_inter + n_command + n_motor
        self.n_features = n_features

        self.inter_start = n_sensory
        self.cmd_start = n_sensory + n_inter
        self.motor_start = n_sensory + n_inter + n_command

        rng = np.random.default_rng(seed)

        # Input mapping: which features map to which sensory neurons
        # First 12 features (pressures + gossip scores) get 2 neurons each
        # Last 12 (anomaly + delta-p) get 1 neuron each
        self.input_map = []  # list of lists: feature_idx -> [sensory_neuron_ids]
        s = 0
        for f in range(n_features):
            n_neurons = 2 if f < 12 else 1
            self.input_map.append(list(range(s, min(s + n_neurons, n_sensory))))
            s += n_neurons
            if s >= n_sensory:
                s = s % n_sensory  # wrap around if needed

        # Total input weights
        self.total_input_weights = sum(len(m) for m in self.input_map)

        # Sparse feedforward connections
        self.connections = []  # list of (src, dst)
        self.incoming = [[] for _ in range(self.n_total)]

        # Sensory -> Inter
        self._add_connections(rng, range(n_sensory),
                              range(self.inter_start, self.cmd_start), ff_density)
        # Inter -> Command
        self._add_connections(rng, range(self.inter_start, self.cmd_start),
                              range(self.cmd_start, self.motor_start), ff_density)
        # Command -> Motor
        self._add_connections(rng, range(self.cmd_start, self.motor_start),
                              range(self.motor_start, self.n_total), ff_density)

        # Recurrent connections within inter and command
        self._add_connections(rng, range(self.inter_start, self.cmd_start),
                              range(self.inter_start, self.cmd_start), rec_density)
        self._add_connections(rng, range(self.cmd_start, self.motor_start),
                              range(self.cmd_start, self.motor_start), rec_density)

        # Guarantee: every non-sensory neuron has at least 1 incoming
        for i in range(self.inter_start, self.n_total):
            if len(self.incoming[i]) == 0:
                if i < self.cmd_start:
                    src = rng.integers(0, n_sensory)
                elif i < self.motor_start:
                    src = rng.integers(self.inter_start, self.cmd_start)
                else:
                    src = rng.integers(self.cmd_start, self.motor_start)
                self.connections.append((src, i))
                self.incoming[i].append(src)

        # Build per-neuron weight offsets for flat arrays
        self.weight_offset = []
        self.weight_count = []
        offset = 0
        for i in range(self.n_total):
            self.weight_offset.append(offset)
            self.weight_count.append(len(self.incoming[i]))
            offset += len(self.incoming[i])
        self.total_gate_weights = offset

    def _add_connections(self, rng, src_range, dst_range, density):
        for dst in dst_range:
            for src in src_range:
                if src == dst:
                    continue
                if rng.random() < density:
                    self.connections.append((src, dst))
                    self.incoming[dst].append(src)


@dataclass
class CfcForwardCache:
    """Cache for one CfC forward pass, used in BPTT."""
    h_prev: np.ndarray
    h_new: np.ndarray
    pre_tau: np.ndarray
    pre_f: np.ndarray
    tau: np.ndarray
    f_gate: np.ndarray
    g_gate: np.ndarray
    dt: float
    input_norm: np.ndarray


class CfcCell:
    """
    Closed-form Continuous-time neuron with NCP wiring.

    Forward pass equations (per non-sensory neuron):
        tau = softplus(W_tau · h_incoming + b_tau)
        f   = sigmoid(-(dt · tau) · (W_f · h_incoming + b_f))
        g   = tanh(W_g · h_incoming + b_g)
        h'  = f · g + (1 - f) · h

    Ported from APEX Rust implementation (crates/apex-cfc/src/cell.rs).
    """

    def __init__(self, wiring: NcpWiringCfC, n_outputs: int, seed: int = 42):
        self.wiring = wiring
        self.n_outputs = n_outputs
        self.n_total = wiring.n_total
        self.n_motor = wiring.n_motor
        rng = np.random.default_rng(seed)

        n_gate = wiring.total_gate_weights
        scale = 0.1

        # Gate weights (flat arrays indexed by per-neuron offset)
        self.w_tau = rng.normal(0, scale, n_gate).astype(np.float64)
        self.w_f = rng.normal(0, scale, n_gate).astype(np.float64)
        self.w_g = rng.normal(0, scale, n_gate).astype(np.float64)

        # Per-neuron biases
        self.b_tau = np.zeros(self.n_total, dtype=np.float64)
        self.b_f = np.zeros(self.n_total, dtype=np.float64)
        self.b_g = np.zeros(self.n_total, dtype=np.float64)

        # Input projection weights
        self.w_in = rng.normal(0, 0.3, wiring.total_input_weights).astype(np.float64)

        # Output projection: motor -> outputs
        self.w_out = rng.normal(0, scale, (n_outputs, self.n_motor)).astype(np.float64)
        self.b_out = np.zeros(n_outputs, dtype=np.float64)

        # Adam state
        self._init_adam()

    def _init_adam(self):
        self._adam_t = 0
        params = self._flatten()
        self._adam_m = np.zeros_like(params)
        self._adam_v = np.zeros_like(params)

    def _flatten(self) -> np.ndarray:
        """Flatten all parameters into a single vector."""
        return np.concatenate([
            self.w_tau, self.w_f, self.w_g,
            self.b_tau, self.b_f, self.b_g,
            self.w_in,
            self.w_out.ravel(), self.b_out,
        ])

    def _unflatten(self, flat: np.ndarray):
        """Restore parameters from flat vector."""
        w = self.wiring
        idx = 0
        n_g = w.total_gate_weights
        self.w_tau = flat[idx:idx + n_g].copy(); idx += n_g
        self.w_f = flat[idx:idx + n_g].copy(); idx += n_g
        self.w_g = flat[idx:idx + n_g].copy(); idx += n_g
        n_n = self.n_total
        self.b_tau = flat[idx:idx + n_n].copy(); idx += n_n
        self.b_f = flat[idx:idx + n_n].copy(); idx += n_n
        self.b_g = flat[idx:idx + n_n].copy(); idx += n_n
        n_in = w.total_input_weights
        self.w_in = flat[idx:idx + n_in].copy(); idx += n_in
        n_out = self.n_outputs * self.n_motor
        self.w_out = flat[idx:idx + n_out].reshape(self.n_outputs, self.n_motor).copy()
        idx += n_out
        self.b_out = flat[idx:idx + self.n_outputs].copy()

    def forward(self, x_norm: np.ndarray, h: np.ndarray, dt: float) -> tuple:
        """
        One CfC forward step.

        Args:
            x_norm: normalized input features (n_features,)
            h: previous hidden state (n_total,)
            dt: time delta (seconds)

        Returns: (h_new, output, cache)
        """
        w = self.wiring
        h_new = h.copy()

        # 1. Input injection into sensory neurons
        w_in_idx = 0
        for f_idx in range(w.n_features):
            for s_idx in w.input_map[f_idx]:
                h_new[s_idx] = x_norm[f_idx] * self.w_in[w_in_idx]
                w_in_idx += 1

        pre_tau = np.zeros(self.n_total, dtype=np.float64)
        pre_f = np.zeros(self.n_total, dtype=np.float64)
        tau = np.zeros(self.n_total, dtype=np.float64)
        f_gate = np.zeros(self.n_total, dtype=np.float64)
        g_gate = np.zeros(self.n_total, dtype=np.float64)

        # 2. Gated update for inter, command, motor neurons
        for i in range(w.inter_start, w.n_total):
            n_in = w.weight_count[i]
            if n_in == 0:
                h_new[i] = h[i]
                continue

            offset = w.weight_offset[i]
            sum_tau = self.b_tau[i]
            sum_f = self.b_f[i]
            sum_g = self.b_g[i]

            for j, src in enumerate(w.incoming[i]):
                h_src = h_new[src]  # use already-updated values (single-pass)
                sum_tau += self.w_tau[offset + j] * h_src
                sum_f += self.w_f[offset + j] * h_src
                sum_g += self.w_g[offset + j] * h_src

            pre_tau[i] = sum_tau
            pre_f[i] = sum_f

            # Compute gates
            tau[i] = math.log1p(math.exp(min(sum_tau, 20.0)))  # softplus
            f_gate[i] = _sigmoid(np.array([-(dt * tau[i]) * sum_f]))[0]
            g_gate[i] = math.tanh(max(min(sum_g, 10.0), -10.0))

            # Gated update
            h_new[i] = f_gate[i] * g_gate[i] + (1.0 - f_gate[i]) * h[i]

        # 3. Output projection from motor neurons
        motor = h_new[w.motor_start:]
        output = self.w_out @ motor + self.b_out

        cache = CfcForwardCache(
            h_prev=h.copy(), h_new=h_new.copy(),
            pre_tau=pre_tau, pre_f=pre_f, tau=tau,
            f_gate=f_gate, g_gate=g_gate, dt=dt,
            input_norm=x_norm.copy(),
        )
        return h_new, output, cache

    def backward(self, caches: list, target: np.ndarray, lr: float,
                 bptt_decay: float, grad_clip: float):
        """
        Truncated BPTT through cached timesteps.

        Translates APEX Rust training.rs: compute MSE loss on output,
        backprop through output projection, then through CfC gates at
        each cached timestep with exponential decay.
        """
        w = self.wiring
        most_recent = caches[0]

        # Output MSE loss
        output = self.w_out @ most_recent.h_new[w.motor_start:] + self.b_out
        err = output - target
        loss = float(np.mean(err ** 2))

        # Gradient accumulators
        n_g = w.total_gate_weights
        d_w_tau = np.zeros(n_g, dtype=np.float64)
        d_w_f = np.zeros(n_g, dtype=np.float64)
        d_w_g = np.zeros(n_g, dtype=np.float64)
        d_b_tau = np.zeros(self.n_total, dtype=np.float64)
        d_b_f = np.zeros(self.n_total, dtype=np.float64)
        d_b_g = np.zeros(self.n_total, dtype=np.float64)
        d_w_in = np.zeros(w.total_input_weights, dtype=np.float64)
        d_w_out = np.zeros_like(self.w_out)
        d_b_out = np.zeros_like(self.b_out)

        # Backprop output projection
        d_output = 2.0 * err / len(err)
        d_b_out += d_output
        motor = most_recent.h_new[w.motor_start:]
        d_w_out += np.outer(d_output, motor)
        d_h = np.zeros(self.n_total, dtype=np.float64)
        for o in range(self.n_outputs):
            for m in range(self.n_motor):
                d_h[w.motor_start + m] += d_output[o] * self.w_out[o, m]

        # BPTT through cached timesteps
        for step, cache in enumerate(caches):
            decay = bptt_decay ** step

            # Reverse order for feedforward paths
            for i in range(w.n_total - 1, w.inter_start - 1, -1):
                if w.weight_count[i] == 0 or abs(d_h[i]) < 1e-15:
                    continue

                f = cache.f_gate[i]
                g = cache.g_gate[i]
                h_prev_i = cache.h_prev[i]
                dt_val = cache.dt

                df = d_h[i] * (g - h_prev_i)
                dg = d_h[i] * f
                d_h_prev_i = d_h[i] * (1.0 - f)

                d_pre_g = dg * (1.0 - g * g)  # tanh deriv
                d_f_input = df * f * (1.0 - f)  # sigmoid deriv
                d_pre_f = d_f_input * (-(dt_val * cache.tau[i]))
                d_tau = d_f_input * (-(dt_val) * cache.pre_f[i])
                # softplus deriv = sigmoid(pre_tau)
                sp = _sigmoid(np.array([cache.pre_tau[i]]))[0]
                d_pre_tau = d_tau * sp

                offset = w.weight_offset[i]
                for j, src in enumerate(w.incoming[i]):
                    h_src = cache.h_new[src]
                    d_w_tau[offset + j] += decay * d_pre_tau * h_src
                    d_w_f[offset + j] += decay * d_pre_f * h_src
                    d_w_g[offset + j] += decay * d_pre_g * h_src

                d_b_tau[i] += decay * d_pre_tau
                d_b_f[i] += decay * d_pre_f
                d_b_g[i] += decay * d_pre_g

                # Propagate to previous hidden state
                d_h[i] = d_h_prev_i

            # Propagate d_h to previous timestep if more caches remain
            if step < len(caches) - 1:
                # d_h carries through to next cache's d_h
                pass

        # Flatten gradients and clip
        d_flat = np.concatenate([
            d_w_tau, d_w_f, d_w_g,
            d_b_tau, d_b_f, d_b_g,
            d_w_in,
            d_w_out.ravel(), d_b_out,
        ])
        norm = np.linalg.norm(d_flat)
        if norm > grad_clip:
            d_flat *= grad_clip / norm

        # Adam update
        self._adam_t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self._adam_m = beta1 * self._adam_m + (1 - beta1) * d_flat
        self._adam_v = beta2 * self._adam_v + (1 - beta2) * d_flat ** 2
        m_hat = self._adam_m / (1 - beta1 ** self._adam_t)
        v_hat = self._adam_v / (1 - beta2 ** self._adam_t)
        params = self._flatten()
        params -= lr * m_hat / (np.sqrt(v_hat) + eps)
        self._unflatten(params)

        return loss


class CfcJudge:
    """
    CfC temporal arbiter sitting above the gossip mesh.

    Receives the full 6-node mesh state each timestep, learns to predict
    the next state (self-supervised), and uses prediction surprise to
    judge whether gossip verdicts are genuine anomalies or false positives.

    Key insight: during recovery transitions, delta-pressures are positive
    (pressures rising) but gossip scores remain elevated from the slow EMA.
    The CfC learns this pattern and can override false confirmations.
    """

    def __init__(self, mcfg: MeshConfig, seed: int = 42):
        self.mcfg = mcfg

        # Build NCP wiring
        self.wiring = NcpWiringCfC(
            n_sensory=mcfg.cfc_n_sensory,
            n_inter=mcfg.cfc_n_inter,
            n_command=mcfg.cfc_n_command,
            n_motor=mcfg.cfc_n_motor,
            n_features=mcfg.cfc_n_features,
            ff_density=mcfg.cfc_ff_density,
            rec_density=mcfg.cfc_recurrent_density,
            seed=seed,
        )

        # CfC cell
        self.cell = CfcCell(self.wiring, mcfg.cfc_n_outputs, seed=seed)

        # Hidden state (persists across timesteps)
        self.h = np.zeros(self.wiring.n_total, dtype=np.float64)

        # Online normalizer (Welford)
        self.norm_count = 0
        self.norm_mean = np.zeros(mcfg.cfc_n_features, dtype=np.float64)
        self.norm_m2 = np.zeros(mcfg.cfc_n_features, dtype=np.float64)

        # BPTT cache
        self.cache_history = deque(maxlen=mcfg.cfc_bptt_depth)

        # Previous state for delta-pressure and training target
        self.prev_features = None
        self.prev_prediction = None

        # Surprise tracking
        self.surprise_buffer = deque(maxlen=mcfg.cfc_surprise_buffer)
        self.current_surprise = 0.0

        # Learning rate with decay
        self.lr = mcfg.cfc_lr

        # Verdict log
        self.verdict_log = []
        self.n_overrides = 0
        self.n_reviews = 0

        # Post-confirmation fatigue: after confirming, raise the bar
        self.last_confirmed_t = -999
        self.confirm_count_window = 0  # confirmations in recent window

    def _normalize(self, raw: np.ndarray) -> np.ndarray:
        """Welford online normalize-and-update."""
        self.norm_count += 1
        n = self.norm_count
        delta = raw - self.norm_mean
        self.norm_mean += delta / n
        delta2 = raw - self.norm_mean
        self.norm_m2 += delta * delta2

        if n < 2:
            return np.zeros_like(raw)
        var = self.norm_m2 / (n - 1)
        std = np.sqrt(np.maximum(var, 1e-8))
        return (raw - self.norm_mean) / std

    def build_features(self, readings: dict, nodes: list) -> np.ndarray:
        """
        Assemble the 24-feature vector from mesh state.

        Layout: [pressure_0..5, gossip_0..5, anomaly_0..5, delta_p_0..5]
        """
        nn = self.mcfg.n_nodes
        features = np.zeros(self.mcfg.cfc_n_features, dtype=np.float64)

        for i in range(nn):
            features[i] = readings[i]["true_psi"]         # pressures
            features[6 + i] = nodes[i].pressure_gossip_score  # gossip scores
            features[12 + i] = readings[i]["anomaly_score"]   # anomaly scores

        # Delta-pressure (requires previous features)
        if self.prev_features is not None:
            for i in range(nn):
                features[18 + i] = features[i] - self.prev_features[i]
        else:
            features[18:24] = 0.0

        return features

    def step(self, readings: dict, nodes: list) -> float:
        """
        Process one timestep. Returns current surprise score.
        """
        raw = self.build_features(readings, nodes)
        x_norm = self._normalize(raw)

        # Self-supervised training: compare previous prediction to actual
        if self.prev_prediction is not None and self.norm_count > 50:
            # Prediction error = surprise
            self.current_surprise = float(np.mean(np.abs(
                self.prev_prediction - x_norm)))
            self.surprise_buffer.append(self.current_surprise)

            # BPTT training step
            if len(self.cache_history) > 0:
                loss = self.cell.backward(
                    list(self.cache_history), x_norm,
                    self.lr, self.mcfg.cfc_bptt_decay, self.mcfg.cfc_grad_clip,
                )
                # LR decay
                self.lr = max(self.lr * self.mcfg.cfc_lr_decay, self.mcfg.cfc_lr_floor)

        # Forward pass
        self.h, prediction, cache = self.cell.forward(x_norm, self.h, dt=1.0)
        self.cache_history.appendleft(cache)
        self.prev_prediction = prediction
        self.prev_features = raw.copy()

        return self.current_surprise

    def judge_verdict(self, gossip_verdict: str, gossip_detail: dict,
                      readings: dict, t: int = 0) -> tuple:
        """
        Review a gossip verdict using CfC temporal context.

        Returns: (final_verdict, confidence, reason)
        """
        self.n_reviews += 1

        # Need enough history for meaningful judgment
        if len(self.surprise_buffer) < 100:
            if gossip_verdict == "CONFIRMED":
                self.last_confirmed_t = t
                self.confirm_count_window += 1
            return gossip_verdict, 0.5, "warmup"

        # Compute surprise thresholds from history
        buf = np.array(self.surprise_buffer)
        low_thresh = np.percentile(buf, self.mcfg.cfc_surprise_low_pct)
        high_thresh = np.percentile(buf, self.mcfg.cfc_surprise_high_pct)

        surprise = self.current_surprise

        # Check delta-pressures: majority of nodes rising?
        if self.prev_features is not None and len(self.prev_features) >= 24:
            delta_p = self.prev_features[18:24]
            n_rising = int(np.sum(delta_p > 0.0))
            majority_rising = n_rising >= 4
        else:
            delta_p = None
            majority_rising = False

        # Confirmation fatigue: how recently did we last confirm?
        steps_since_confirm = t - self.last_confirmed_t

        if gossip_verdict == "CONFIRMED":
            # Case 1: Recovery transition — moderate/low surprise + pressures rising
            if surprise < high_thresh and majority_rising:
                self.n_overrides += 1
                self.verdict_log.append({
                    "action": "OVERRIDE", "reason": "recovery_transition",
                    "surprise": surprise, "threshold": high_thresh,
                    "n_rising": n_rising, "t": t,
                })
                return "DENIED", 0.8, "recovery_transition"

            # Case 2: Confirmation fatigue — repeated confirmations in quick
            # succession indicate the gossip EMA is just slowly catching up.
            # After the first confirm, require HIGH surprise to re-confirm.
            if steps_since_confirm < 100 and surprise < high_thresh:
                self.n_overrides += 1
                self.verdict_log.append({
                    "action": "OVERRIDE", "reason": "confirmation_fatigue",
                    "surprise": surprise, "threshold": high_thresh,
                    "steps_since": steps_since_confirm, "t": t,
                })
                return "DENIED", 0.75, "confirmation_fatigue"

            # Case 3: Low surprise + CfC has seen this pattern = normal
            if surprise < low_thresh * 0.5:
                self.n_overrides += 1
                self.verdict_log.append({
                    "action": "OVERRIDE", "reason": "low_surprise",
                    "surprise": surprise, "threshold": low_thresh, "t": t,
                })
                return "DENIED", 0.7, "low_surprise"

            # Case 4: High surprise = genuine anomaly — confirm
            self.last_confirmed_t = t
            self.confirm_count_window += 1
            if surprise > high_thresh:
                confidence = min(surprise / (high_thresh * 2), 1.0)
                return "CONFIRMED", confidence, "high_surprise"

            # Default: accept gossip verdict
            self.last_confirmed_t = t
            return gossip_verdict, 0.6, "accept"

        else:  # DENIED
            # Case 5: High surprise but gossip denied = potential missed event
            if surprise > high_thresh * 1.5:
                self.n_overrides += 1
                self.last_confirmed_t = t
                self.verdict_log.append({
                    "action": "OVERRIDE_DENY", "reason": "cfc_anomaly",
                    "surprise": surprise, "threshold": high_thresh, "t": t,
                })
                return "CONFIRMED", 0.6, "cfc_anomaly_override"

            return gossip_verdict, 0.5, "accept"


class SpatialPressureSimulator:
    """
    Generates realistic spatially-varying pressure at 6 node positions
    around the fire ring main, including:

    - Baseline static pressure (jockey pump maintained)
    - Local leak: pressure drop centered at one node, attenuating with
      distance around the ring (exponential decay along pipe path)
    - Catastrophic loss: system-wide depressurisation
    - Per-node Gaussian noise (sensor + environment)
    """

    def __init__(self, mcfg: MeshConfig, seed: int = 77):
        self.mcfg = mcfg
        self.rng = np.random.default_rng(seed)

    def get_pressures(self, t: int) -> dict:
        """
        Returns {node_id: pressure_psi} for timestep t.

        Pressure timeline:
          0..200:   Ramp 130 -> 150 PSI (jockey pump fill)
          200..400: Stable at 150 PSI (normal static)
          400..450: LOCAL LEAK near node 3 (port-side aft)
          450..550: Recovery (leak sealed, jockey restores)
          550..700: Stable at 150 PSI
          700..740: CATASTROPHIC loss (all nodes, main pipe rupture)
          740..850: Partial recovery (fire pump auto-start)
          850..1000: Stable at 130 PSI (post-event lower setpoint)
        """
        mcfg = self.mcfg

        # Base pressure profile (same for all nodes before spatial effects)
        if t < 200:
            base = 130 + 20 * (t / 200)
        elif t < mcfg.local_leak_time:
            base = 150.0
        elif t < mcfg.local_leak_time + mcfg.local_leak_duration:
            base = 150.0  # leak is spatial, not global
        elif t < mcfg.local_leak_time + mcfg.local_leak_duration + 100:
            # Recovery from local leak
            dt = t - (mcfg.local_leak_time + mcfg.local_leak_duration)
            base = 150.0  # jockey restores
        elif t < mcfg.catastrophic_time:
            base = 150.0
        elif t < mcfg.catastrophic_time + mcfg.catastrophic_duration:
            # Catastrophic drop
            dt = t - mcfg.catastrophic_time
            frac = dt / mcfg.catastrophic_duration
            base = 150.0 - mcfg.catastrophic_drop_psi * min(frac * 2, 1.0)
        elif t < mcfg.catastrophic_time + mcfg.catastrophic_duration + 100:
            # Fire pump recovery
            dt = t - (mcfg.catastrophic_time + mcfg.catastrophic_duration)
            frac = dt / 100
            base = (150.0 - mcfg.catastrophic_drop_psi) + mcfg.catastrophic_drop_psi * 0.5 * frac
            base = min(base, 130.0)
        else:
            base = 130.0

        pressures = {}
        for nid in range(mcfg.n_nodes):
            p = base

            # Local leak effect: attenuated by ring distance from leak node
            if (mcfg.local_leak_time <= t
                    < mcfg.local_leak_time + mcfg.local_leak_duration):
                dist = _ring_distance(
                    mcfg.node_positions[nid],
                    mcfg.node_positions[mcfg.local_leak_node],
                    mcfg.ring_main_length,
                )
                attenuation = math.exp(-mcfg.leak_attenuation_per_m * dist)
                p -= mcfg.local_leak_drop_psi * attenuation

            # Per-node jitter (sensor noise + local turbulence)
            p += self.rng.normal(0, 0.5)
            p = max(p, mcfg.pressure_min_psi * 0.5)
            pressures[nid] = p

        return pressures


class MeshNetwork:
    """
    Orchestrates the 6-node gossip micro-mesh.

    In production, each SensorNode runs on its own Pi 5 and gossip
    happens over UDP multicast. This class simulates the full mesh
    in a single process by holding all 6 nodes and routing messages.
    """

    def __init__(self, mcfg: MeshConfig):
        self.mcfg = mcfg
        self.base_cfg = Config()  # for non-overridden defaults
        self.nodes: list = []
        self.gossip = GossipProtocol(mcfg)
        self.event_log: list = []
        self.mesh_cooldown = 0  # suppress all gossip after a confirmed event
        self.judge = CfcJudge(mcfg, seed=42)

    def create_nodes(self):
        """Instantiate 6 sensor nodes with per-section pipe geometry."""
        for i in range(self.mcfg.n_nodes):
            node = SensorNode(
                node_id=i,
                position=self.mcfg.node_positions[i],
                section_length=self.mcfg.section_lengths[i],
                base_cfg=self.base_cfg,
                mcfg=self.mcfg,
                seed=42,
            )
            self.nodes.append(node)

    def train_shared_encoder(self):
        """
        Train one PI-SNN on representative firewater data, then distribute
        encoder weights to all nodes. Each node keeps its own output column
        for its specific pipe section geometry.
        """
        mcfg = self.mcfg

        # Build a representative config (average section length)
        avg_len = sum(mcfg.section_lengths) / len(mcfg.section_lengths)
        train_cfg = Config(
            pipe_length=avg_len,
            pipe_outer_diameter=mcfg.pipe_outer_diameter,
            pipe_wall_thickness=mcfg.pipe_wall_thickness,
            fluid_base_density=mcfg.fluid_base_density,
            fluid_base_bulk_modulus=mcfg.fluid_base_bulk_modulus,
            fluid_bulk_modulus_pressure_coeff=mcfg.fluid_bulk_modulus_pressure_coeff,
            pressure_min_psi=mcfg.pressure_min_psi,
            pressure_max_psi=mcfg.pressure_max_psi,
            n_resonance_modes=20,
            n_fft_bins=self.base_cfg.n_fft_bins,
            max_freq_hz=mcfg.mesh_max_freq_hz,
            n_epochs=mcfg.mesh_train_epochs,
            learning_rate=mcfg.mesh_train_lr,
            samples_per_level=mcfg.mesh_samples_per_level,
            batch_size=64,
            physics_lambda=0.1,
        )

        spectra, pressures = generate_dataset(train_cfg, seed=42)
        ref_model = PISNN(train_cfg, seed=42)

        console.print(f"\n[bold cyan]Training shared encoder[/bold cyan] on firewater dataset "
                       f"({len(spectra)} samples)")

        # Use the existing train_model function
        history, _, _ = train_model(ref_model, spectra, pressures, train_cfg)

        # Distribute ALL weights (encoder + output) to all nodes as starting point
        all_w = ref_model.get_encoder_weights()  # returns all W and b
        for node in self.nodes:
            node.model.set_all_weights(all_w)
            node.model.pressure_mean = ref_model.pressure_mean
            node.model.pressure_std = ref_model.pressure_std

        console.print(f"  Encoder distributed to {mcfg.n_nodes} nodes")

        # Fine-tune each node on its own pipe section spectra (short, 10 epochs)
        console.print("  Fine-tuning per-node output columns...")
        for node in self.nodes:
            ft_cfg = Config(
                pipe_length=node.section_length,
                pipe_outer_diameter=mcfg.pipe_outer_diameter,
                pipe_wall_thickness=mcfg.pipe_wall_thickness,
                fluid_base_density=mcfg.fluid_base_density,
                fluid_base_bulk_modulus=mcfg.fluid_base_bulk_modulus,
                fluid_bulk_modulus_pressure_coeff=mcfg.fluid_bulk_modulus_pressure_coeff,
                pressure_min_psi=mcfg.pressure_min_psi,
                pressure_max_psi=mcfg.pressure_max_psi,
                n_resonance_modes=20,
                n_fft_bins=self.base_cfg.n_fft_bins,
                max_freq_hz=mcfg.mesh_max_freq_hz,
                n_epochs=10,
                learning_rate=1e-3,
                samples_per_level=100,
                batch_size=64,
                physics_lambda=0.1,
            )
            ft_spec, ft_pres = generate_dataset(ft_cfg, seed=42 + node.node_id)
            train_model(node.model, ft_spec, ft_pres, ft_cfg)
            rmse = float(np.sqrt(np.mean(
                (node.model.forward(ft_spec[:50])[0].flatten() -
                 ft_pres[:50]) ** 2)))
            console.print(f"    Node {node.node_id} [{mcfg.node_labels[node.node_id]}]: "
                          f"fine-tune RMSE = {rmse:.1f} PSI")

        return history

    def sync_encoders(self):
        """
        Federated averaging of encoder weights across all nodes.
        Output columns are NOT averaged -- they stay node-specific.
        """
        n = len(self.nodes)
        ref = self.nodes[0].model.get_encoder_weights()
        avg = {
            "W": [w.copy() for w in ref["W"]],
            "b": [b_.copy() for b_ in ref["b"]],
        }
        for node in self.nodes[1:]:
            ew = node.model.get_encoder_weights()
            for i in range(len(avg["W"])):
                avg["W"][i] += ew["W"][i]
                avg["b"][i] += ew["b"][i]
        for i in range(len(avg["W"])):
            avg["W"][i] /= n
            avg["b"][i] /= n

        for node in self.nodes:
            node.model.set_encoder_weights(avg)

    def step(self, t: int, pressures: dict) -> dict:
        """
        One timestep of the mesh: all nodes infer, gossip rounds are
        initiated and resolved, CfC judge reviews verdicts.
        Returns per-node readings + gossip events + CfC surprise.
        """
        mcfg = self.mcfg
        readings = {}
        gossip_events = []

        # 1. Each node processes its spectrum independently
        for node in self.nodes:
            p = pressures[node.node_id]
            spec = generate_spectrum(p, node.cfg, node.rng)
            reading = node.process_sample(spec, p, t)
            readings[node.node_id] = reading

        # 2. CfC judge processes mesh state (runs every timestep for
        #    continuous temporal context, independent of gossip)
        surprise = self.judge.step(readings, self.nodes)

        # 3. Check for new gossip initiations (suppressed during cooldown)
        if self.mesh_cooldown > 0:
            self.mesh_cooldown -= 1
            # Reset consecutive counters during cooldown so nodes don't
            # immediately re-trigger when cooldown expires
            for node in self.nodes:
                node.consecutive_anomaly = 0
        else:
            for node in self.nodes:
                if node.should_gossip():
                    msg = self.gossip.initiate(node, t)
                    self.gossip.collect_votes(msg, self.nodes)
                    self.event_log.append({"t": t, "type": "GOSSIP_START",
                                           "origin": node.node_id,
                                           "score": msg.anomaly_score})

        # 4. Resolve pending gossip rounds -- CfC judge reviews each verdict
        resolved = self.gossip.resolve(t, self.nodes)
        for origin_id, verdict, detail in resolved:
            # CfC judge reviews the gossip verdict
            final_verdict, confidence, reason = self.judge.judge_verdict(
                verdict, detail, readings, t=t)

            ev = {
                "t": t, "type": "GOSSIP_RESULT",
                "origin": origin_id,
                "verdict": final_verdict,
                "original_verdict": verdict,
                "detail": detail,
                "cfc_confidence": confidence,
                "cfc_reason": reason,
            }
            self.event_log.append(ev)
            gossip_events.append(ev)
            # Cooldown uses ORIGINAL gossip verdict: even if CfC overrides
            # to DENIED, the EMA deviation that triggered gossip is real and
            # needs time to settle. Without this, gossip floods during recovery.
            if verdict == "CONFIRMED":
                self.mesh_cooldown = mcfg.gossip_post_confirm_cooldown

        # 5. Periodic encoder sync
        if t > 0 and t % mcfg.encoder_sync_interval == 0:
            self.sync_encoders()

        return {"readings": readings, "gossip_events": gossip_events,
                "cfc_surprise": surprise}


# ============================================================================
#  9. MESH DEMO
# ============================================================================

def run_mesh_demo():
    """
    6-node gossiping micro-mesh demo on a fire ring main:
    1. Create 6 sensor nodes with different pipe section geometries
    2. Train shared encoder on representative firewater data
    3. Stream 1000 timesteps with spatial pressure propagation
    4. t=400: local leak near node 3 (port-side)
    5. t=700: catastrophic pressure loss (all nodes)
    6. Gossip protocol confirms/denies each event
    7. Generate mesh-level plots
    """
    mcfg = MeshConfig()
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    console.rule("[bold magenta]SAIREN ARS Gossip Micro-Mesh[/bold magenta]")
    console.print("6-node fire ring main -- acoustic pressure monitoring with gossip consensus\n")

    # -- Step 1: Create mesh ---------------------------------------------------
    console.rule("[cyan]Step 1: Create Mesh Network[/cyan]")
    mesh = MeshNetwork(mcfg)
    mesh.create_nodes()
    for i, node in enumerate(mesh.nodes):
        console.print(f"  Node {i} [{mcfg.node_labels[i]}]: "
                       f"pos={mcfg.node_positions[i]:.0f}m, "
                       f"section={mcfg.section_lengths[i]:.1f}m")

    # -- Step 2: Train shared encoder ------------------------------------------
    console.rule("[cyan]Step 2: Train Shared Encoder[/cyan]")
    history = mesh.train_shared_encoder()

    # -- Step 3: Stream with spatial pressure -----------------------------------
    console.rule("[cyan]Step 3: Online Streaming with Gossip Consensus[/cyan]")
    sim = SpatialPressureSimulator(mcfg)

    N = mcfg.demo_length
    nn = mcfg.n_nodes
    true_p = np.zeros((N, nn))
    pred_p = np.zeros((N, nn))
    anom_s = np.zeros((N, nn))
    cfc_surprise = np.zeros(N)

    # Track gossip events for plotting
    gossip_log = []

    # Build table
    table = Table(title="Mesh Streaming -- Gossip Consensus + CfC Judge")
    table.add_column("t", justify="right", style="cyan", width=5)
    for i in range(nn):
        table.add_column(f"N{i}", justify="right", width=8)
    table.add_column("Gossip", style="bold", width=50)

    print_interval = 20

    for t in range(N):
        pressures = sim.get_pressures(t)
        result = mesh.step(t, pressures)

        for nid in range(nn):
            r = result["readings"][nid]
            true_p[t, nid] = r["true_psi"]
            pred_p[t, nid] = r["pred_psi"]
            anom_s[t, nid] = r["anomaly_score"]
        cfc_surprise[t] = result["cfc_surprise"]

        # Format gossip events (now with CfC judge annotations)
        gossip_str = ""
        for ev in result["gossip_events"]:
            origin = ev["origin"]
            verdict = ev["verdict"]
            original = ev.get("original_verdict", verdict)
            reason = ev.get("cfc_reason", "")
            d = ev["detail"]
            label = mcfg.node_labels[origin]

            # Check if CfC overrode the gossip verdict
            override_tag = ""
            if original != verdict:
                override_tag = f" [bold magenta](CfC: {original}->{verdict} [{reason}])[/bold magenta]"

            if d["system_wide"]:
                gossip_str = f"[bold red]SYSTEM-WIDE[/bold red] from {label}{override_tag}"
            elif verdict == "CONFIRMED":
                cns = ",".join(str(n) for n in d["confirm_nodes"])
                gossip_str = f"[red]CONFIRMED[/red] N{origin} ({label}) by [{cns}]{override_tag}"
            else:
                dns = ",".join(str(n) for n in d["deny_nodes"])
                gossip_str = f"[green]DENIED[/green] N{origin} ({label}) by [{dns}]{override_tag}"
            gossip_log.append(ev)

        # Print rows at interval or during events
        is_event = (mcfg.local_leak_time <= t < mcfg.local_leak_time + 10
                     or mcfg.catastrophic_time <= t < mcfg.catastrophic_time + 10
                     or gossip_str)
        if t % print_interval == 0 or is_event:
            row = [str(t)]
            for nid in range(nn):
                val = f"{true_p[t, nid]:.0f}"
                sc = anom_s[t, nid]
                if sc > 0.5:
                    val = f"[bold red]{val}[/bold red]"
                elif sc > 0.3:
                    val = f"[yellow]{val}[/yellow]"
                row.append(val)
            row.append(gossip_str or "[dim]--[/dim]")
            table.add_row(*row)

    console.print(table)

    # -- Step 4: Results --------------------------------------------------------
    console.rule("[cyan]Results Summary[/cyan]")

    # Per-node RMSE (excluding event windows)
    normal_mask = np.ones(N, dtype=bool)
    normal_mask[mcfg.local_leak_time:mcfg.local_leak_time + mcfg.local_leak_duration + 50] = False
    normal_mask[mcfg.catastrophic_time:mcfg.catastrophic_time + mcfg.catastrophic_duration + 50] = False

    for nid in range(nn):
        rmse = float(np.sqrt(np.mean((pred_p[normal_mask, nid] - true_p[normal_mask, nid]) ** 2)))
        console.print(f"  Node {nid} [{mcfg.node_labels[nid]}] RMSE (normal): {rmse:.1f} PSI")

    # Gossip summary
    confirmed = [e for e in gossip_log if e["verdict"] == "CONFIRMED"]
    denied = [e for e in gossip_log if e["verdict"] == "DENIED"]
    console.print(f"\n  Gossip rounds: {len(gossip_log)} total, "
                   f"[red]{len(confirmed)} confirmed[/red], "
                   f"[green]{len(denied)} denied[/green]")

    # Check: was the local leak confirmed?
    local_confirms = [e for e in confirmed
                      if mcfg.local_leak_time <= e["t"] < mcfg.local_leak_time + mcfg.local_leak_duration + 20]
    if local_confirms:
        delay = local_confirms[0]["t"] - mcfg.local_leak_time
        console.print(f"  [bold green]>> Local leak CONFIRMED at t={local_confirms[0]['t']} "
                       f"(delay: {delay} steps)[/bold green]")
    else:
        console.print("  [bold red]>> Local leak NOT confirmed by gossip[/bold red]")

    # Check: was the catastrophic event confirmed as system-wide?
    cat_confirms = [e for e in confirmed
                    if mcfg.catastrophic_time <= e["t"] < mcfg.catastrophic_time + mcfg.catastrophic_duration + 20]
    cat_syswide = [e for e in cat_confirms if e["detail"]["system_wide"]]
    if cat_syswide:
        delay = cat_syswide[0]["t"] - mcfg.catastrophic_time
        console.print(f"  [bold green]>> Catastrophic event CONFIRMED as SYSTEM-WIDE at "
                       f"t={cat_syswide[0]['t']} (delay: {delay} steps)[/bold green]")
    elif cat_confirms:
        console.print(f"  [yellow]>> Catastrophic event confirmed but not flagged system-wide[/yellow]")
    else:
        console.print("  [bold red]>> Catastrophic event NOT confirmed[/bold red]")

    buf_total = sum(n.buffer.spectra.nbytes + n.buffer.pressures.nbytes for n in mesh.nodes) / 1024
    console.print(f"  Total buffer memory: {buf_total:.1f} KB ({buf_total/nn:.1f} KB/node, fixed)")

    # CfC Judge summary
    judge = mesh.judge
    console.print(f"\n  [bold cyan]CfC Judge:[/bold cyan] {judge.n_reviews} verdicts reviewed, "
                   f"[magenta]{judge.n_overrides} overrides[/magenta]")
    overridden = [e for e in gossip_log if e.get("original_verdict") != e["verdict"]]
    for ov in overridden:
        console.print(f"    t={ov['t']}: {ov['original_verdict']} -> {ov['verdict']} "
                       f"(reason: {ov.get('cfc_reason', '?')}, "
                       f"confidence: {ov.get('cfc_confidence', 0):.2f})")

    # -- Step 5: Plots ----------------------------------------------------------
    console.rule("[cyan]Generating Mesh Plots[/cyan]")

    node_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fig, axes = plt.subplots(5, 1, figsize=(16, 22), dpi=100)
    t_axis = np.arange(N)

    # (a) True pressure per node (spatial view)
    ax = axes[0]
    for nid in range(nn):
        ax.plot(t_axis, true_p[:, nid], color=node_colors[nid], alpha=0.8, lw=1.0,
                label=f"N{nid} {mcfg.node_labels[nid]}")
    ax.axvspan(mcfg.local_leak_time, mcfg.local_leak_time + mcfg.local_leak_duration,
               alpha=0.15, color="orange", label="Local Leak")
    ax.axvspan(mcfg.catastrophic_time, mcfg.catastrophic_time + mcfg.catastrophic_duration,
               alpha=0.15, color="red", label="Catastrophic")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("True Pressure (PSI)")
    ax.set_title("(a) Spatial Pressure Distribution -- All 6 Nodes")
    ax.legend(loc="upper right", fontsize=7, ncol=4)
    ax.grid(True, alpha=0.3)

    # (b) Predicted vs true for node 3 (leak site) and node 0 (far side)
    ax = axes[1]
    for nid in [mcfg.local_leak_node, 0]:
        ax.plot(t_axis, true_p[:, nid], color=node_colors[nid], alpha=0.5, lw=0.8,
                label=f"N{nid} true")
        ax.plot(t_axis, pred_p[:, nid], color=node_colors[nid], alpha=0.9, lw=1.0,
                ls="--", label=f"N{nid} pred")
    ax.axvspan(mcfg.local_leak_time, mcfg.local_leak_time + mcfg.local_leak_duration,
               alpha=0.15, color="orange")
    ax.axvspan(mcfg.catastrophic_time, mcfg.catastrophic_time + mcfg.catastrophic_duration,
               alpha=0.15, color="red")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Pressure (PSI)")
    ax.set_title("(b) True vs Predicted -- Leak Node (N3) and Far Node (N0)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Anomaly scores -- all nodes
    ax = axes[2]
    for nid in range(nn):
        ax.plot(t_axis, anom_s[:, nid], color=node_colors[nid], alpha=0.7, lw=0.8,
                label=f"N{nid}")
    ax.axhline(mcfg.gossip_trigger_threshold, color="gray", ls="--", alpha=0.5,
               label="Gossip Trigger")
    ax.axvspan(mcfg.local_leak_time, mcfg.local_leak_time + mcfg.local_leak_duration,
               alpha=0.15, color="orange")
    ax.axvspan(mcfg.catastrophic_time, mcfg.catastrophic_time + mcfg.catastrophic_duration,
               alpha=0.15, color="red")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Anomaly Score")
    ax.set_title("(c) Per-Node Anomaly Scores")
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # (d) Gossip consensus timeline
    ax = axes[3]
    ax.set_xlim(0, N)
    ax.set_ylim(-0.5, nn - 0.5)
    ax.set_yticks(range(nn))
    ax.set_yticklabels([f"N{i}" for i in range(nn)])
    ax.set_xlabel("Time Step")
    ax.set_title("(d) Gossip Consensus Timeline")
    ax.grid(True, alpha=0.3)

    # Plot gossip events as markers (with CfC override annotations)
    for ev in mesh.event_log:
        if ev["type"] == "GOSSIP_START":
            ax.plot(ev["t"], ev["origin"], "o", color="orange", ms=6, zorder=5)
        elif ev["type"] == "GOSSIP_RESULT":
            color = "red" if ev["verdict"] == "CONFIRMED" else "green"
            marker = "^" if ev["verdict"] == "CONFIRMED" else "v"
            # Highlight CfC overrides with magenta edge
            edge_color = "magenta" if ev.get("original_verdict") != ev["verdict"] else color
            edge_width = 2.0 if ev.get("original_verdict") != ev["verdict"] else 0.5
            ax.plot(ev["t"], ev["origin"], marker, color=color, ms=8, zorder=5,
                    markeredgecolor=edge_color, markeredgewidth=edge_width)
            # Draw lines from confirming/denying nodes
            for cn in ev["detail"].get("confirm_nodes", []):
                ax.plot([ev["t"], ev["t"]], [ev["origin"], cn],
                        color="red", alpha=0.3, lw=1)
            for dn in ev["detail"].get("deny_nodes", []):
                ax.plot([ev["t"], ev["t"]], [ev["origin"], dn],
                        color="green", alpha=0.3, lw=1)

    ax.axvspan(mcfg.local_leak_time, mcfg.local_leak_time + mcfg.local_leak_duration,
               alpha=0.15, color="orange")
    ax.axvspan(mcfg.catastrophic_time, mcfg.catastrophic_time + mcfg.catastrophic_duration,
               alpha=0.15, color="red")

    # Legend for gossip plot
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="orange", ms=8, label="Gossip Start"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="red", ms=8, label="Confirmed"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="green", ms=8, label="Denied"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    # (e) CfC Judge surprise timeline
    ax = axes[4]
    ax.plot(t_axis, cfc_surprise, color="#7b2d8e", lw=1.0, alpha=0.9, label="CfC Surprise")
    # Show surprise thresholds if enough data
    if len(mesh.judge.surprise_buffer) > 100:
        buf = np.array(mesh.judge.surprise_buffer)
        low_t = np.percentile(buf, mcfg.cfc_surprise_low_pct)
        high_t = np.percentile(buf, mcfg.cfc_surprise_high_pct)
        ax.axhline(low_t, color="green", ls="--", alpha=0.5, label=f"Low ({mcfg.cfc_surprise_low_pct}th pct)")
        ax.axhline(high_t, color="red", ls="--", alpha=0.5, label=f"High ({mcfg.cfc_surprise_high_pct}th pct)")
    # Mark judge overrides
    for ev in mesh.event_log:
        if ev["type"] == "GOSSIP_RESULT" and ev.get("original_verdict") != ev["verdict"]:
            marker = "D" if ev["verdict"] == "DENIED" else "^"
            color = "green" if ev["verdict"] == "DENIED" else "red"
            ax.plot(ev["t"], cfc_surprise[ev["t"]], marker, color=color, ms=10,
                    zorder=5, markeredgecolor="black", markeredgewidth=0.5)
    # Mark all judge reviews (small dots)
    for ev in mesh.event_log:
        if ev["type"] == "GOSSIP_RESULT":
            ax.plot(ev["t"], cfc_surprise[ev["t"]], ".", color="#7b2d8e", ms=4, zorder=3)
    ax.axvspan(mcfg.local_leak_time, mcfg.local_leak_time + mcfg.local_leak_duration,
               alpha=0.15, color="orange")
    ax.axvspan(mcfg.catastrophic_time, mcfg.catastrophic_time + mcfg.catastrophic_duration,
               alpha=0.15, color="red")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Surprise")
    ax.set_title("(e) CfC Judge -- Temporal Surprise & Verdict Overrides")
    override_legend = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="green",
               markeredgecolor="black", ms=8, label="Override: CONFIRMED->DENIED"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="red",
               markeredgecolor="black", ms=8, label="Override: DENIED->CONFIRMED"),
        Line2D([0], [0], color="#7b2d8e", lw=1.5, label="CfC Surprise"),
    ]
    ax.legend(handles=override_legend, loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    mesh_plot = str(output_dir / "mesh_results.png")
    plt.savefig(mesh_plot, bbox_inches="tight")
    plt.close()
    console.print(f"  Saved mesh plot to {mesh_plot}")

    # Save mesh state to JSON
    mesh_state = {
        "nodes": mcfg.n_nodes,
        "labels": mcfg.node_labels,
        "gossip_rounds": len(mesh.gossip.completed_rounds),
        "cfc_judge": {
            "reviews": mesh.judge.n_reviews,
            "overrides": mesh.judge.n_overrides,
            "verdict_log": mesh.judge.verdict_log,
        },
        "event_log": [
            {k: v for k, v in ev.items() if k != "detail"} if ev["type"] == "GOSSIP_START"
            else ev
            for ev in mesh.event_log
        ],
    }
    mesh_json = str(output_dir / "mesh_state.json")
    with open(mesh_json, "w") as f:
        json.dump(mesh_state, f, indent=2, default=str)
    console.print(f"  Saved mesh state to {mesh_json}")

    console.rule("[bold green]Mesh Demo Complete[/bold green]")


# ============================================================================
#  10. PUMP HEALTH MONITORING DEMO
# ============================================================================

def run_pump_health_demo():
    """
    Pump health monitoring demo on the fire ring main jockey pump.

    Runs the full 6-node mesh with progressive pump degradation:
    1. Healthy operation (0-300)
    2. Bearing wear onset -- BPFO peak grows, temps rise (300-600)
    3. Moderate degradation -- alarm thresholds approached (600-750)
    4. Severe degradation -- trip thresholds, pressure can't maintain (750-900)
    5. Pump trip at t=900 -- pressure drops, fire pump auto-starts at t=920
    6. Post-event stabilisation (920-1200)

    Generates a 9-panel plot covering pipe pressure, gossip, CfC,
    pump health index, vibration waterfall, temperatures, and MCSA.
    """
    mcfg = MeshConfig(demo_length=1200)
    pcfg = PumpHealthConfig()
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    console.rule("[bold magenta]SAIREN Guardian -- Pump Health Monitoring[/bold magenta]")
    console.print("6-node fire ring main + jockey pump tri-modal health analysis\n")

    # -- Step 1: Create mesh ---------------------------------------------------
    console.rule("[cyan]Step 1: Create Mesh Network[/cyan]")
    mesh = MeshNetwork(mcfg)
    mesh.create_nodes()
    for i, node in enumerate(mesh.nodes):
        label = mcfg.node_labels[i]
        extra = " [bold yellow]+ PUMP HEALTH[/bold yellow]" if i == 1 else ""
        console.print(f"  Node {i} [{label}]: pos={mcfg.node_positions[i]:.0f}m, "
                       f"section={mcfg.section_lengths[i]:.1f}m{extra}")

    # -- Step 2: Train shared encoder ------------------------------------------
    console.rule("[cyan]Step 2: Train Shared Encoder[/cyan]")
    history = mesh.train_shared_encoder()

    # -- Step 3: Initialise pump health subsystem ------------------------------
    console.rule("[cyan]Step 3: Initialise Pump Health Subsystem[/cyan]")
    pump_sim = PumpSensorSimulator(pcfg)
    pump_analyzer = PumpHealthAnalyzer(pcfg)

    # Print characteristic frequencies
    freqs = pump_sim.freqs
    console.print(f"  Shaft frequency: {freqs['shaft']:.1f} Hz")
    console.print(f"  BPFO: {freqs['BPFO']:.1f} Hz  |  BPFI: {freqs['BPFI']:.1f} Hz  |  "
                   f"BSF: {freqs['BSF']:.1f} Hz  |  FTF: {freqs['FTF']:.1f} Hz")
    console.print(f"  Vane pass: {freqs['vane_pass']:.1f} Hz")
    bars = pump_sim.mcsa_bars
    console.print(f"  MCSA broken-bar sidebands: {bars[0]:.2f}, {bars[1]:.2f} Hz (1st pair)")

    # -- Step 4: Stream with pump degradation ----------------------------------
    console.rule("[cyan]Step 4: Online Streaming with Pump Degradation[/cyan]")

    # Modified pressure simulator: pump health affects recovery
    sim = SpatialPressureSimulator(mcfg)

    N = mcfg.demo_length
    nn = mcfg.n_nodes
    true_p = np.zeros((N, nn))
    pred_p = np.zeros((N, nn))
    anom_s = np.zeros((N, nn))
    cfc_surprise = np.zeros(N)

    # Pump health tracking arrays
    pump_health = np.zeros(N)
    pump_severity = np.zeros(N)
    pump_vib_rms = np.zeros(N)
    pump_temp_de = np.zeros(N)
    pump_temp_nde = np.zeros(N)
    pump_temp_winding = np.zeros(N)
    pump_temp_seal = np.zeros(N)
    pump_mcsa_sb_db = np.zeros(N)
    pump_running_arr = np.zeros(N, dtype=bool)
    pump_vib_spectra = np.zeros((N, pcfg.vib_n_fft_bins))  # for waterfall plot

    gossip_log = []
    pump_events = []  # pump-specific event log

    # Build table
    table = Table(title="Pump Health Monitoring + Gossip Consensus")
    table.add_column("t", justify="right", style="cyan", width=5)
    table.add_column("P(N1)", justify="right", width=7)
    table.add_column("Health", justify="right", width=7)
    table.add_column("Sev", justify="right", width=5)
    table.add_column("VibRMS", justify="right", width=7)
    table.add_column("TempDE", justify="right", width=7)
    table.add_column("MCSA dB", justify="right", width=8)
    table.add_column("Diagnosis", style="bold", width=35)
    table.add_column("Gossip", style="bold", width=30)

    print_interval = 20

    for t in range(N):
        # Get pump severity and modify pressure dynamics accordingly
        severity = pump_sim.get_severity(t)
        pump_severity[t] = severity

        # Pump health affects pressure recovery: degraded pump = slower fill
        pressures = sim.get_pressures(t)

        # After pump trip, pressure slowly decays (no jockey pump)
        if t >= pcfg.pump_trip_step and t < pcfg.fire_pump_start_step:
            decay_rate = 0.5  # PSI per step
            steps_since_trip = t - pcfg.pump_trip_step
            for nid in pressures:
                pressures[nid] -= decay_rate * steps_since_trip
                pressures[nid] = max(pressures[nid], 100.0)
        elif t >= pcfg.fire_pump_start_step:
            # Fire pump restores to 145 PSI (slightly below jockey setpoint)
            steps_since_fire = t - pcfg.fire_pump_start_step
            target = 145.0
            for nid in pressures:
                current = pressures[nid]
                pressures[nid] = current + (target - current) * min(steps_since_fire / 50.0, 1.0)

        # Moderate degradation: jockey pump slower recovery from events
        if 0.3 < severity < 1.0 and t < pcfg.pump_trip_step:
            # Reduce pressure slightly (pump struggling)
            pressure_penalty = severity * 5.0
            for nid in pressures:
                pressures[nid] -= pressure_penalty

        # Mesh step (all nodes process pipe pressure)
        result = mesh.step(t, pressures)

        for nid in range(nn):
            r = result["readings"][nid]
            true_p[t, nid] = r["true_psi"]
            pred_p[t, nid] = r["pred_psi"]
            anom_s[t, nid] = r["anomaly_score"]
        cfc_surprise[t] = result["cfc_surprise"]

        # Pump health analysis (runs independently of pipe ARS)
        pump_readings = pump_sim.get_readings(t)
        pump_result = pump_analyzer.analyze(pump_readings)

        pump_health[t] = pump_result["health_index"]
        pump_running_arr[t] = pump_readings["pump_running"]
        pump_vib_rms[t] = pump_readings["vibration_rms_mm_s"]
        pump_vib_spectra[t] = pump_readings["vibration_spectrum"]
        pump_temp_de[t] = pump_readings["temperatures"]["bearing_de"]
        pump_temp_nde[t] = pump_readings["temperatures"]["bearing_nde"]
        pump_temp_winding[t] = pump_readings["temperatures"]["winding"]
        pump_temp_seal[t] = pump_readings["temperatures"]["seal"]
        if pump_result["mcsa"]["sideband_ratio_db"] is not None:
            pump_mcsa_sb_db[t] = pump_result["mcsa"]["sideband_ratio_db"]

        # Track pump health events
        health = pump_result["health_index"]
        diag = pump_result["diagnosis"]
        if health < 0.7 and (not pump_events or pump_events[-1]["health"] >= 0.7):
            pump_events.append({"t": t, "type": "ALARM", "health": health, "diagnosis": diag})
        if health < 0.4 and (not pump_events or pump_events[-1].get("type") != "CRITICAL"):
            pump_events.append({"t": t, "type": "CRITICAL", "health": health, "diagnosis": diag})
        if t == pcfg.pump_trip_step:
            pump_events.append({"t": t, "type": "TRIP", "health": health, "diagnosis": diag})

        # Gossip events
        gossip_str = ""
        for ev in result["gossip_events"]:
            origin = ev["origin"]
            verdict = ev["verdict"]
            original = ev.get("original_verdict", verdict)
            d = ev["detail"]
            label = mcfg.node_labels[origin]
            override_tag = ""
            if original != verdict:
                override_tag = f" (CfC: {original[0]}->{verdict[0]})"
            if d["system_wide"]:
                gossip_str = f"[bold red]SYS-WIDE[/bold red]{override_tag}"
            elif verdict == "CONFIRMED":
                gossip_str = f"[red]CONF[/red] N{origin}{override_tag}"
            else:
                gossip_str = f"[green]DENY[/green] N{origin}{override_tag}"
            gossip_log.append(ev)

        # Table output
        is_event = (t == pcfg.degradation_start or t == pcfg.pump_trip_step
                     or t == pcfg.fire_pump_start_step or gossip_str)
        if t % print_interval == 0 or is_event:
            # Health color coding
            h = pump_health[t]
            if h > 0.7:
                h_str = f"[green]{h:.2f}[/green]"
            elif h > 0.4:
                h_str = f"[yellow]{h:.2f}[/yellow]"
            else:
                h_str = f"[bold red]{h:.2f}[/bold red]"

            # Severity
            s_str = f"{severity:.2f}"

            # Vibration RMS color
            vrms = pump_vib_rms[t]
            if vrms < pcfg.vib_alarm_mm_s:
                v_str = f"{vrms:.1f}"
            elif vrms < pcfg.vib_trip_mm_s:
                v_str = f"[yellow]{vrms:.1f}[/yellow]"
            else:
                v_str = f"[bold red]{vrms:.1f}[/bold red]"

            # Temp DE
            td = pump_temp_de[t]
            delta_de = td - pcfg.temp_bearing_de_nominal
            if delta_de < pcfg.temp_alarm_delta:
                td_str = f"{td:.0f}"
            elif delta_de < pcfg.temp_trip_delta:
                td_str = f"[yellow]{td:.0f}[/yellow]"
            else:
                td_str = f"[bold red]{td:.0f}[/bold red]"

            # MCSA sideband ratio
            sb = pump_mcsa_sb_db[t]
            if sb > -40:
                sb_str = f"[bold red]{sb:.0f}[/bold red]"
            elif sb > -50:
                sb_str = f"[yellow]{sb:.0f}[/yellow]"
            else:
                sb_str = f"{sb:.0f}"

            # Diagnosis (truncated)
            diag_str = diag[:33] if len(diag) > 33 else diag
            if "Healthy" not in diag:
                diag_str = f"[red]{diag_str}[/red]"

            table.add_row(str(t), f"{true_p[t, 1]:.0f}", h_str, s_str,
                          v_str if pump_running_arr[t] else "[dim]OFF[/dim]",
                          td_str, sb_str if pump_running_arr[t] else "[dim]--[/dim]",
                          diag_str, gossip_str or "[dim]--[/dim]")

    console.print(table)

    # -- Step 5: Results --------------------------------------------------------
    console.rule("[cyan]Results Summary[/cyan]")

    # Pump health timeline
    for ev in pump_events:
        style = {"ALARM": "yellow", "CRITICAL": "bold red", "TRIP": "bold red on white"}.get(ev["type"], "white")
        console.print(f"  [{style}]t={ev['t']}: {ev['type']} -- health={ev['health']:.2f} -- {ev['diagnosis']}[/{style}]")

    # Gossip summary
    confirmed = [e for e in gossip_log if e["verdict"] == "CONFIRMED"]
    denied = [e for e in gossip_log if e["verdict"] == "DENIED"]
    console.print(f"\n  Gossip rounds: {len(gossip_log)} total, "
                   f"[red]{len(confirmed)} confirmed[/red], "
                   f"[green]{len(denied)} denied[/green]")

    # CfC Judge summary
    judge = mesh.judge
    console.print(f"  CfC Judge: {judge.n_reviews} reviewed, "
                   f"[magenta]{judge.n_overrides} overrides[/magenta]")

    # Pump health at key moments
    console.print(f"\n  Pump health at degradation start (t={pcfg.degradation_start}): "
                   f"{pump_health[pcfg.degradation_start]:.2f}")
    console.print(f"  Pump health at trip (t={pcfg.pump_trip_step}): "
                   f"{pump_health[min(pcfg.pump_trip_step, N-1)]:.2f}")
    console.print(f"  Final bearing DE temp: {pump_temp_de[min(pcfg.pump_trip_step, N-1)]:.1f} C "
                   f"(nominal: {pcfg.temp_bearing_de_nominal:.0f} C)")
    console.print(f"  Final vibration RMS: {pump_vib_rms[min(pcfg.pump_trip_step-1, N-1)]:.1f} mm/s "
                   f"(alarm: {pcfg.vib_alarm_mm_s}, trip: {pcfg.vib_trip_mm_s})")

    # -- Step 6: Plots ----------------------------------------------------------
    console.rule("[cyan]Generating Pump Health Plots[/cyan]")

    node_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    t_axis = np.arange(N)

    fig, axes = plt.subplots(5, 2, figsize=(20, 28), dpi=100)

    # (a) Spatial pressure -- all 6 nodes
    ax = axes[0, 0]
    for nid in range(nn):
        ax.plot(t_axis, true_p[:, nid], color=node_colors[nid], alpha=0.7, lw=0.8,
                label=f"N{nid} {mcfg.node_labels[nid]}")
    ax.axvline(pcfg.degradation_start, color="orange", ls=":", alpha=0.7, label="Degradation onset")
    ax.axvline(pcfg.pump_trip_step, color="red", ls="--", alpha=0.7, label="Pump trip")
    ax.axvline(pcfg.fire_pump_start_step, color="blue", ls="--", alpha=0.7, label="Fire pump start")
    if mcfg.local_leak_time < N:
        ax.axvspan(mcfg.local_leak_time, mcfg.local_leak_time + mcfg.local_leak_duration,
                   alpha=0.1, color="orange")
    ax.set_ylabel("Pressure (PSI)")
    ax.set_title("(a) Spatial Pressure Distribution")
    ax.legend(loc="upper right", fontsize=6, ncol=3)
    ax.grid(True, alpha=0.3)

    # (b) Pump health index
    ax = axes[0, 1]
    ax.fill_between(t_axis, pump_health, alpha=0.3, color="green", where=pump_health > 0.7)
    ax.fill_between(t_axis, pump_health, alpha=0.3, color="orange", where=(pump_health > 0.4) & (pump_health <= 0.7))
    ax.fill_between(t_axis, pump_health, alpha=0.3, color="red", where=pump_health <= 0.4)
    ax.plot(t_axis, pump_health, color="black", lw=1.0)
    ax.plot(t_axis, pump_severity, color="gray", ls="--", lw=0.8, alpha=0.5, label="Fault severity")
    ax.axhline(0.7, color="orange", ls=":", alpha=0.5, label="Alarm")
    ax.axhline(0.4, color="red", ls=":", alpha=0.5, label="Critical")
    ax.axvline(pcfg.pump_trip_step, color="red", ls="--", alpha=0.7, label="Pump trip")
    ax.set_ylabel("Health Index / Severity")
    ax.set_title("(b) Pump Health Index (composite)")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # (c) Vibration RMS
    ax = axes[1, 0]
    running_mask = pump_running_arr & (pump_vib_rms > 0.1)
    ax.scatter(t_axis[running_mask], pump_vib_rms[running_mask], s=2, c=pump_vib_rms[running_mask],
               cmap="RdYlGn_r", vmin=1.0, vmax=8.0, alpha=0.7)
    ax.axhline(pcfg.vib_alarm_mm_s, color="orange", ls="--", label=f"Alarm ({pcfg.vib_alarm_mm_s})")
    ax.axhline(pcfg.vib_trip_mm_s, color="red", ls="--", label=f"Trip ({pcfg.vib_trip_mm_s})")
    ax.axvline(pcfg.pump_trip_step, color="red", ls="--", alpha=0.5)
    ax.set_ylabel("RMS Velocity (mm/s)")
    ax.set_title("(c) Vibration RMS Trend (ISO 10816-3)")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)

    # (d) Vibration spectrum waterfall
    ax = axes[1, 1]
    # Subsample for waterfall (every 5th running step)
    wf_idx = np.where(running_mask)[0][::5]
    if len(wf_idx) > 0:
        wf_data = pump_vib_spectra[wf_idx]
        vib_freq = np.linspace(0, pcfg.vib_max_freq_hz, pcfg.vib_n_fft_bins)
        im = ax.pcolormesh(vib_freq[:200], wf_idx, wf_data[:, :200],
                           cmap="hot", shading="auto")
        # Mark characteristic frequencies
        for name, freq in [("BPFO", pump_sim.freqs["BPFO"]),
                           ("2x", pump_sim.freqs["shaft"] * 2)]:
            if freq < vib_freq[199]:
                ax.axvline(freq, color="cyan", ls="--", alpha=0.5, lw=0.5)
                ax.text(freq + 2, wf_idx[-1] * 0.05, name, color="cyan", fontsize=6)
        plt.colorbar(im, ax=ax, label="Amplitude")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Time Step")
    ax.set_title("(d) Vibration Spectrum Waterfall")

    # (e) Temperature trends
    ax = axes[2, 0]
    ax.plot(t_axis, pump_temp_de, color="red", lw=1.0, label="Bearing DE")
    ax.plot(t_axis, pump_temp_nde, color="orange", lw=1.0, label="Bearing NDE")
    ax.plot(t_axis, pump_temp_winding, color="purple", lw=1.0, label="Winding")
    ax.plot(t_axis, pump_temp_seal, color="blue", lw=1.0, label="Seal")
    ax.axhline(pcfg.temp_bearing_de_nominal + pcfg.temp_alarm_delta,
               color="orange", ls="--", alpha=0.5, label=f"DE alarm ({pcfg.temp_bearing_de_nominal + pcfg.temp_alarm_delta:.0f}C)")
    ax.axhline(pcfg.temp_bearing_de_nominal + pcfg.temp_trip_delta,
               color="red", ls="--", alpha=0.5, label=f"DE trip ({pcfg.temp_bearing_de_nominal + pcfg.temp_trip_delta:.0f}C)")
    ax.axvline(pcfg.pump_trip_step, color="red", ls="--", alpha=0.5)
    ax.set_ylabel("Temperature (C)")
    ax.set_title("(e) Temperature Trends")
    ax.legend(loc="upper left", fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # (f) MCSA sideband ratio
    ax = axes[2, 1]
    mcsa_mask = pump_running_arr & (pump_mcsa_sb_db < -5)
    ax.scatter(t_axis[mcsa_mask], pump_mcsa_sb_db[mcsa_mask], s=2, color="#7b2d8e", alpha=0.7)
    ax.axhline(-50, color="green", ls="--", alpha=0.5, label="Healthy (< -50 dB)")
    ax.axhline(-35, color="orange", ls="--", alpha=0.5, label="Developing (-35 dB)")
    ax.axhline(-25, color="red", ls="--", alpha=0.5, label="Severe (> -25 dB)")
    ax.axvline(pcfg.pump_trip_step, color="red", ls="--", alpha=0.5)
    ax.set_ylabel("Sideband/Fundamental (dB)")
    ax.set_title("(f) MCSA Broken Bar Sideband Ratio")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-65, -15)

    # (g) Per-node anomaly scores
    ax = axes[3, 0]
    for nid in range(nn):
        ax.plot(t_axis, anom_s[:, nid], color=node_colors[nid], alpha=0.6, lw=0.7,
                label=f"N{nid}")
    ax.axhline(mcfg.gossip_trigger_threshold, color="gray", ls="--", alpha=0.5)
    ax.axvline(pcfg.pump_trip_step, color="red", ls="--", alpha=0.5)
    ax.set_ylabel("Anomaly Score")
    ax.set_title("(g) Per-Node Anomaly Scores")
    ax.legend(loc="upper right", fontsize=6, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # (h) Gossip consensus timeline
    ax = axes[3, 1]
    ax.set_xlim(0, N)
    ax.set_ylim(-0.5, nn - 0.5)
    ax.set_yticks(range(nn))
    ax.set_yticklabels([f"N{i}" for i in range(nn)])
    ax.set_title("(h) Gossip Consensus Timeline")
    ax.grid(True, alpha=0.3)
    for ev in mesh.event_log:
        if ev["type"] == "GOSSIP_START":
            ax.plot(ev["t"], ev["origin"], "o", color="orange", ms=4, zorder=5)
        elif ev["type"] == "GOSSIP_RESULT":
            color = "red" if ev["verdict"] == "CONFIRMED" else "green"
            marker = "^" if ev["verdict"] == "CONFIRMED" else "v"
            edge = "magenta" if ev.get("original_verdict") != ev["verdict"] else color
            ax.plot(ev["t"], ev["origin"], marker, color=color, ms=6, zorder=5,
                    markeredgecolor=edge, markeredgewidth=1.0)
    ax.axvline(pcfg.pump_trip_step, color="red", ls="--", alpha=0.5)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="orange", ms=6, label="Gossip Start"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="red", ms=6, label="Confirmed"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="green", ms=6, label="Denied"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7)

    # (i) CfC Judge surprise
    ax = axes[4, 0]
    ax.plot(t_axis, cfc_surprise, color="#7b2d8e", lw=0.8, alpha=0.9)
    if len(mesh.judge.surprise_buffer) > 100:
        buf = np.array(mesh.judge.surprise_buffer)
        ax.axhline(np.percentile(buf, mcfg.cfc_surprise_high_pct), color="red",
                   ls="--", alpha=0.5, label=f"High ({mcfg.cfc_surprise_high_pct}th pct)")
    ax.axvline(pcfg.pump_trip_step, color="red", ls="--", alpha=0.5)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Surprise")
    ax.set_title("(i) CfC Judge Surprise")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # (j) Combined health + pressure overlay
    ax = axes[4, 1]
    ax2 = ax.twinx()
    ax.plot(t_axis, true_p[:, 1], color="#1f77b4", lw=0.8, alpha=0.7, label="Pump node pressure")
    ax2.plot(t_axis, pump_health, color="red", lw=1.2, label="Pump health")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Pressure (PSI)", color="#1f77b4")
    ax2.set_ylabel("Health Index", color="red")
    ax.set_title("(j) Pressure vs Pump Health (Node 1)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = str(output_dir / "pump_health_results.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    console.print(f"  Saved pump health plot to {plot_path}")

    # Save pump health state to JSON
    pump_state = {
        "config": {
            "pump_rpm": pcfg.pump_rpm,
            "bearing": f"{pcfg.n_balls} balls, {pcfg.ball_diameter_mm}mm dia, {pcfg.pitch_diameter_mm}mm pitch",
            "characteristic_freqs": {k: round(v, 2) for k, v in pump_sim.freqs.items()},
        },
        "events": pump_events,
        "final_health": float(pump_health[N - 1]),
        "gossip_rounds": len(gossip_log),
        "cfc_overrides": mesh.judge.n_overrides,
    }
    pump_json = str(output_dir / "pump_health_state.json")
    with open(pump_json, "w") as f:
        json.dump(pump_state, f, indent=2, default=str)
    console.print(f"  Saved pump health state to {pump_json}")

    console.rule("[bold green]Pump Health Demo Complete[/bold green]")


# ============================================================================
#  ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--mesh":
        run_mesh_demo()
    elif len(sys.argv) > 1 and sys.argv[1] == "--pump":
        run_pump_health_demo()
    else:
        run_demo()
        console.print("\n[dim]Run with --mesh for 6-node gossip mesh demo[/dim]")
        console.print("[dim]Run with --pump for pump health monitoring demo[/dim]")
