# Physics-Based Spiking Neural Networks as a Drop-In Replacement for KNN in Self-Supervised Online Learning

## Overview

Yes — a physics-informed spiking neural network (PI-SNN) can replace the KNN memory module in a self-supervised online learning system, and in several respects it is a *superior* architectural choice. The key insight is that a KNN serves as an episodic memory: given a query embedding, it retrieves relevant past states by proximity. An SNN can perform the same function — but parametrically, adaptively, and with native temporal dynamics — by encoding past experience into synaptic weights and spike timing patterns rather than an explicit buffer. The tradeoff is that the SNN is *distributed* memory (harder to inspect, harder to delete entries from) versus KNN's *explicit* memory (exact lookup, but grows linearly with data and has no gradient by default).

The architectural question is not just "can it work" but "how does the SNN replace each function the KNN was performing, and what do you gain or lose?" This report addresses that directly.

***

## What the KNN Was Actually Doing

In the prior BPTT + KNN + self-supervised online setup, the KNN memory served three roles:

1. **Temporal context provider** — retrieved embeddings of similar past states to augment the current query
2. **Hard-negative miner** — provided nearby embeddings from different operational regimes for contrastive loss
3. **Anomaly scorer** — high minimum-KNN-distance = no similar past state seen = anomaly signal

A PI-SNN can replicate all three, natively and with gradient flow, but via fundamentally different mechanisms.

***

## SNN as a Parametric Memory: How It Works

### Leaky Integrate-and-Fire Dynamics as Temporal Memory

The LIF neuron is defined by the ODE:

\[\tau_m \frac{dU}{dt} = -U(t) + I(t)\]

where \(U(t)\) is membrane potential, \(\tau_m\) is the membrane time constant (RC product), and \(I(t)\) is synaptic input current. Solved with the forward Euler method and a threshold-reset rule, this becomes a discrete recurrence:[^1][^2]

\[U_t = \beta U_{t-1} + W_{\text{in}} x_t - S_{t-1} \cdot U_{\text{thr}}\]

where \(\beta = e^{-1/\tau_m}\) is the decay factor (leakage), \(S_t \in \{0,1\}\) is the spike output, and \(U_{\text{thr}}\) is the firing threshold. Crucially, **the membrane potential \(U_t\) is a running integral of past inputs weighted by \(\beta\)**. This makes every LIF neuron a native short-term memory element with a physically meaningful time constant — no external buffer required.[^3]

For a recurrent SNN (RSNN), the lateral connections \(W_{\text{rec}}\) extend this to full spatio-temporal memory, encoding sequences of past activity into the network's current state. The RSNN hidden state at time \(t\) is a function of all prior inputs, shaped by the learned weights.[^4][^5]

### Physics Constraint as Inductive Bias

A physics-informed SNN adds a residual loss term based on governing equations of the physical system being monitored. For a drilling application (vibration dynamics, fluid pressure, etc.) this typically takes the form:

\[\mathcal{L}_\text{total} = \mathcal{L}_\text{data} + \lambda \mathcal{L}_\text{physics}\]

where \(\mathcal{L}_\text{physics}\) penalizes violations of known PDEs or ODE residuals (e.g., momentum balance, pressure wave equations). The physics term provides a strong regularizer that:[^6][^7]
- Constrains the learned representations to lie on physically plausible manifolds
- Enables learning from far less data than a purely data-driven network
- Prevents the network from memorizing spurious noise patterns that violate physical laws[^8][^9]

Recent work on PI-SNNs (Tandale 2024, 2026; Wang 2023) demonstrates that physics-based loss functions guide LIF neuron dynamics to encode physically meaningful temporal patterns, and that this yields faster convergence and better generalization than data-driven SNNs alone.[^9][^6][^8]

***

## The Training Problem: SNN Gradients are Non-Differentiable by Default

Just like the KNN's `argmin`, the Heaviside step function used for SNN spike generation has zero gradient almost everywhere:

\[S_t = \Theta(U_t - U_{\text{thr}}), \quad \frac{dS_t}{dU_t} = \delta(U_t - U_{\text{thr}})\]

This is a hard non-differentiability that blocks standard backpropagation. The solution is **surrogate gradients** — replacing the true Heaviside derivative with a smooth approximation during the backward pass only:

\[\frac{d\tilde{S}}{dU} = \frac{1}{(k|U - U_{\text{thr}}| + 1)^2}\]

The forward pass uses the true binary spike; the backward pass uses the surrogate, allowing gradients to flow through spike events. This is conceptually identical to the straight-through estimator used in discrete VAEs and is fully compatible with PyTorch autograd.[^10][^11][^12]

With surrogate gradients in place, BPTT through the unrolled SNN is well-defined and effective.[^13]

***

## Online Training Without Full BPTT: OTTT and BrainTrace

Applying full BPTT to an SNN online is expensive — memory scales linearly with sequence length \(T\), which is prohibitive for continuous streaming data.[^4][^13]

**Online Training Through Time (OTTT)** (NeurIPS 2022) solves this by deriving an equivalent forward-in-time update rule from BPTT. Rather than storing full trajectories and backpropagating later, OTTT tracks *presynaptic activity traces* and computes weight updates instantaneously at each timestep:[^14][^13]

> OTTT requires only *constant* training memory regardless of sequence length, avoiding the large memory costs of BPTT for GPU training.

The update rule takes the form of a **three-factor Hebbian learning rule** — presynaptic trace × postsynaptic error × neuromodulatory signal — which maps directly to biologically plausible plasticity and is implementable on neuromorphic hardware.[^13][^14]

**BrainTrace** (Nature Communications, 2026) goes further, providing a model-agnostic compiler that automatically generates linear-memory online learning code from arbitrary user-defined SNN models, validated at whole-brain scale (Drosophila connectome). This is the state-of-the-art framework for SNN online learning as of 2026.[^15][^4]

For edge deployment (e.g., Raspberry Pi 5 on a drilling platform), the constant memory footprint of OTTT is the enabling property — BPTT on a sequence of even a few thousand timesteps at 1kHz sensor rate would exhaust GPU memory on a data-center GPU, let alone an edge device.

***

## Self-Supervised Learning Signal in an SNN: CSDP

The KNN in the prior architecture provided a contrastive signal via soft nearest-neighbor loss. An SNN can generate its own self-supervised learning signal without labels via **Contrastive Signal-Dependent Plasticity (CSDP)** (Ororbia, *Science Advances*, 2024).[^16][^17][^18]

CSDP generalizes the "forward-forward" contrastive principle into spiking dynamics. Each layer of the SNN locally maximizes the "goodness" (aggregate spike activation) for real inputs and minimizes it for corrupted/OOD inputs. The weight update at each synapse depends only on:
- Local presynaptic spike traces
- Local postsynaptic spike traces
- A local contrastive modulatory signal (the difference in goodness between positive and negative inputs)

This is **fully local** — no global backpropagation is needed, and no labels are required. The system continuously adjusts its representations to distinguish real system states from anomalous or OOD states, which is exactly the self-supervised objective needed for predictive maintenance.[^19][^16]

An additional approach is **SpikeMatch** (2025), which exploits the LIF leakage factor \(\beta\) as a source of temporal diversity for pseudo-label generation without any external labeling infrastructure. By running the same input through neurons with different \(\tau_m\), different predictions are generated and agreement is used to produce reliable pseudo-labels — a co-training scheme entirely native to SNN dynamics.[^10]

***

## SNN vs. KNN: Detailed Comparison

| Property | KNN Memory | Physics-Informed SNN |
|---|---|---|
| **Temporal context** | Explicit buffer lookup (exact) | Membrane potential + synaptic weights (distributed) | 
| **Gradient flow** | Non-differentiable (needs soft surrogate)[^20] | Non-differentiable spikes, but surrogate gradients well-established[^11][^12] |
| **Memory cost** | O(n · d) grows with data[^21] | O(neurons) — fixed regardless of history length[^4][^13] |
| **Update mechanism** | FIFO buffer, entries added/removed explicitly | Synaptic plasticity (STDP, OTTT, CSDP)[^22][^13][^16] |
| **Physics constraint** | None (external) | Native loss term; conserves PDEs by construction[^6][^7] |
| **Anomaly score** | Min KNN distance in embedding space | Spike rate deviation from baseline; reconstruction error |
| **Online adaptability** | Immediate (new entries stored directly) | Requires weight updates (fast, but latency exists)[^23] |
| **Hardware efficiency** | CPU-bound, no neuromorphic benefit | Runs natively on neuromorphic chips (Loihi, Intel)[^24][^25] |
| **Catastrophic forgetting** | None (explicit memory never overwrites) | Significant risk; needs mitigation (EWC, columnar structure)[^26][^27] |
| **Entry-level deletion** | O(1) index removal | Not possible (distributed representation)[^21] |
| **Self-supervised signal** | Soft KNN loss, contrastive retrieval | CSDP, SpikeMatch, STDP-based rules[^16][^10][^28] |

***

## The Critical Trade-off: Catastrophic Forgetting

This is the biggest risk when replacing KNN with an SNN. A KNN buffer never forgets — every entry persists until explicitly evicted. An SNN trained online with STDP or OTTT *will* overwrite previously learned patterns when new data arrives, a phenomenon called catastrophic forgetting (CF).[^26]

The most effective mitigation strategies for online SNN learning are:

- **Columnar organization**: Separate microcolumns of neurons specialize for different tasks/regimes. New tasks recruit fresh columns rather than overwriting old ones. A 2025 study achieved 92% accuracy across 10 sequential tasks with only 4% degradation on the first task.[^27]
- **Bayesian continual learning**: Each synapse is represented by a distribution rather than a point estimate; uncertainty from prior tasks prevents rapid overwriting. Validated on Intel's Lava neuromorphic platform.[^24]
- **Heterogeneous STDP**: Different synapses have different LTP/LTD dynamics, providing natural timescale separation between fast-adapting and stable connections.[^5]
- **Few-shot reminders**: Retaining a small episodic replay buffer (10% of prior data) and replaying periodically substantially reduces CF without the storage cost of a full KNN buffer.[^26]

For a streaming sensor application like SAIREN where the operational baseline drifts continuously (changing formation, new drill bit, different mud weight), controlled plasticity is actually desirable — the network *should* adapt its baseline. The challenge is preventing rapid excursions from valid past knowledge. Bayesian weight distributions or columnar organization provide this stability.

***

## Anomaly Detection with an SNN: Replacing KNN Distance

In the KNN-based anomaly detector, the signal was: *if the current embedding has no close neighbors in memory, it is anomalous*. In an SNN, the equivalent signals are:

1. **Reconstruction error** — an SNN autoencoder trained on normal states will have high reconstruction loss on anomalous inputs. SNN autoencoders have been validated for anomaly detection under strict latency constraints (e.g., LHC trigger systems).[^29][^30]
2. **Spike rate deviation** — a well-trained SNN produces characteristic spike rate patterns for normal states. Anomalies produce spike patterns that deviate from the learned norm, measurable as a real-time distance in spike-rate space.
3. **Membrane potential saturation** — neurons that have never seen a particular input pattern will either fire excessively (oversaturation) or not at all (undersaturation), both of which are detectable without a distance index.

Continuous local learning SNN implementations (biorXiv 2023) have demonstrated that a continuously-trained SNN can adapt to sudden changes in input neural structure — directly analogous to adapting to changing rig baseline — with 92% reduction in training memory versus BPTT.[^23]

***

## Practical Architecture: PI-SNN Replacing KNN in SAIREN

```
Sensor stream: x_t (vibration, pressure, RPM, temperature, etc.)

Physics encoder:
  - Leaky integrate-and-fire RSNN with τ_m tuned to sensor sampling rate
  - Physics loss: L_phys = ||residual of known dynamics equations||²
  - Surrogate gradient (triangular) for BPTT / OTTT for online mode

Online learning:
  - OTTT (constant memory) OR BrainTrace compiler
  - CSDP local contrastive signal: distinguish real states vs. augmented OOD states
  - Columnar structure for stability-plasticity balance

Anomaly signal:
  - Spike rate deviation from running exponential moving average
  - Reconstruction error from SNN autoencoder head
  - Physics residual magnitude (direct physical violation indicator)

No KNN memory buffer required — synaptic weights ARE the memory
```

The physics residual term is particularly powerful in a drilling context because it means the network can flag anomalies that violate known physical laws (e.g., pressure spikes that violate conservation of mass in the mud circuit) *even without having seen that specific fault mode before* — a capability completely absent from a KNN or standard neural network.[^7][^6]

***

## When to Keep the KNN (Hybrid Approach)

Replacing KNN with an SNN is not always optimal. Specific scenarios where a **hybrid PI-SNN + small KNN** outperforms either alone:

- **Rare fault types** that occur too infrequently for the SNN to consolidate into stable weights — keep a small KNN buffer for catalogued fault signatures
- **Post-hoc explainability** — a KNN can retrieve the specific historical event that matches the current anomaly ("this looks like the mud pump cavitation event from 2024-03-15"), which an SNN cannot do natively
- **Hard reset requirements** — if regulations or safety systems require the ability to *remove* a specific learned pattern (e.g., a mislabelled reference event), KNN supports O(1) deletion while the SNN does not

In a hybrid design, the SNN handles the continuous baseline adaptation and physics-constrained anomaly detection, while the KNN holds a curated reference library of confirmed fault events for classification and explainability.

---

## References

1. [A quantum leaky integrate-and-fire spiking neuron and network](https://www.nature.com/articles/s41534-024-00921-x) - The membrane potential, U(t), is decomposed into a resistor and capacitor which form a linear differ...

2. [1.3 Integrate-And-Fire Models | Neuronal Dynamics online book](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html) - Neuron models where action potentials are described as events are called 'Integrate-and-Fire' models...

3. [Tutorial 2 - The Leaky Integrate-and-Fire Neuron — snntorch 0.9.4 ...](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_2.html) - Learn the fundamentals of the leaky integrate-and-fire (LIF) neuron model ... The different versions...

4. [Model-agnostic linear-memory online learning in spiking neural ...](https://www.nature.com/articles/s41467-026-68453-w) - Online learning algorithms for SNNs have emerged as a compelling alternative. By updating parameters...

5. [Heterogeneous recurrent spiking neural network for spatio-temporal ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC9922697/) - This paper presents a heterogeneous recurrent spiking neural network (HRSNN) with unsupervised learn...

6. [Physics-Based Self-Learning Spiking Neural Network enhanced time ...](https://www.sciencedirect.com/science/article/pii/S0045782524001038) - by SB Tandale · 2024 · Cited by 14 — The present study introduces a new physics-based self-learning ...

7. [Physics-Informed Spiking Neural Networks via Conservative Flux ...](https://arxiv.org/abs/2511.21784) - Physics-Informed Neural Networks (PINNs) combine data-driven learning with physics-based constraints...

8. [Meta-learning Hybrid Spiking networks as physics-based ...](https://www.nature.com/articles/s44335-025-00048-y) - by SB Tandale · 2026 · Cited by 1 — This study proposes a physics-based, self-learning framework tha...

9. [Physics Informed Spiking Neural Networks](https://ieeexplore.ieee.org/iel7/6287639/6514899/10122961.pdf) - by S Wang · 2023 · Cited by 14 — In case where any knowledge of physical laws are known, the problem...

10. [SpikeMatch: Semi-Supervised Learning with Temporal Dynamics of ...](https://arxiv.org/html/2509.22581v1) - Spiking neural networks (SNNs) use the Heaviside step function as the non-linear activation of leaky...

11. [Surrogate gradient learning in spiking networks trained on event ...](https://pubmed.ncbi.nlm.nih.gov/38859258/) - We effectively apply the surrogate gradient method to overcome this issue achieving over 99% classif...

12. [Tutorial on surrogate gradient learning in spiking networks online](https://zenkelab.org/2019/03/tutorial-on-surrogate-gradient-learning-in-spiking-networks-online/) - This tutorial version illustrates how to use surrogate gradients in modern ML auto-diff frameworks. ...

13. [[PDF] Online Training Through Time for Spiking Neural Networks - NeurIPS](https://papers.neurips.cc/paper_files/paper/2022/file/82846e19e6d42ebfd4ace4361def29ae-Paper-Conference.pdf) - Spiking neural networks (SNNs) are promising brain-inspired energy-efficient models. Recent progress...

14. [Online Training Through Time for Spiking Neural Networks | Papers](https://hyper.ai/en/papers/2210.04195) - WithOTTT, it is the first time that two mainstream supervised SNN training methods,BPTT with SG and ...

15. [Model-agnostic linear-memory online learning in spiking neural ...](https://ideas.repec.org/a/nat/natcom/v17y2026i1d10.1038_s41467-026-68453-w.html) - Here, we introduce BrainTrace, a model-agnostic, linear-memory, and automated online learning system...

16. [Contrastive signal–dependent plasticity: Self-supervised learning ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC11639678/) - by AG Ororbia · 2024 · Cited by 9 — A promising pathway to mortal computation centers around a famil...

17. [Contrastive signal–dependent plasticity: Self-supervised learning in ...](https://www.science.org/doi/10.1126/sciadv.adn6076) - A process which generalizes ideas behind self-supervised learning to facilitate local adaptation in ...

18. [Self-Supervised Learning in Spiking Neural Circuits - arXiv](https://arxiv.org/html/2303.18187v3) - Ororbia, Contrastive signal–dependent plasticity: Self-supervised learning in spiking neural circuit...

19. [Self-Supervised Learning in Spiking Neural Circuits - arXiv](https://arxiv.org/abs/2303.18187) - Contrastive-Signal-Dependent Plasticity: Self-Supervised Learning in Spiking Neural Circuits. Author...

20. [Knn doubt and question related to distances between k neigh #1322](https://github.com/rusty1s/pytorch_geometric/issues/1322) - I see that to find the nearest neighbor, and you're using KDTree from spatial scypi. The first quest...

21. [What is the k-nearest neighbors algorithm? - IBM](https://www.ibm.com/think/topics/knn) - The k-nearest neighbors (KNN) algorithm is a non-parametric, supervised learning classifier, which u...

22. [S-TLLR: STDP-inspired Temporal Local Learning Rule for Spiking ...](https://arxiv.org/html/2306.15220v4) - The idea behind this approach is to mimic key features of biological neurons, such as spike-based co...

23. [A Spiking Neural Network with Continuous Local Learning for ...](https://www.biorxiv.org/content/10.1101/2023.08.16.553602v1.full-text) - The continuous learning SNN shows a significantly reduced memory footprint compared to the BPTT lear...

24. [Bayesian continual learning via spiking neural networks - PMC - NIH](https://pmc.ncbi.nlm.nih.gov/articles/PMC9708898/) - The main aim of this paper is to introduce algorithmic solutions to endow neuromorphic models, namel...

25. [[PDF] Spiking Physics-Informed Neural Networks on Loihi 2 - OSTI](https://www.osti.gov/servlets/purl/2564042) - Recent efforts have demonstrated how to convert a trained PINN to a spiking network architecture. In...

26. [Continuous learning of spiking networks trained with local rules](https://www.sciencedirect.com/science/article/abs/pii/S0893608022003379) - In this paper, we study the susceptibility of SNNs to CF and test several biologically inspired meth...

27. [Continual Learning with Columnar Spiking Neural Networks - arXiv](https://arxiv.org/html/2506.17169v2) - First, CF represents the plasticity-stability dilemma: excessive learning plasticity interferes with...

28. [Incremental self-organization of spatio-temporal spike pattern ...](https://www.nature.com/articles/s41598-025-21460-1) - by M Dehghani Habibabadi · 2025 — A simple one-layer spiking neural network model is presented that ...

29. [Anomaly detection with spiking neural networks for LHC physics](https://arxiv.org/abs/2508.00063) - Abstract:Anomaly detection offers a promising strategy for discovering new physics at the Large Hadr...

30. [[PDF] Anomaly Detection with Spiking Neural Networks - CERN Indico](https://indico.cern.ch/event/1074443/contributions/4518238/attachments/2335509/3980722/IRIS_HEP_emoreno.pdf) - Anomaly detection sequence: 1. Train autoencoder to encoder and decode data on data with no anomalie...

