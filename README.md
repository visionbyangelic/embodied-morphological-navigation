
# Embodied Morphological Navigation (EM-NAV)
> Teaching an AI agent to build a mental map of a 3D world using a brain-inspired neural network.

EM-NAV is a computational neuroscience project that investigates whether a **Spiking Neural Network (SNN)** agent, trained via reinforcement learning, develops emergent spatial representations analogous to the **place cells** and **grid cells** that give biological organisms their sense of space.

The hippocampus of a rat weighing 1.5 grams can build and maintain a precise map of its environment on roughly 20 watts of whole-brain power. The best deep RL agents solving equivalent maze tasks need GPUs drawing 300+ watts just for inference. EM-NAV asks whether biologically inspired sparsity closes that gap and whether it forces the same spatial structure to emerge.

The entire pipeline runs on a CPU laptop for development and Google Colab free tier for training. No GPU. No cloud credits. No institutional compute.

---

## The Core Question

Dense neural networks activate every neuron on every forward pass. The brain does not. At any moment, roughly 2–5% of hippocampal neurons are active — a property called **sparse coding**. Spiking Neural Networks replicate this by only computing when a neuron crosses a firing threshold.

**Does enforcing that biological sparsity cause a navigation agent to develop spatial structure in its internal representations and does it do so more efficiently than a standard dense network?**

This is not a question about building a better navigation system. It is a question about whether the *constraints* biology operates under — sparse, event-driven computation are sufficient to *cause* the spatial structure we observe in the hippocampus, or whether that structure requires something else entirely.

---

## Architecture

The project uses a two-environment strategy:

```
MiniGrid (training)                  Python / snnTorch
───────────────────                  ──────────────────
Standard Gym maze env   ←─────────→  LIF neuron network
Fast CPU stepping        state        64 → 32 → 3 neurons
Reproducible, citable   action       PPO reinforcement learning

Blender 5.x (visualisation)
────────────────────────────
3D maze environment
5 raycast sensors on Bio-Bot agent
Physics stepping
Visual demonstration of learned policy
Place field overlay rendering
```

**Training** happens in MiniGrid — a well-established, fast, and reproducible Gym environment used extensively in spatial navigation research. This ensures the training pipeline is credible, citable, and runnable by anyone without specialist software.

**Visualisation** happens in Blender — the trained policy is transferred into a 3D environment where the Bio-Bot agent navigates using 5 raycasting sensors, producing visual demonstrations of emergent behaviour and spatial firing overlays for analysis and communication.

**Input:** Grid observation (MiniGrid) / 5 normalised raycast distances (Blender)  
**Output:** 3 motor commands — Forward, Turn Left, Turn Right  
**Analysis:** Spatial firing rate maps of hidden layer neurons post-training, scored for place-field and grid-like structure against shuffled controls

---

## The Experimental Design

Two agents are trained on identical maze tasks under identical conditions:

| Agent | Network | Firing Pattern |
|-------|---------|---------------|
| SNN agent | Leaky Integrate-and-Fire neurons (snnTorch) | Sparse, event-driven |
| Dense baseline | Standard ReLU MLP | Dense, continuous |

Both are trained with PPO. After training, the internal representations of each agent are analysed:

- **Firing rate maps** — does any hidden neuron fire preferentially in a specific region of space?
- **Place field scoring** — spatial autocorrelation and peak-to-mean ratio against shuffled controls
- **Grid-like structure** — hexagonal symmetry scoring (borrowed from grid cell literature)
- **Efficiency comparison** — task performance per unit of compute (FLOPs, wall-clock time, active neuron ratio)

The hypothesis: the SNN's enforced sparsity will produce more localised, spatially structured representations than the dense baseline, at lower computational cost.

---

## Stack

| Tool | Role |
|------|------|
| MiniGrid | Primary training environment — standard, fast, reproducible maze navigation |
| Blender 5.x | 3D visualisation environment — policy demonstration, raycast sensors, place field overlays |
| snnTorch + PyTorch | Leaky Integrate-and-Fire neurons, surrogate gradient training |
| Gymnasium | RL environment wrapper |
| Stable-Baselines3 | PPO implementation |
| Google Colab | Free-tier training runtime |
| NumPy / SciPy / Matplotlib | Spatial analysis and firing rate visualisation |

---

## Project Phases

**Phase 1 — Environment & Pipeline** *(current)*
- [x] Blender 5.x scene with Bio-Bot agent and 5-sensor raycast system
- [x] Socket-based communication between Blender and external Python
- [x] MiniGrid environment verified and integrated
- [ ] Gymnasium wrapper connecting MiniGrid to SNN policy
- [ ] End-to-end step/reset loop confirmed

**Phase 2 — SNN Agent**
- [ ] LIF neuron network (snnTorch) — 64 → 32 → 3 architecture
- [ ] Dense MLP baseline — matched parameter count
- [ ] PPO training loop for both agents
- [ ] Training convergence confirmed on MiniGrid-Empty and MiniGrid-FourRooms

**Phase 3 — Spatial Analysis**
- [ ] Firing rate map generation for all hidden neurons
- [ ] Place field scoring (spatial autocorrelation, peak-to-mean ratio)
- [ ] Grid-like structure scoring
- [ ] Shuffled control comparisons
- [ ] Efficiency metrics (FLOPs, active neuron ratio, wall-clock time)

**Phase 4 — Visualisation & Communication**
- [ ] Trained policy transferred to Blender 3D environment
- [ ] Place field overlays rendered in Blender viewport
- [ ] Figures and results writeup
- [ ] Potential preprint submission

---

## Research Context

This project is part of a broader trajectory toward MSc Computational Neuroscience. It builds directly on prior work in multimodal deep learning ([Emotiwave preprint](https://doi.org/10.6084/m9.figshare.31567024)) and physiological signal processing (Neuro-Fatigue Predictor, 1D-CNN on FatigueSet).

The two-environment architecture (MiniGrid for training, Blender for visualisation) is a deliberate design choice: it separates the concerns of *scientific reproducibility* (anyone can run MiniGrid) from *visual communication* (Blender produces compelling figures and demonstrations for non-specialist audiences and portfolio use).

**Status:** Phase 1 — Ongoing  
**Author:** Angelic Charles  
**ORCID:** [0009-0008-7279-9663](https://orcid.org/0009-0008-7279-9663)  
**Portfolio:** [visionbyangelic.github.io](https://visionbyangelic.github.io)



## License

MIT — reproduce, extend, cite freely.
```

