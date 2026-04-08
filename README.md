
# Embodied Morphological Navigation (EM-NAV)

> Teaching an AI agent to build a mental map of a 3D world using a brain-inspired neural network.

EM-NAV is a computational neuroscience project that investigates whether a **Spiking Neural Network (SNN)** agent, trained via reinforcement learning inside a 3D Blender environment, develops emergent spatial representations analogous to the **place cells** and **grid cells** that give biological organisms their sense of space.

The hippocampus of a rat weighing 1.5 grams can build and maintain a precise map of its environment on roughly 20 watts of whole-brain power. The best deep RL agents solving equivalent maze tasks need GPUs drawing 300+ watts just for inference. EM-NAV asks whether biologically inspired sparsity closes that gap — and whether it forces the same spatial structure to emerge.

The entire pipeline runs on a CPU laptop for development and Google Colab free tier for training. No GPU. No cloud credits. No institutional compute.

---

## The Core Question

Dense neural networks activate every neuron on every forward pass. The brain does not. At any moment, roughly 2–5% of hippocampal neurons are active — a property called **sparse coding**. Spiking Neural Networks replicate this by only computing when a neuron crosses a firing threshold.

**Does enforcing that biological sparsity cause a navigation agent to develop spatial structure in its internal representations — and does it do so more efficiently than a standard dense network?**

---

## Architecture

```
Blender 4.x (headless)              Python / snnTorch
──────────────────────              ──────────────────
3D maze environment      ←───────→  LIF neuron network
5 raycast sensors         state     64 → 32 → 3 neurons
physics stepping          action    PPO reinforcement learning
```

**Input:** 5 normalised distance values from the Bio-Bot's raycasting sensors  
**Output:** 3 motor commands — Forward, Turn Left, Turn Right  
**Analysis:** Spatial firing rate maps of hidden neurons post-training, scored for place-field and grid-like structure

---

## Stack

| Tool | Role |
|------|------|
| Blender 4.x (Workbench renderer) | 3D world, physics simulation, headless sensor data |
| snnTorch + PyTorch | Leaky Integrate-and-Fire neurons, surrogate gradient training |
| Gymnasium | RL environment wrapper bridging Blender and Python |
| PPO | Policy optimisation algorithm |
| Google Colab | Free-tier GPU/CPU training runtime |
| NumPy / SciPy / Matplotlib | Spatial analysis and firing rate visualisation |

---

## Research Context

This project is part of a broader trajectory toward MSc Computational Neuroscience. It builds directly on prior work in multimodal deep learning ([Emotiwave preprint](https://doi.org/10.6084/m9.figshare.31567024)) and physiological signal processing (Neuro-Fatigue Predictor, 1D-CNN on FatigueSet).

**Status:** Ongoing

**Author:** Angelic Charles  
**ORCID:** [0009-0008-7279-9663](https://orcid.org/0009-0008-7279-9663)  
**Portfolio:** [visionbyangelic.github.io](https://visionbyangelic.github.io)

---

## License

MIT — reproduce, extend, cite freely.
