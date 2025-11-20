# UW-Noise-Selective-Kernel  
### **Hybrid Selective Kernel Encoder for Underwater Noise Representation Learning**

<p align="center">
  <img src="logo.svg" width="260">
</p>

This repository contains the implementation of a **Selective Kernel (SKConv1D + SKConv2D) audio encoder** trained using **self-supervised learning (Barlow Twins)** to learn robust, domain-aware representations of **underwater acoustic noise**, including machinery signatures, flow noise, propeller tonals, and platform radiated noise.

Traditional pipelines depend on hand-crafted spectrograms (STFT/Mel).  
Here, the model **learns its own spectrogram** directly from raw waveforms.

---

## 🚀 **Architecture Overview**

### **Raw → SKConv1D → Learned Spectrogram → SKConv2D → Embedding → SSL → Clustering**

```mermaid
flowchart TD
%% Style settings
classDef module fill:#f2f7ff,stroke:#3366cc,stroke-width:1px,color:#000;
classDef process fill:#e8fff2,stroke:#33aa55,stroke-width:1px,color:#000;
classDef loss fill:#fff2e6,stroke:#ff9933,stroke-width:1px,color:#000;

%% Nodes
A1[1. Raw Audio (x)]:::module
A2[2. SKConv1D Filterbank<br/>(Multi-Scale Time-Domain Filters)]:::process
A3[3. Learned Time–Feature Map<br/>(Learned Spectrogram)]:::module
A4[4. SKConv2D Encoder<br/>(Hierarchical 2D Feature Extraction)]:::process
A5[5. Base Embedding h ∈ ℝᴰ]:::module
B1[6. Augmentation 1: Aug₁(x)]:::process
B2[7. Augmentation 2: Aug₂(x)]:::process
C1[8. Siamese Encoder f(x)<br/>(Shared SKConv1D + SKConv2D)]:::process
D1[9. Projector Head g(h)]:::module
D2[10. Projected Embeddings z₁, z₂]:::module
E1[11. Barlow Twins Loss<br/>(Invariance + Decorrelation)]:::loss
E2[12. Update Encoder Parameters]:::process

%% Main forward path
A1 --> A2 --> A3 --> A4 --> A5

%% SSL Branching
A5 -->|Used during SSL| B1
A5 -->|Used during SSL| B2
B1 --> C1
B2 --> C1
C1 --> D1 --> D2 --> E1 --> E2
```
