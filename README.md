# UW-Noise-Selective-Kernel  
### **Hybrid Selective Kernel Encoder for Underwater Noise Representation Learning**

<p align="center">
  <img src="logo.svg" width="260">
</p>

This repository contains the implementation of a **Selective Kernel (SKConv1D + SKConv2D) audio encoder** trained using **self-supervised learning (Barlow Twins)** to learn robust, domain-aware representations of **underwater acoustic noise**, including machinery signatures, flow noise, propeller tonals, and platform radiated noise.

Traditional pipelines depend on hand-crafted spectrograms (STFT/Mel).  
Here, the model **learns its own spectrogram** directly from raw waveforms.

---

## 🚀 Architecture Overview

### Raw → SKConv1D → Learned Spectrogram → SKConv2D → Embedding → SSL → Clustering

```mermaid
flowchart TD

%% STYLE DEFINITIONS
classDef module fill:#f2f7ff,stroke:#3366cc,stroke-width:1px,color:#000;
classDef process fill:#e8fff2,stroke:#33aa55,stroke-width:1px,color:#000;
classDef loss fill:#fff2e6,stroke:#ff9933,stroke-width:1px,color:#000;

%% NODES
A1[Raw Audio]:::module
A2[SKConv1D Filterbank]:::process
A3[Learned Time-Feature Map]:::module
A4[SKConv2D Encoder]:::process
A5[Base Embedding h]:::module

B1[Augmentation 1]:::process
B2[Augmentation 2]:::process

C1[Siamese Encoder (shared weights)]:::process
D1[Projector Head]:::module
D2[Projected Embeddings]:::module

E1[Barlow Twins Loss]:::loss
E2[Update Encoder]:::process

%% MAIN PIPELINE
A1 --> A2 --> A3 --> A4 --> A5

%% SSL BRANCHES
A5 --> B1 --> C1
A5 --> B2 --> C1

%% PROJECTION + LOSS
C1 --> D1 --> D2 --> E1 --> E2
```
