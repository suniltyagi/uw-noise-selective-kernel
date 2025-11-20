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

%% ========== STYLE DEFINITIONS ==========
classDef module fill:#f2f7ff,stroke:#3366cc,stroke-width:1px,color:#000;
classDef process fill:#e8fff2,stroke:#33aa55,stroke-width:1px,color:#000;
classDef loss fill:#fff2e6,stroke:#ff9933,stroke-width:1px,color:#000;

%% ========== NODE DEFINITIONS ==========
A1[1. Raw Audio (x)]:::module
A2[2. SKConv1D Filterbank (Multi-Scale Time-Domain Filters)]:::process
A3[3. Learned Time-Feature Map (Learned Spectrogram)]:::module
A4[4. SKConv2D Encoder (Hierarchical 2D Feature Extraction)]:::process
A5[5. Base Embedding h (R^D)]:::module

B1[6. Augmentation 1: Aug1(x)]:::process
B2[7. Augmentation 2: Aug2(x)]:::process

C1[8. Siamese Encoder f(x) - Shared SKConv1D + SKConv2D]:::process

D1[9. Projector Head g(h)]:::module
D2[10. Projected Embeddings z1, z2]:::module

E1[11. Barlow Twins Loss (Invariance + Decorrelation)]:::loss
E2[12. Update Encoder Parameters]:::process

%% ========== MAIN ENCODER PIPELINE ==========
A1 --> A2 --> A3 --> A4 --> A5

%% ========== SSL BRANCHING ==========
A5 -->|Used during SSL| B1
A5 -->|Used during SSL| B2

B1 --> C1
B2 --> C1

C1 --> D1 --> D2 --> E1 --> E2
```

---

## 🧠 **Key Features**

- SKConv1D learned filterbank replaces STFT/Mel  
- Learned spectrogram representation  
- SKConv2D hierarchical encoder  
- Barlow Twins self-supervised training  
- Works directly from raw waveform  
- Embeddings suitable for clustering (HDBSCAN / DBSCAN)  
- Ideal for sonar, HAVS, and underwater noise analysis  

---

## 📁 **Repository Structure**

```
uw-noise-selective-kernel/
│
├── src/
│   ├── models/
│   ├── ssl/
│   ├── training/
│   └── clustering/
│
├── notebooks/
├── configs/
├── logo.svg
├── requirements.txt
└── README.md
```

---

## ⚙️ **Installation**
```bash
git clone https://github.com/suniltyagi/uw-noise-selective-kernel
cd uw-noise-selective-kernel
pip install -r requirements.txt
```

---

## 🏋️ **Training**
```bash
python src/training/trainer.py --config configs/training.yaml
```

---

## 📊 **Clustering**
```bash
python src/clustering/hdbscan_cluster.py
```

---

## 📜 **License**
MIT License (or your chosen license)
