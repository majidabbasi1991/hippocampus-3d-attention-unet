# Hippocampus 3D Attention U-Net

This repository contains the implementation of a **3D Attention U-Net** for **hippocampus MRI segmentation**, developed as part of an academic research project in medical image analysis.

The proposed model integrates:
- 3D U-Net architecture
- Attention Gates for region-focused feature learning
- Dilated convolutions to enhance contextual representation

The method was evaluated on the **Task04 Hippocampus dataset (Medical Segmentation Decathlon)** and achieved a **Dice similarity coefficient of 0.908** on the test set.

Project Status

**Manuscript in preparation.**  
The full methodological description, experimental protocol, and quantitative analysis will be published in a peer-reviewed journal.

Until publication, this repository is provided **for research transparency and reproducibility purposes only**.
Key Features
- 3D patch-based training strategy
- Attention Gates for suppressing irrelevant background regions
- Robust MRI preprocessing pipeline (resampling, normalization, foreground sampling)
- Dice-based evaluation for volumetric segmentation
  
## Repository Structure
hippocampus-3d-attention-unet/
│
├── preprocessing/        # MRI preprocessing pipelines
├── dataset/              # Dataset loading and patch sampling
├── models/               # 3D Attention U-Net implementation
├── training/             # Training and evaluation scripts
├── inference/            # Inference on full MRI volumes
└── README.md
