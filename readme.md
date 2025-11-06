# Hierarchical Adaptive Masking Graph Convolutional Network (HAM-GCN)

## Overview

This repository contains the PyTorch implementation of the **Hierarchical Adaptive Masking Graph Convolutional Network (HAM-GCN)**, a novel framework for EEG-based emotion classification. The model addresses the challenge of volume conduction effects in EEG signals by selectively suppressing spurious connections while preserving meaningful synchronizations through supervised training and a progressive masking mechanism.

## Key Features

- **Dynamic Graph Construction**: Adaptive pruning of spurious connections while preserving meaningful synchronizations
- **Progressive Node Masking**: Curriculum learning-based masking strategy that increases sparsity across network layers
- **Spectral Graph Convolution**: Chebyshev polynomial approximation for efficient spectral filtering
- **Interpretability Framework**: SHAP-based analysis for node importance and frequency band contribution
- **Multi-scale Feature Aggregation**: Hierarchical architecture capturing both local and global emotional patterns

## Model Architecture

### Core Components

1. **ChebyshevPoly**: K-order Chebyshev polynomial approximation with node masking
2. **HAM-GCN**: Main model class featuring:Attention-based masking mechanismProgressive node masking across layersDynamic adjacency matrix learningCross-layer feature fusion

### Progressive Masking Mechanism

The model implements a curriculum learning strategy where masking ratios progressively increase across layers:

- **Layer 1**: Lower masking ratio (preserves more nodes)
- **Layer 2**: Intermediate masking ratio
- **Layer 3**: Higher masking ratio (focuses on most critical nodes)



![img](https://hunyuan-plugin-private-1258344706.cos.ap-nanjing.myqcloud.com/pdf_youtu/img/f404886e8324dfa1ed9af65348565038-image.png?q-sign-algorithm=sha1&q-ak=AKID372nLgqocp7HZjfQzNcyGOMTN3Xp6FEA&q-sign-time=1762413906%3B2077773906&q-key-time=1762413906%3B2077773906&q-header-list=host&q-url-param-list=&q-signature=53f8f8c7bfd13fc3b183222b1b8caa1f21da9c34)



## Installation

bash

```
pip install torch numpy scipy torch-geometric
```

## Usage

### Basic Example



```
import torch
from ham_gcn import DGCNN_ProgressiveMask

# Model parameters
num_nodes = 62  # Number of EEG channels
input_dim = 5   # Features per channel (δ, θ, α, β, γ bands)
hidden_dim = 64
num_classes = 3  # Emotion categories
K = 3  # Chebyshev polynomial order
mask_ratios = [0.0, 0.3, 0.6]  # Layer-wise masking ratios

# Initialize model
model = DGCNN_ProgressiveMask(
    num_nodes=num_nodes,
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_classes=num_classes,
    K=K,
    mask_ratios=mask_ratios
)

# Example input (batch_size=16, 62 channels, 5 features)
x = torch.randn(16, num_nodes, input_dim)

# Forward pass
output = model(x)
print(f"Output shape: {output.shape}")  # [16, 3]

# Get learned adjacency matrix for interpretability
adj_matrix = model.get_final_adjacency()
print(f"Adjacency matrix shape: {adj_matrix.shape}")  # [62, 62]
```