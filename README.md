# WPINN: Windkessel Physics Informed Neural Network

This repository contains implementations of the **Windkessel Physics Informed Neural Network (WPINN)** framework, designed to estimate patient-specific cardiovascular parameters and blood pressure waveforms.

## 🔍 Overview

The WPINN framework leverages a **baseline neural network architecture** that combines **CNNs** for local feature extraction and **Transformers** for capturing long-range temporal dependencies. This architecture is also adapted to estimate different types of Windkessel model parameters:

- **Constant Parameters (CP)**: Parameters that remain fixed across time
- **Beat-to-Beat Parameters (BBP)**: Dynamic parameters that vary on a per-beat basis
- **Pressure Dependent Parameters (PDP)**: Parameters that adapt in response to pressure changes

## 📁 Repository Structure

```
.
├── cnn_transformer_nn.py     # Baseline CNN + Transformer neural network architecture
├── cp_model.py               # Training model for estimating Constant Parameters (CP)
├── bbp_model.py              # Training model for estimating Beat-to-Beat Parameters (BBP)
└── pdp_model.py              # Training model for estimating Pressure Dependent Parameters (PDP)
```

## 🧠 Model Details

### `cnn_transformer_nn.py`

This file defines the **core neural network architecture**, combining:

- **Convolutional layers (CNN)** to capture input waveform spatial features
- **Transformer encoder blocks** to model long-range dependencies and temporal context

This modular architecture serves as the foundation for the CP, BBP, and PDP models.

## 📄 Citation

This project is private and intended for research use only.
