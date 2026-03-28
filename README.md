# WPINN: Windkessel Physics Informed Neural Network

This repository contains implementations of the **Windkessel Physics Informed Neural Network (WPINN)** framework, designed to estimate patient-specific cardiovascular parameters and blood pressure waveforms.

## 🔍 Overview

The WPINN framework leverages a **baseline neural network architecture** that combines **CNNs** for local feature extraction and **Transformers** for capturing temporal dependencies. This architecture is also adapted to estimate different types of Windkessel model parameters:

- **Constant Parameters (CP)**: Parameters that remain fixed across time
- **Beat-to-Beat Parameters (BBP)**: Dynamic parameters that vary on a per-beat basis
- **Pressure Dependent Parameters (PDP)**: Parameters that adapt in response to pressure changes

## 📁 Repository Structure

```
data\
├── example_data.pkl            # Contains example data with added noise used in notebook analysis
└── simulated_2E_PDP.zip        # Contains simulated data with parameter values for the 2E Windkessel model

models\
├── cnn_transformer_nn.py       # Baseline CNN + Transformer neural network architecture
├── cp_model.py                 # Training model using constant parameters (CP)
├── bbp_model.py                # Training model using beat-to-beat parameters (BBP)
├── pdp_model.py                # Training model using pressure dependent parameters (PDP)
└── conv_model.py               # Training model using conventional (CONV) data driven approach]

utils\
└── preprocessing_functions.py  # Contains functions for pre/post processing

model_analysis.ipynb            # Notebook for running example analysis

requirements.txt                # List of packages used in this repo
```

## 🧠 Neural Network Architecture Details

### `cnn_transformer_nn.py`

This file defines the **baseline neural network architecture**, combining:

- **Convolutional layers (CNN)** to capture input waveform spatial features
- **Transformer encoder blocks** to capture temporal context

This modular architecture serves as the foundation for the CP, BBP, and PDP models.

## 📄 Citation

Add citation
