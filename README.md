# Plurigaussian Random Field Scattering Analysis

A Python-based toolkit for simulating and analyzing plurigaussian random fields (PGRF) with applications in scattering intensity calculations. This project provides tools for data processing, PGRF simulation, and machine learning model training.

## Overview

This project implements a plurigaussian random field approach for analyzing three-phased hierarchical systems. It includes functionality for:
- Data preprocessing and normalization
- PGRF simulation and intensity calculations
- Deep learning models for forward and inverse problems
- Ensemble model support
- Dataset generation for machine learning
- Custom PyTorch dataset handling
- Curve fitting and parameter optimization
- Data interpolation for scattering data

[Previous sections remain the same until Installation]

## Installation

### Prerequisites
```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Key dependencies:
- PyTorch 2.0.1 (CUDA 11.8)
- NumPy 1.23.5
- SciPy 1.10.1
- Pandas 2.0.1
- Optuna 3.1.1
- Matplotlib 3.7.1
- configobj 5.0.8
- tqdm 4.65.0

## Project Structure

```
project/
├── configs/
│   ├── params.config          # Training configuration parameters
│   └── pgrf_params.json       # PGRF-specific parameters
│
├── data/
│   ├── train/
│   ├── validate/
│   └── test/
│
├── measurement_data/
│   ├── Cell2Curve5.txt       # Reference measurement data
│   ├── Cell2Full.txt         # Full cell measurement data
│   └── data_generation_reference.txt
│
├── scripts/
│   ├── forward_curve_fit.py
│   ├── forward_train_with_fwd_generated_data.py
│   ├── forward_train.py
│   ├── interpolate_scattering_data.py
│   ├── inverse_curve_fit.py
│   ├── inverse_train_validate_fwd_generated_data.py
│   ├── inverse_train.py
│   └── test_trained_model.py
│
├── utils/
│   ├── __pycache__/
│   ├── combine_csv.py
│   ├── config_reader.py
│   ├── custom_dataset.py
│   ├── data_generation.py
│   ├── data_pipeline.py
│   ├── models.py
│   ├── performance_plotting.py
│   ├── plurigaussianrandomfield.py
│   ├── split_dataset.py
│   └── train.py
│
├── README.md
├── LICENSE.md            
└── requirements.txt
```

### Directory Structure Overview

- `configs/`: Configuration files for model training and PGRF parameters
- `data/`: Project documentation and license
- `measurement_data/`: Raw experimental data files
- `scripts/`: Main execution scripts for training and evaluation
  - Forward model training and fitting
  - Inverse model training and validation
  - Data interpolation utilities
  - Model testing scripts
- `utils/`: Core implementation modules
  - Data processing and pipeline utilities
  - Model architectures and training functions
  - Performance evaluation and plotting
  - Dataset handling and management

[Rest of the README remains the same]

## Neural Network Models

### Forward CNN
The project implements several CNN architectures for forward modeling:
- `ForwardCNN`: Main architecture for parameter-to-intensity prediction
- `inverseCNN`: Inverse architecture for intensity-to-parameter prediction
- `EnsembleModel`: Combines multiple models for improved predictions

Model architecture features:
- Transposed convolutions for upsampling
- Multiple convolutional layers with LeakyReLU activation
- Dropout layers (p=0.5) for regularization
- MaxPooling layers for dimension reduction
- Configurable input and output sizes

### Training

```python
from models import ForwardCNN
from train import train

# Initialize model
model = ForwardCNN(input_size, output_size)

# Configure training
optimizer = optim.Adam(model.parameters(), lr=8e-4)
scheduler = StepLR(optimizer, step_size=4, gamma=0.5)
loss_function = torch.nn.L1Loss()

# Train
training_history, validation_history = train(
    model, train_dataloader, validation_dataloader,
    num_epochs, optimizer, loss_function, scheduler,
    logdir, model_name
)
```

The training process includes:
- Custom accuracy metrics
- Validation monitoring
- Model checkpointing
- Learning rate scheduling
- Training history logging

## Curve Fitting

The project includes an advanced curve fitting module using Optuna for hyperparameter optimization:

```python
from forward_curve_fit import objective

# Create and optimize study
study = optuna.create_study()
study.optimize(lambda trial: objective(trial, model, target), n_trials=10000)

# Get best parameters
best_params = study.best_params
```

Fitting features:
- Weighted MSE loss for targeted fitting
- Parameter range optimization
- Multiple model parameter configurations (4, 7, 8, or 11 parameters)
- Ensemble model support
- Result visualization and logging

## Data Interpolation

The project provides tools for interpolating scattering data to match model requirements:

```python
from interpolate_scattering_data import interpolate_data

interpolated_y = interpolate_data(
    input_file="data.txt",
    output_file="interpolated.txt",
    target_x_values=model_x_values
)
```

Features:
- Cubic interpolation
- Support for standard q-vector values
- Automated file handling
- Error checking
## Data Setup

The simulated PGRF data (~4GB) is not included in the repository due to size constraints. The data is hosted on Zenodo and can be downloaded automatically using the provided script.

### Downloading the Data
```bash
# Install the download script dependencies
pip install requests tqdm

# Run the download script
python scripts/download_data.py
```

The script will:
1. Create a `data/` directory if it doesn't exist
2. Download the PGRF dataset from Zenodo
3. Extract the files to the correct location
4. Clean up temporary files

### Data Files
After running the download script, the `data/` directory will contain:
- Training dataset
- Validation dataset
- Test dataset

Note: The `data/` directory is included in `.gitignore` to prevent accidentally committing large files.

### Manual Data Generation
Alternatively, you can generate the PGRF data yourself using:
```bash
python utils/data_generation.py
```
Note: Data generation may take several weeks depending on your system specifications.

### Data Access Issues
If you encounter any issues accessing or downloading the data, please:
1. Check your internet connection
2. Verify you have sufficient disk space (~4GB required)
3. Ensure you have proper permissions to write to the data directory
4. Contact the repository maintainers if problems persist
## Configuration

The project uses two main configuration files:

1. `params.config`: Training parameters
   - Model architecture settings
   - Training hyperparameters
   - File paths and names
   - Batch sizes
   - Number of epochs

2. `pgrf_params.json`: PGRF parameters
   - Physical constants
   - Range parameters
   - Model-specific settings

### Key Parameters

- `int_const`: Intensity constant
- `phi_A`: Volume fraction of phase A (range: [0, 0.33])
- `vol_factor`: Ratio between volume fraction of phase S and A (range: [0, 2])
- `beta_angle`: Correlation between phase A and S (range: [30, 90])
- `l_z`: Peak position of phase A (range: [0.1, 3])
- `l_y`: Peak position of phase B (range: [3, 10])
- `b`: Correlation parameter (range: [0.1, 10])
- `porod_const`: Prefactor for Porod decay (range: [0, 1])

## Advanced Features

- Ensemble modeling with model averaging
- GAN training support
- Denoising capabilities
- Custom loss functions
- Multiple training modes (standard, denoising, GAN)
- Comprehensive performance plotting
- Automated data preprocessing pipeline

## License

MIT License

Copyright (c) 2023 Jean-Marc von Mentlen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributing

Code by Jean-Marc von Mentlen, ETH Zürich, 2024

## Citations

If you use this code in your research, please cite:
[Add relevant citations here]