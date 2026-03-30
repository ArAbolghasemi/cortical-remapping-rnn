# cortical-remapping-rnn
A lightweight repository for training and analyzing recurrent neural network (RNN) models for population BCI-style control tasks.

---

## Overview

This repo is organized around a simple workflow:

- define and run an RNN model
- decode population activity into a cursor-like control signal
- generate trial/task inputs
- train the model with configurable losses
- analyze geometry, trajectories, and remapping behavior
- visualize training and analysis outputs

The code is mainly organized inside the `core/` folder, with test scripts in `scripts/` and exploratory analysis in `notebooks/`.

---

## Repository structure

state-bci-rnn/
├── core/
│   ├── __init__.py\\
│   ├── rnn.py
│   ├── decoder.py
│   ├── task.py
│   ├── losses.py
│   ├── trainer.py
│   ├── analysis.py
│   └── ploting.py
├── scripts/
│   └── ... test codes
├── notebooks/
│   └── ... analysis notebooks
├── requirements.txt
├── pyproject.toml
└── README.md

---

## Core modules

### core/rnn.py

Defines the main recurrent network model.

Main components:

- RNNConfig
- MotorCortexRNN

Implements a continuous-time firing-rate RNN using Euler discretization.

---

### core/decoder.py

Defines the population decoder that converts neural population activity into a scalar cursor value.

Main components:

- BCIDecoderConfig
- PopulationBCIDecoder

The decoder projects activity onto a unit-norm axis defined over a subset of neurons.

---

### core/task.py

Generates trial inputs and defines the epoch structure of each trial.

Main components:

- TrialInputConfig
- generate_trial_inputs

Trials contain three epochs:

- baseline (holding the cursor at zero)
- task (ramping up to a target value)
- late (keeping the target value)

Each epoch uses a constant input vector with optional noise.

---

### core/losses.py

Defines cursor targets and training loss functions.

Main components:

- CursorTargetConfig
- make_cursor_target
- weighted_cursor_mse

Targets can differ across epochs, and each epoch can have different loss weights.

---

### core/trainer.py

Implements the main training loop.

Main components:

- TrainerConfig
- TrainHistory
- BCITrainer

Responsibilities include:

- generating trials
- running forward passes through the RNN
- decoding cursor outputs
- computing loss
- updating parameters
- storing training history and snapshots

The trainer supports multiple input scaling modes:

- "none" -> no scaling the inputs
- "cursor_mean" -> scale inputs by the mean cursor value across trials
- "cursor_prev" -> scale inputs by the previous time step's cursor value in each trial

---

### core/analysis.py

Provides utilities for analyzing neural population geometry and dynamics.

Examples of included analyses:

- PCA dimensionality estimation
- trajectory variance
- manifold alignment
- principal angle computation
- endpoint cloud analysis
- trajectory–decoder alignment

These tools allow studying how population geometry evolves during learning and decoder remapping.

---

### core/ploting.py

Visualization utilities for training and analysis.

Examples include:

- training loss curves
- cursor vs target trajectories
- epoch-wise cursor statistics
- geometry metrics across training
- decoder remapping visualizations

---

## Installation

Install dependencies:

pip install -r requirements.txt

Editable install:

pip install -e .

## Tip for commitng the notebook:
install nbstripout and have it run automatically whenever committing notebook changes to the (local) git repository. To install nbstripout for use in the repository, navigate to your local copy and call (again assuming the use of uv)
pip install nbstripout

Then install nbstripout as a git filter to your local repository using

nbstripout --install

You can check its installation by
nbstripout --status

---

## Quick start

Example import:

from core import (
    RNNConfig,
    MotorCortexRNN,
    BCIDecoderConfig,
    PopulationBCIDecoder,
    TrialInputConfig,
    CursorTargetConfig,
    TrainerConfig,
    BCITrainer,
)

---

## Example setup

from core import *

# RNN
rnn_cfg = RNNConfig(
    n_inp=10,
    n_rec=200,
)

rnn = MotorCortexRNN(rnn_cfg)

# Decoder
decoder_cfg = BCIDecoderConfig(
    n_rec=200,
    n_bci=50,
)

decoder = PopulationBCIDecoder(decoder_cfg)

# Trial configuration
trial_cfg = TrialInputConfig()

# Target configuration
target_cfg = CursorTargetConfig()

# Trainer configuration
trainer_cfg = TrainerConfig()

trainer = BCITrainer(
    rnn=rnn,
    decoder=decoder,
    trial_cfg=trial_cfg,
    target_cfg=target_cfg,
    trainer_cfg=trainer_cfg,
)

---

## Typical workflow

A typical experiment follows these steps:

1. Initialize the RNN model and decoder.
2. Train the model to control the cursor along an initial decoder axis.
3. Remap (perturb) the decoder axis.
4. Continue training to study adaptation.
5. Analyze neural population geometry before and after remapping.

---

## Scripts

The `scripts/` folder contains runnable experiments and testing code.

Examples:

- quick model tests
- training experiments
- decoder remapping experiments
- figure generation

Example:

python scripts/test_rnn.py

---

## Notebooks

The `notebooks/` folder contains exploratory analyses such as:

- runing the input vs recurrent palastic model comparison
- analyzing the geometry of the input vs recurrent plasticity models
- (To be continued with more notebooks...)
---

## Future improvements

This is fully moudular and extensible, but currently only includes a basic set of features. Future improvements could include especially in the following areas:
- more complex trial structures and input patterns
- additional loss functions and training objectives
- add more biologically realistic RNN models (e.g., spiking networks with possion noise)
- diffrent RNN architectures (e.g., LSTM, GRU)
## License

Add your preferred license here (e.g., MIT).


TODO: make the noise Poiss
