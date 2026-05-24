# VolDy-VAE

Official implementation of **VolDy-VAE** for probabilistic time series forecasting.

This repository accompanies the paper:

**Beyond Static Uncertainty: Modeling Temporal Uncertainty Dynamics for Probabilistic Time Series Forecasting**

VolDy-VAE is a Volatility Dynamics Variational Autoencoder that models not only the predictive location and scale, but also how uncertainty evolves across the forecast horizon. The model uses a location-scale decoder with a recurrent volatility dynamics module, allowing confidence intervals to expand in volatile regimes and contract in stable periods.

## Overview

Real-world time series often exhibit temporally persistent uncertainty: volatility clusters, regime shifts, and structural breaks. Standard VAE-based forecasting models usually rely on MSE-style objectives or memoryless variance heads, which can make uncertainty estimates temporally incoherent and can let high-variance observations dominate the location predictor.

VolDy-VAE addresses this by combining:

- **Patch-based variational encoding** for efficient local context extraction.
- **Non-autoregressive latent dynamics** for direct long-horizon generation.
- **Location-scale decoding** to jointly predict the mean and scale of the future distribution.
- **GRU-based volatility dynamics** to maintain a dedicated volatility hidden state across time.
- **Heteroscedastic Gaussian NLL training** for volatility-aware reconstruction and prediction.

## Model

The decoder predicts a Gaussian distribution at each time step:

```text
y_t ~ N(mu_t, sigma_t^2)
```

where `mu_t` is produced by the location head and `sigma_t` is produced by a recurrent scale head:

```text
h_t^vol = GRU_vol(h_{t-1}^vol, z_t)
sigma_t = Softplus(Linear(h_t^vol)) + eps
```

This recurrent scale head models volatility clustering, regime persistence, and smooth transitions between uncertainty regimes. The resulting objective applies inverse-variance weighting through the Gaussian NLL, reducing the influence of high-volatility observations on the location estimate while preserving explicit uncertainty estimates.

## Files

| File | Description |
| --- | --- |
| `VolDy_VAE_nn.py` | Core VolDy-VAE architecture, including the volatility dynamics module. |
| `VolDy_VAE_Forecaster.py` | ProbTS forecaster wrapper with training, loss, and inference logic. |
| `VolDy_VAE.yaml` | Example configuration for long-term probabilistic forecasting. |
| `run.sh` | Example script for running multiple prediction horizons. |

## Installation

This implementation is designed to be used with **ProbTS** and the **K2VAE** code structure.

Place the files in the corresponding ProbTS modules:

```text
probts/model/nn/prob/k2VAE/VolDy_VAE_nn.py
probts/model/forecaster/prob_forecaster/VolDy_VAE_Forecaster.py
config/ltsf/<dataset>/VolDy_VAE.yaml
```

Then install the dependencies required by your ProbTS/K2VAE environment, including PyTorch and einops.

## Usage

Edit `run.sh` to select the GPU, dataset path, dataset name, and output directory, then run:

```bash
bash run.sh
```

The example script evaluates four long-term forecasting horizons:

```text
96, 192, 336, 720
```

## Configuration

Important options in `VolDy_VAE.yaml` include:

| Option | Description |
| --- | --- |
| `patch_len` | Patch length used by the encoder. |
| `dynamic_dim` | Latent dimension. |
| `hidden_layers` | Number of MLP encoder/decoder layers. |
| `hidden_dim` | Hidden dimension for MLP blocks. |
| `vol_hidden_dim` | Hidden dimension of the GRU volatility dynamics module. |
| `weight_beta` | KL-divergence weight in the VAE objective. |
| `sample_schedule` | Sampling schedule parameter used by the forecaster. |

## Paper Summary

VolDy-VAE formalizes **Temporal Uncertainty Dynamics**: faithful probabilistic forecasting should model what will happen, how uncertain it is, and how that uncertainty evolves over time.

The paper shows that:

- MSE-based objectives impose a homoscedastic assumption and can be statistically inefficient under regime-switching heteroscedasticity.
- Feed-forward variance heads can model time-varying scale, but often lack a dedicated memory mechanism for volatility evolution.
- A recurrent volatility dynamics module provides temporally coherent scale estimates.
- VolDy-VAE improves accuracy and calibration across nine real-world forecasting benchmarks.
- The VolDy principle can also benefit GAN, Koopman VAE, and Transformer backbones in plug-in studies.

## Acknowledgments

This implementation builds on ideas and infrastructure from:

- ProbTS: unified probabilistic time series forecasting framework.
- K2VAE: Koopman-based VAE for probabilistic time series forecasting.
- RevIN: reversible instance normalization for non-stationary time series.

