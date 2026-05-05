# Final Project For CSCI 357: Hybrid GRU-ESN Models for Stock Prediction

**Credits**: `my_engine` is adapted from the one created by Professor Brain King (Bucknell University) for CSCI 357: AI and Nueral Networks.

This project explores neural network architectures for stock-return prediction, with a focus on Echo State Networks (ESNs) and custom GRU-ESN hybrid models. The experiments use historical stock features to predict log returns, then evaluate models by directional accuracy: whether the model correctly predicts the sign of the next price movement.

The core implementation lives in `src/my_engine/model.py`, with training utilities in `src/my_engine/trainer.py` and experimental analysis in `notebooks/Report.ipynb`.

## Architecture Overview

The project includes a general modeling engine with MLP, CNN, recurrent, attention, transformer, ESN, and hybrid recurrent-reservoir models. The most relevant models for this project are the ESN family and the custom deep ESN-gated GRU.

### Echo State Networks

Echo State Networks are reservoir computing models. Instead of learning all recurrent weights through backpropagation, an ESN uses a randomly initialized, fixed recurrent reservoir to transform an input sequence into a rich dynamical representation. Only the final readout layer is trained.

In this project:

- `ESN` builds a single frozen reservoir and fits its readout with ridge regression.
- `DeepESN` stacks multiple ESN reservoir layers and trains one final ridge-regression readout.
- `ESNForest` creates an ensemble of randomly initialized ESNs and DeepESNs, then averages their predictions.

This design makes ESNs relatively fast to train while still giving the model a nonlinear memory of recent sequence dynamics. Because the reservoir is random, performance can vary between runs, especially on noisy financial data.

### Custom Hybrid Model: DeepESNGatedGRU

The custom hybrid architecture is implemented as `DeepESNGatedGRU` in `src/my_engine/model.py`. It combines learned GRU dynamics with fixed ESN reservoirs at each layer.

Each layer contains:

- a frozen ESN reservoir that updates at every timestep,
- an `ESNGatedGRUCell`,
- a learned projection from reservoir state to GRU hidden-state size,
- a learned `esn_gate` that decides how much reservoir information should replace or supplement the GRU hidden state.

At each timestep, the GRU first computes its normal hidden update. The ESN reservoir state is then projected into the GRU hidden space, and a learned gate blends the two:

```text
h_new = (1 - esn_gate) * h_gru + esn_gate * projected_r_t
```

This lets the model learn when to rely on the adaptive recurrent memory of the GRU and when to inject the fixed nonlinear memory provided by the ESN reservoir. In the deep version, each layer receives the full hidden-state sequence from the previous layer, applies its own reservoir and gated GRU cell, and passes the final hidden state through a feed-forward output head.

## Example Results

The report notebook evaluates models using directional accuracy and directional precision. The baseline is always predicting upward price movement.

| Model | Stock | Directional Accuracy | Directional Precision | Baseline Accuracy |
| --- | --- | ---: | ---: | ---: |
| GRU | MSFT | 54.44% | 54.44% | 54.44% |
| ESN / DeepESN | MSFT | 60.00% | 63.27% | 54.44% |
| ESN / DeepESN | NVDA | 57.78% | 58.33% | 46.67% |
| ESN / DeepESN | QCOM | 56.67% | 61.29% | 51.11% |
| DeepESNGatedGRU | MSFT | 53.33% | 56.60% | 54.44% |
| DeepESNGatedGRU | NVDA | 50.00% | 45.45% | 46.67% |
| DeepESNGatedGRU | QCOM | 55.56% | 71.43% | 51.11% |

The ESN models generally achieved the strongest directional accuracy in the reported runs, beating the baseline on MSFT, NVDA, and QCOM. The hybrid `DeepESNGatedGRU` was more mixed: it trailed the ESN models in directional accuracy for these examples, but produced notably high directional precision on QCOM. Because reservoir-based models depend on random initialization, these results should be interpreted as preliminary rather than definitive.
