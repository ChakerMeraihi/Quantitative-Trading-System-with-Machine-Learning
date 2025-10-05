# Financial Market Analysis System

This repository contains a comprehensive framework for financial market analysis using advanced machine learning and generative AI techniques. The system is designed to integrate several key components to enhance trading performance and risk estimation accuracy.

## System Components

- **Market Regime Detection**
  - Hidden Markov Models
  - Gaussian Mixture Models
  - Temporal Convolutional Networks with K-Means
  - UMAP with K-Means
  - Identifies distinct market states

- **Generative AI**
  - TimeGAN
  - Conditional Variational Autoencoders
  - FiLM Transformers
  - Generates synthetic financial data conditioned on detected regimes
  - Augments training datasets for trading strategies
  - Addresses the limited historical data problem in finance

- **Algorithmic Trading**
  - Regime-adaptive strategies
    - Transformer-based momentum
    - LSTM short signals
    - Kalman filter pairs trading

- **Risk Forecasting**
  - Regime-switching GARCH with Extreme Value Theory
  - Quantile Random Forests
  - Estimates Value at Risk

## Key Features

- **Extensive Backtesting**
  - Demonstrates significant improvements in trading performance and risk estimation accuracy compared to baseline methods
- **Mathematical Foundations**
  - Thoroughly presented and analyzed
- **Implementation Details**
  - Detailed explanations provided
- **Empirical Results**
  - Each component's performance is thoroughly evaluated

## Disclaimer

This project is purely theoretical and for educational purposes only. The models and strategies presented are not intended for live trading or financial advice. There are several limitations and potential drawbacks to consider:

## Limitations

- **No Transaction-Cost Layer**: The model does not account for transaction costs, which can significantly impact real-world performance.
- **Overfitting Risk**: Parameters are tuned on a single back-test path, which risks overfitting to that specific historical period.
- **Fixed Regime Count**: The system assumes a fixed number of regimes (3) and instant switches, which may not reflect real-world execution lags.
- **Omitted Fractionally-Differentiated Features**: The model does not include fractionally-differentiated features that could retain price memory without stationarity loss.
- **Time-Series Leakage Prevention**: Vanilla cross-validation is used instead of purged/embargoed k-fold to prevent time-series leakage.

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies.
3. Run the provided scripts to replicate the results.

### Cloning the Repository

```bash
