# -*- coding: utf-8 -*-
"""
Advanced Financial Market Analysis - Algorithmic Trading Strategy and Risk Forecasting Component

This script implements the Algorithmic Trading Strategy and Risk Forecasting components
using the UMAP+KMeans regime detection results. It runs backtests using both
original data (baseline) and augmented data (synthetic features) and enforces
a maximum of one trade per day.

Includes:
1. Transformer-based Momentum Strategy
2. LSTM Short Signal Strategy
3. Kalman Filter Pairs Trading
4. Regime-Adaptive Strategy Portfolio (UMAP-based)
5. REGARCH-EVT Risk Forecasting
6. Quantile Random Forest Risk Forecasting
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import warnings
import joblib
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv1D, Dense, Flatten, Input, Dropout, BatchNormalization, Layer, Reshape,
    LSTM, GRU, Bidirectional, TimeDistributed, Embedding, MultiHeadAttention,
    LayerNormalization, Add, Concatenate, Lambda, RepeatVector
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import arch
from arch.univariate import ARX, GARCH, Normal
from scipy import stats
from scipy.stats import genpareto
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from pykalman import KalmanFilter
import quantstats as qs  # Use quantstats for reporting
import warnings

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Configuration (Updated with user input and relative paths) ---
CONFIG = {
    # Base paths (relative)
    "data_dir": "C:/Users/chake/Documents/ML Project",  # Assuming script runs from project root
    "output_dir": "C:/Users/chake/Documents/ML Project/trading_results",

    # Input files (relative)
    "price_data_path": "C:/Users/chake/Documents/ML Project/processed_data/my_data_full_v2_aligned.csv",
    "feature_data_path": "C:/Users/chake/Documents/ML Project/processed_data/my_data_full_v2_features.csv",
    "regime_labels_path": "C:/Users/chake/Documents/ML Project/regime_detection_results/umap_kmeans_regime_labels.csv",
    "regime_chars_path": "C:/Users/chake/Documents/ML Project/regime_detection_results/umap_kmeans_regime_characteristics.csv",

    # GenAI methods and paths
    "genAi_method": ["cvae", "film_transformer", "timegan"],
    "genai_base_dir": "C:/Users/chake/Documents/ML Project/GenAI/umap_kmeans/",
    "genai_feature_file_pattern": "umap_kmeans_{genAi_method}_synthetic_data_regime_{regime_num}.csv",

    # Regime method
    "regime_method": "umap_kmeans",
    "regime_column": "umap_kmeans",

    # Backtesting parameters
    "asset_to_trade": "SPY",
    "lookback_window": 60,
    "train_test_split": 0.7,
    "rebalance_frequency": 4,
    "max_position": 1,
    "augmentation_factor": 1.0,

    # Risk forecasting parameters
    "var_confidence_level": 0.99,
    "risk_horizon": 10,
    "stress_scenarios": ["2008_crisis", "2020_covid", "synthetic_extreme"],

    # Output file patterns
    "strategy_results_pattern": "{data_type}_{regime_method}_strategy_results.csv",
    "risk_results_pattern": "{data_type}_{regime_method}_risk_results.csv",
    "performance_report_pattern": "{data_type}_{regime_method}_performance_report.html"
}

# Create output directory if it doesn't exist
if not os.path.exists(CONFIG["output_dir"]):
    os.makedirs(CONFIG["output_dir"])
    print(f"Created results directory: {CONFIG['output_dir']}")


# --- Data Loading ---
def load_data(config):
    """
    Load price data, feature data, and regime labels.
    """
    print(f"Loading price data from {config['price_data_path']}...")
    try:
        price_data = pd.read_csv(config['price_data_path'], index_col=0, parse_dates=True)
        print(f"Loaded price data with shape: {price_data.shape}")
        # Calculate returns
        returns = price_data.pct_change()
        # Drop initial NaN row from pct_change
        returns = returns.iloc[1:]
        price_data = price_data.iloc[1:]
        print(f"Calculated returns with shape: {returns.shape}")
    except Exception as e:
        print(f"Error loading price data: {e}")
        return None, None, None, None, None

    print(f"Loading feature data from {config['feature_data_path']}...")
    try:
        feature_data = pd.read_csv(config['feature_data_path'], index_col=0, parse_dates=True)
        print(f"Loaded feature data with shape: {feature_data.shape}")

        # --- Manus Modification: Add cleaning logic from GenAI script ---
        if feature_data is not None:
            print("Cleaning feature data (matching GenAI script)...")
            feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
            nan_count_before = feature_data.isna().sum().sum()
            print(f"Found {nan_count_before} NaN/inf values before filling.")
            feature_data = feature_data.fillna(feature_data.mean())
            nan_count_after = feature_data.isna().sum().sum()
            print(f"Found {nan_count_after} NaN values after filling with mean.")

            if feature_data.isna().sum().sum() > 0 or np.isinf(feature_data.values).any():
                print("Warning: Some NaN or infinity values remain after mean fill. Applying aggressive cleaning...")
                nan_cols = feature_data.columns[feature_data.isna().mean() > 0.1]
                if len(nan_cols) > 0:
                    print(f"Dropping {len(nan_cols)} columns with >10% NaN values: {list(nan_cols)}")
                    feature_data = feature_data.drop(columns=nan_cols)
                feature_data = feature_data.fillna(0)  # Fill remaining NaNs with 0
                feature_data = feature_data.replace([np.inf], 1e10)  # Replace remaining inf
                feature_data = feature_data.replace([-np.inf], -1e10)  # Replace remaining -inf
                print("Aggressive cleaning applied.")

            print(f"Feature data cleaned. New shape: {feature_data.shape}")
        # --- End Manus Modification ---

    except Exception as e:
        print(f"Error loading feature data: {e}")
        # Continue without features if needed, but DL models will fail
        feature_data = None

    print(f"Loading regime labels from {config['regime_labels_path']}...")
    try:
        regime_labels = pd.read_csv(config['regime_labels_path'], index_col=0, parse_dates=True)
        # Ensure regime column name matches config
        if config['regime_column'] not in regime_labels.columns:
            raise ValueError(f"Regime column '{config['regime_column']}' not found in labels file.")
        print(f"Loaded regime labels with shape: {regime_labels.shape}")
    except Exception as e:
        print(f"Error loading regime labels: {e}")
        return price_data, returns, feature_data, None, None

    print(f"Loading regime characteristics from {config['regime_chars_path']}...")
    try:
        regime_chars = pd.read_csv(config['regime_chars_path'])
        # Create mapping from regime number to label (e.g., 0 -> 'Bullish')
        regime_mapping = regime_chars.set_index('Regime')['Label'].to_dict()
        print(f"Loaded regime mapping: {regime_mapping}")
    except Exception as e:
        print(f"Error loading regime characteristics: {e}")
        return price_data, returns, feature_data, regime_labels, None

    # Align data
    common_idx = price_data.index.intersection(regime_labels.index)
    if feature_data is not None:
        common_idx = common_idx.intersection(feature_data.index)

    price_data = price_data.loc[common_idx]
    returns = returns.loc[common_idx]
    regime_labels = regime_labels.loc[common_idx]
    if feature_data is not None:
        feature_data = feature_data.loc[common_idx]

    print(
        f"Aligned data. New shapes - Price: {price_data.shape}, Returns: {returns.shape}, Regime: {regime_labels.shape}",
        end="")
    if feature_data is not None:
        print(f", Features: {feature_data.shape}")
    else:
        print()

    return price_data, returns, feature_data, regime_labels, regime_mapping


def load_synthetic_features(config, original_feature_columns):
    """Load synthetic feature data generated by GenAI script."""
    synthetic_features = {}
    genai_dir = config["genai_base_dir"]
    pattern = config["genai_feature_file_pattern"]
    regime_col = config["regime_column"]
    genai_methods = config.get("genAi_method", ["timegan"])  # Default to timegan if not specified

    # Assuming regimes are numbered 0, 1, 2
    for regime_num in range(3):
        regime_features_list = []
        for method in genai_methods:
            # Construct file path using the specific GenAI method
            specific_pattern = pattern.format(genAi_method=method, regime_num=regime_num)
            file_path = os.path.join(genai_dir, specific_pattern)

            if os.path.exists(file_path):
                try:
                    # Load synthetic data - DO NOT use index_col=0 as files have no index
                    df = pd.read_csv(file_path, parse_dates=False)
                    print(f"Loaded raw synthetic data {file_path} with shape: {df.shape}")

                    # --- Manus Modification: Remove last two columns (regime label, genai method) ---
                    if df.shape[1] > 2:  # Check if there are columns to remove
                        df = df.iloc[:, :-2]
                        print(f"Removed last 2 columns. New shape: {df.shape}")
                    else:
                        print(f"Warning: Synthetic data {file_path} has <= 2 columns. Cannot remove last two.")
                    # --- End Manus Modification ---

                    # Ensure column names match original features
                    if len(df.columns) == len(original_feature_columns):
                        df.columns = original_feature_columns  # Assign correct names
                        regime_features_list.append(df)
                        print(
                            f"Loaded synthetic features for regime {regime_num} (method: {method}). Shape: {df.shape}")
                    else:
                        # --- Manus Modification: Robust reindexing for column mismatch ---
                        print(
                            f"Warning: Column count mismatch for regime {regime_num} (method: {method}). Expected {len(original_feature_columns)}, got {len(df.columns)}. Attempting to reindex...")
                        try:
                            # Identify missing and extra columns
                            expected_cols = set(original_feature_columns)
                            # Assign temporary numbered columns if headers are missing/wrong
                            if not all(isinstance(c, str) for c in df.columns):
                                print("  Warning: Non-string headers found, assigning temporary names.")
                                df.columns = [f"col_{i}" for i in range(len(df.columns))]

                            loaded_cols = set(df.columns)
                            missing_cols = list(expected_cols - loaded_cols)
                            extra_cols = list(loaded_cols - expected_cols)

                            if extra_cols:
                                print(f"  Found unexpected extra columns: {extra_cols}. Dropping them.")
                                df = df.drop(columns=extra_cols)

                            # Reindex to match expected columns, filling missing ones with 0
                            print(
                                f"  Reindexing to match {len(original_feature_columns)} expected columns. Missing: {missing_cols}")
                            df = df.reindex(columns=original_feature_columns, fill_value=0.0)

                            # Verify reindex worked
                            if len(df.columns) == len(original_feature_columns):
                                print(f"  Successfully reindexed to {len(df.columns)} columns.")
                                regime_features_list.append(df)
                                print(
                                    f"Loaded and reindexed synthetic features for regime {regime_num} (method: {method}). Final shape: {df.shape}")
                            else:
                                print(f"  Error: Reindexing failed. Still have {len(df.columns)} columns. Skipping.")

                        except Exception as reindex_e:
                            print(f"  Error during reindexing: {reindex_e}. Skipping.")
                        # --- End Manus Modification ---
                except Exception as e:
                    print(
                        f"Error processing synthetic features for regime {regime_num} (method: {method}) from {file_path}: {e}")
            else:
                print(
                    f"Warning: Synthetic feature file not found for regime {regime_num} (method: {method}): {file_path}")

        if regime_features_list:
            synthetic_features[regime_num] = pd.concat(regime_features_list)
            print(
                f"Combined synthetic features for regime {regime_num}. Total shape: {synthetic_features[regime_num].shape}")
        else:
            print(f"No synthetic features loaded for regime {regime_num}.")

    return synthetic_features


# --- Data Preparation for DL Models ---
def prepare_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i])  # Assuming target is the next step
    return np.array(X), np.array(y)


# --- Algorithmic Trading Strategy Component ---
# (Keeping original strategy implementations: Transformer, LSTM, Kalman)

def create_transformer_momentum_strategy(input_shape, num_heads=4, d_model=64, dropout_rate=0.2):
    """Create a Transformer-based momentum strategy model."""
    print("Creating Transformer Momentum Strategy model...")
    inputs = Input(shape=input_shape)
    x = Dense(d_model)(inputs)
    seq_length = input_shape[0]

    # Simplified Positional Encoding Layer
    position = tf.range(start=0, limit=seq_length, delta=1)
    pos_encoding = Embedding(input_dim=seq_length, output_dim=d_model)(position)
    x = x + pos_encoding

    for _ in range(2):  # 2 transformer blocks
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
        attn_output = Dropout(dropout_rate)(attn_output)  # Added dropout
        attn_output = Add()([x, attn_output])
        attn_output = LayerNormalization(epsilon=1e-6)(attn_output)
        ffn_output = Dense(d_model * 2, activation='relu')(attn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        ffn_output = Dense(d_model)(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)  # Added dropout
        ffn_output = Add()([attn_output, ffn_output])
        x = LayerNormalization(epsilon=1e-6)(ffn_output)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = Dense(1)(x)  # Predicting next period's return for one asset
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


def create_lstm_short_strategy(input_shape, lstm_units=64, dropout_rate=0.2):
    """Create an LSTM-based short signal strategy model."""
    print("Creating LSTM Short Strategy model...")
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(lstm_units // 2, return_sequences=False),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='tanh')  # tanh activation for short signal (-1 to 1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


class KalmanPairsStrategy:
    """Kalman Filter-based pairs trading strategy."""

    def __init__(self, window_size=60):
        self.window_size = window_size
        self.pairs = []
        self.kf_models = {}
        self.hedge_ratios = {}
        self.spreads = {}
        self.spread_means = {}
        self.spread_stds = {}

    def find_pairs(self, price_data, threshold=0.05):
        """Find cointegrated pairs of assets."""
        print("Finding cointegrated pairs...")
        n = price_data.shape[1]
        tickers = price_data.columns
        pairs = []
        # Limit pairs search for efficiency if too many assets
        max_assets_to_check = min(n, 50)
        print(f"Checking first {max_assets_to_check} assets for pairs.")
        for i in range(max_assets_to_check):
            for j in range(i + 1, max_assets_to_check):
                asset1 = tickers[i]
                asset2 = tickers[j]
                try:
                    series1 = price_data[asset1].dropna()
                    series2 = price_data[asset2].dropna()
                    common_idx = series1.index.intersection(series2.index)
                    if len(common_idx) < self.window_size: continue  # Skip if not enough overlapping data
                    series1 = series1.loc[common_idx]
                    series2 = series2.loc[common_idx]
                    score, pvalue, _ = coint(series1, series2)
                    if pvalue < threshold:
                        pairs.append((asset1, asset2))
                        print(f"Found cointegrated pair: {asset1} - {asset2} (p-value: {pvalue:.4f})")
                except Exception as e:
                    print(f"Error testing pair {asset1}-{asset2}: {e}")
        self.pairs = pairs
        print(f"Found {len(self.pairs)} pairs.")
        return pairs

    def train(self, price_data):
        """Train Kalman Filter models for each pair."""
        print("Training Kalman Filter models...")
        for asset1, asset2 in self.pairs:
            try:
                series1 = price_data[asset1].dropna()
                series2 = price_data[asset2].dropna()
                common_idx = series1.index.intersection(series2.index)
                if len(common_idx) < 2: continue
                series1 = series1.loc[common_idx].values
                series2 = series2.loc[common_idx].values

                n_dim_state = 2
                n_dim_obs = 1
                obs_matrices = np.zeros((len(series1), n_dim_obs, n_dim_state))
                for i in range(len(series1)):
                    obs_matrices[i, 0, 0] = series1[i]
                    obs_matrices[i, 0, 1] = 1.0

                kf = KalmanFilter(
                    n_dim_state=n_dim_state, n_dim_obs=n_dim_obs,
                    initial_state_mean=np.zeros(n_dim_state),
                    initial_state_covariance=np.eye(n_dim_state),
                    transition_matrices=np.eye(n_dim_state),
                    observation_matrices=obs_matrices,
                    observation_covariance=1.0,  # Simplified
                    transition_covariance=0.01 * np.eye(n_dim_state)
                )
                state_means, state_covs = kf.em(series2.reshape(-1, 1), n_iter=5).filter(series2.reshape(-1, 1))
                self.kf_models[(asset1, asset2)] = kf
                self.hedge_ratios[(asset1, asset2)] = state_means
                spread = series2 - (series1 * state_means[:, 0] + state_means[:, 1])
                self.spreads[(asset1, asset2)] = spread
                self.spread_means[(asset1, asset2)] = np.mean(spread)
                self.spread_stds[(asset1, asset2)] = np.std(spread)
            except Exception as e:
                print(f"Error training Kalman for pair {asset1}-{asset2}: {e}")
        return self.kf_models

    def generate_signals(self, price_data, z_threshold=1.5):
        """Generate trading signals for each pair based on latest data point."""
        # Returns signals for the last time step in price_data
        signals = {}
        if price_data.empty or len(self.pairs) == 0:
            return pd.Series(0.0, index=['pairs_signal'])  # Return zero signal if no data or pairs

        latest_prices = price_data.iloc[-1]
        total_signal = 0.0
        active_pairs = 0

        for asset1, asset2 in self.pairs:
            if (asset1, asset2) not in self.hedge_ratios or asset1 not in latest_prices or asset2 not in latest_prices:
                continue
            try:
                hedge_ratio = self.hedge_ratios[(asset1, asset2)][-1, 0]
                intercept = self.hedge_ratios[(asset1, asset2)][-1, 1]
                spread_mean = self.spread_means[(asset1, asset2)]
                spread_std = self.spread_stds[(asset1, asset2)]

                if spread_std == 0: continue  # Avoid division by zero

                current_spread = latest_prices[asset2] - (latest_prices[asset1] * hedge_ratio + intercept)
                z_score = (current_spread - spread_mean) / spread_std

                # Generate signal based on z-score crossing threshold
                if z_score > z_threshold:
                    total_signal -= 1  # Short the spread (short asset2, long asset1)
                elif z_score < -z_threshold:
                    total_signal += 1  # Long the spread (long asset2, short asset1)
                # else: signal remains 0 (no trade)
                active_pairs += 1
            except Exception as e:
                print(f"Error generating signal for pair {asset1}-{asset2}: {e}")

        # Average signal across active pairs (simple approach)
        final_signal = total_signal / active_pairs if active_pairs > 0 else 0.0
        return pd.Series(final_signal, index=['pairs_signal'])


# --- Regime-Adaptive Strategy Portfolio ---
class RegimeAdaptiveStrategyPortfolio:
    """Combines multiple strategies based on market regime."""

    def __init__(self, regime_mapping, config):
        self.regime_mapping = regime_mapping
        self.config = config
        self.strategies = {}
        self.regime_strategy_map = {
            'Bullish': ['transformer_momentum', 'kalman_pairs'],
            'Bearish': ['lstm_short', 'kalman_pairs'],
            'Risky/Neutral': ['kalman_pairs']  # No trading in risky/neutral -> MODIFIED: Enable Kalman Pairs
        }
        self.weights = {}
        self.trained_models = {}
        self.scalers = {}
        self.kalman_strategy = None

    def train_strategies(self, train_features, train_prices, train_regimes, synthetic_features=None,
                         use_augmented=False):
        """Train individual strategy models."""
        print(f"Training strategies (Augmented: {use_augmented})...")
        lookback = self.config['lookback_window']
        asset = self.config['asset_to_trade']

        # Prepare training data (potentially augmented)
        X_train_full, y_train_full = None, None
        if train_features is not None:
            print("Scaling features...")
            scaler = StandardScaler()  # Use StandardScaler for DL models
            train_features_scaled = scaler.fit_transform(train_features)
            self.scalers['features'] = scaler

            # Prepare sequences
            X_train_seq, y_train_seq = prepare_sequences(train_features_scaled, lookback)
            # Target for momentum is the next return of the main asset
            y_train_returns = train_prices[asset].pct_change().iloc[lookback:].values.reshape(-1, 1)
            # Target for short signal is also based on returns (e.g., negative return -> short)
            # Simple target: -1 if next return < 0, 1 otherwise (or use tanh activation)
            y_train_short_signal = np.sign(y_train_returns)  # Simple sign as target

            # Augment data if requested and available
            if use_augmented and synthetic_features:
                print("Augmenting training data...")
                augmented_X, augmented_y_returns, augmented_y_short = [], [], []
                for regime_num, synth_df in synthetic_features.items():
                    if not synth_df.empty:
                        # Scale synthetic features using the same scaler
                        synth_df_scaled = self.scalers['features'].transform(synth_df)
                        # Prepare sequences for synthetic data
                        synth_X, _ = prepare_sequences(synth_df_scaled, lookback)
                        # Find original data points corresponding to this regime
                        regime_indices = train_regimes.iloc[lookback:][self.config['regime_column']] == regime_num
                        if regime_indices.sum() > 0 and len(synth_X) > 0:
                            # Sample synthetic sequences to match original count (or use augmentation factor)
                            num_samples = min(len(synth_X),
                                              int(regime_indices.sum() * self.config['augmentation_factor']))
                            sample_indices = np.random.choice(len(synth_X), num_samples, replace=False)
                            augmented_X.append(synth_X[sample_indices])
                            # Use original targets corresponding to the regime
                            augmented_y_returns.append(y_train_returns[regime_indices][:num_samples])
                            augmented_y_short.append(y_train_short_signal[regime_indices][:num_samples])

                if augmented_X:
                    augmented_X = np.concatenate(augmented_X, axis=0)
                    augmented_y_returns = np.concatenate(augmented_y_returns, axis=0)
                    augmented_y_short = np.concatenate(augmented_y_short, axis=0)

                    # Combine original and augmented data
                    X_train_seq = np.concatenate([X_train_seq, augmented_X], axis=0)
                    y_train_returns = np.concatenate([y_train_returns, augmented_y_returns], axis=0)
                    y_train_short_signal = np.concatenate([y_train_short_signal, augmented_y_short], axis=0)
                    print(
                        f"Augmented data shapes - X: {X_train_seq.shape}, y_ret: {y_train_returns.shape}, y_short: {y_train_short_signal.shape}")

            # Train Transformer Momentum
            if 'transformer_momentum' in [s for regime_list in self.regime_strategy_map.values() for s in regime_list]:
                print("Training Transformer Momentum model...")
                transformer_model = create_transformer_momentum_strategy(input_shape=(lookback, X_train_seq.shape[2]))
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                transformer_model.fit(X_train_seq, y_train_returns, epochs=50, batch_size=32, validation_split=0.2,
                                      callbacks=[early_stopping], verbose=1)
                self.trained_models['transformer_momentum'] = transformer_model

            # Train LSTM Short Signal
            if 'lstm_short' in [s for regime_list in self.regime_strategy_map.values() for s in regime_list]:
                print("Training LSTM Short Signal model...")
                lstm_model = create_lstm_short_strategy(input_shape=(lookback, X_train_seq.shape[2]))
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                lstm_model.fit(X_train_seq, y_train_short_signal, epochs=50, batch_size=32, validation_split=0.2,
                               callbacks=[early_stopping], verbose=1)
                self.trained_models['lstm_short'] = lstm_model
        else:
            print("Warning: Feature data not available. Skipping training of Transformer and LSTM models.")

        # Train Kalman Pairs Strategy
        if 'kalman_pairs' in [s for regime_list in self.regime_strategy_map.values() for s in regime_list]:
            print("Training Kalman Pairs Strategy...")
            self.kalman_strategy = KalmanPairsStrategy(window_size=lookback)
            self.kalman_strategy.find_pairs(train_prices)
            self.kalman_strategy.train(train_prices)
            self.trained_models['kalman_pairs'] = self.kalman_strategy  # Store the instance

    def generate_signals(self, current_features, current_prices, current_regime_num):
        """Generate signals from active strategies based on the current regime."""
        lookback = self.config['lookback_window']
        asset = self.config['asset_to_trade']
        signals = pd.Series(0.0, index=['transformer_momentum', 'lstm_short', 'kalman_pairs'])

        # Map regime number to label (e.g., 0 -> 'Bullish')
        current_regime_label = self.regime_mapping.get(current_regime_num, 'Risky/Neutral')
        active_strategies = self.regime_strategy_map.get(current_regime_label, [])

        if not active_strategies:
            return pd.Series(0.0, index=['portfolio_signal'])  # No trading signal

        # Prepare input sequence for DL models
        X_current_seq = None
        if current_features is not None and len(current_features) >= lookback:
            if 'features' in self.scalers:
                current_features_scaled = self.scalers['features'].transform(current_features.iloc[-lookback:])
                X_current_seq = np.expand_dims(current_features_scaled, axis=0)
            else:
                print("Warning: Feature scaler not found. Cannot generate DL signals.")

        # Generate signals from individual strategies
        if 'transformer_momentum' in active_strategies and 'transformer_momentum' in self.trained_models and X_current_seq is not None:
            pred_return = self.trained_models['transformer_momentum'].predict(X_current_seq, verbose=0)[0, 0]
            signals['transformer_momentum'] = 1.0 if pred_return > 0 else 0.0  # Simple momentum signal

        if 'lstm_short' in active_strategies and 'lstm_short' in self.trained_models and X_current_seq is not None:
            pred_signal = self.trained_models['lstm_short'].predict(X_current_seq, verbose=0)[0, 0]
            signals['lstm_short'] = -1.0 if pred_signal < 0 else 0.0  # Only take short signals

        if 'kalman_pairs' in active_strategies and 'kalman_pairs' in self.trained_models:
            if len(current_prices) >= 1:
                # Pass only the latest price data needed for signal generation
                kalman_signal_series = self.trained_models['kalman_pairs'].generate_signals(current_prices.iloc[-1:])
                signals['kalman_pairs'] = kalman_signal_series.iloc[0] if not kalman_signal_series.empty else 0.0
            else:
                signals['kalman_pairs'] = 0.0

        # Combine signals (simple averaging for now)
        active_signals = signals[active_strategies]
        portfolio_signal = active_signals.mean() if not active_signals.empty else 0.0

        # Apply max position constraint
        portfolio_signal = np.clip(portfolio_signal, -self.config['max_position'], self.config['max_position'])

        return pd.Series(portfolio_signal, index=['portfolio_signal'])


# --- Risk Forecasting Component ---
class RiskForecaster:
    """Forecasts risk using REGARCH-EVT and Quantile Random Forest."""

    def __init__(self, config):
        self.config = config
        self.regarch_models = {}
        self.evt_params = {}
        self.qrf_model = None
        self.scaler_risk = None

    def train_regarch_evt(self, returns, regimes):
        """Train REGARCH-EVT model for each regime."""
        print("Training REGARCH-EVT models...")
        regime_col = self.config['regime_column']
        unique_regimes = regimes[regime_col].unique()
        asset_returns = returns[self.config['asset_to_trade']].dropna()

        for regime in unique_regimes:
            print(f"Training for Regime {regime}...")
            regime_returns = asset_returns[regimes[regime_col] == regime]
            if len(regime_returns) < 50:  # Need sufficient data
                print(f"Skipping Regime {regime} due to insufficient data ({len(regime_returns)} points).")
                continue

            try:
                # Fit GARCH(1,1) model (can be extended to REGARCH later)
                model = arch.arch_model(regime_returns * 100, vol='Garch', p=1, q=1, dist='Normal')  # Scale returns
                res = model.fit(disp='off')
                self.regarch_models[regime] = res

                # Fit EVT (GPD) to standardized residuals' tails
                std_resid = res.resid / res.conditional_volatility
                threshold = np.percentile(std_resid, 95)  # Example threshold for upper tail
                exceedances = std_resid[std_resid > threshold] - threshold
                if len(exceedances) > 10:
                    shape, loc, scale = genpareto.fit(exceedances)
                    self.evt_params[regime] = {'shape': shape, 'loc': loc, 'scale': scale, 'threshold': threshold,
                                               'tail_fraction': len(exceedances) / len(std_resid)}
                else:
                    print(f"Warning: Not enough exceedances for EVT fit in Regime {regime}.")
                    self.evt_params[regime] = None
            except Exception as e:
                print(f"Error training REGARCH-EVT for Regime {regime}: {e}")
        return self.regarch_models, self.evt_params

    def forecast_regarch_evt_var(self, latest_returns, current_regime):
        """Forecast VaR using the trained REGARCH-EVT model for the current regime."""
        if current_regime not in self.regarch_models or self.evt_params.get(current_regime) is None:
            # Fallback to simple historical VaR if model not available
            if len(latest_returns) > 20:
                return np.percentile(latest_returns, (1 - self.config['var_confidence_level']) * 100)
            else:
                return -0.05  # Default high risk if insufficient data

        try:
            model_res = self.regarch_models[current_regime]
            evt_p = self.evt_params[current_regime]
            conf_level = self.config['var_confidence_level']
            horizon = self.config['risk_horizon']

            # Forecast volatility
            forecast = model_res.forecast(horizon=1, reindex=False)
            cond_vol_forecast = forecast.variance.iloc[0, 0] ** 0.5 / 100  # Rescale back

            # Calculate VaR using GPD parameters
            tail_prob = (1 - conf_level) / evt_p['tail_fraction']
            gpd_quantile = genpareto.ppf(1 - tail_prob, evt_p['shape'], loc=evt_p['loc'], scale=evt_p['scale'])
            var_std_resid = evt_p['threshold'] + gpd_quantile
            var_forecast = var_std_resid * cond_vol_forecast

            # Adjust for horizon (sqrt(T) rule - approximation)
            var_horizon = var_forecast * np.sqrt(horizon)
            return -var_horizon  # VaR is typically reported as positive loss
        except Exception as e:
            print(f"Error forecasting REGARCH-EVT VaR for Regime {current_regime}: {e}")
            # Fallback
            if len(latest_returns) > 20:
                return np.percentile(latest_returns, (1 - self.config['var_confidence_level']) * 100)
            else:
                return -0.05

    def train_qrf(self, train_features, train_returns):
        """Train Quantile Random Forest for VaR forecasting."""
        print("Training Quantile Random Forest for VaR...")
        if train_features is None or train_returns is None:
            print("Skipping QRF training: Missing features or returns.")
            return

        asset_returns = train_returns[self.config['asset_to_trade']].dropna()
        common_idx = train_features.index.intersection(asset_returns.index)
        train_features = train_features.loc[common_idx]
        asset_returns = asset_returns.loc[common_idx]

        if len(train_features) < 50:
            print("Skipping QRF training: Insufficient data.")
            return

        # Scale features
        self.scaler_risk = StandardScaler().fit(train_features)
        X_train_scaled = self.scaler_risk.transform(train_features)
        y_train = asset_returns.values

        try:
            # Use RandomForestRegressor from sklearn - direct quantile regression is complex
            # We predict the return and estimate quantile from prediction errors or use a dedicated library if available
            # Simple approach: Fit standard RF and use prediction + residual quantile
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train_scaled, y_train)
            self.qrf_model = rf
            # Calculate residuals for quantile estimation
            train_preds = rf.predict(X_train_scaled)
            self.residuals = y_train - train_preds
            print("QRF model trained (using standard RF + residuals).")
        except Exception as e:
            print(f"Error training QRF: {e}")
            self.qrf_model = None

    def forecast_qrf_var(self, latest_features):
        """Forecast VaR using the trained Quantile Random Forest."""
        if self.qrf_model is None or self.scaler_risk is None or latest_features is None or not hasattr(self,
                                                                                                        'residuals'):
            return -0.05  # Default high risk

        if len(latest_features) < 1:
            return -0.05

        try:
            latest_features_scaled = self.scaler_risk.transform(latest_features.iloc[-1:])  # Scale latest features
            pred_return = self.qrf_model.predict(latest_features_scaled)[0]
            # Estimate VaR using predicted return + quantile of historical residuals
            residual_quantile = np.percentile(self.residuals, (1 - self.config['var_confidence_level']) * 100)
            var_forecast = pred_return + residual_quantile
            # Adjust for horizon (approximation)
            var_horizon = var_forecast * np.sqrt(self.config['risk_horizon'])
            return -var_horizon  # Return as positive loss
        except Exception as e:
            print(f"Error forecasting QRF VaR: {e}")
            return -0.05


# --- Backtesting Engine ---
def run_backtest(config, price_data, returns, feature_data, regime_labels, regime_mapping, synthetic_features=None,
                 use_augmented=False):
    """
    Run the backtest for the regime-adaptive portfolio.
    """
    data_type = "augmented" if use_augmented else "baseline"
    print(f"\n--- Running Backtest ({data_type}) ---")

    # Split data
    split_idx = int(len(price_data) * config['train_test_split'])
    train_prices = price_data.iloc[:split_idx]
    test_prices = price_data.iloc[split_idx:]
    train_returns = returns.iloc[:split_idx]
    test_returns = returns.iloc[split_idx:]
    train_regimes = regime_labels.iloc[:split_idx]
    test_regimes = regime_labels.iloc[split_idx:]
    train_features = feature_data.iloc[:split_idx] if feature_data is not None else None
    test_features = feature_data.iloc[split_idx:] if feature_data is not None else None

    print(f"Train period: {train_prices.index.min()} to {train_prices.index.max()} ({len(train_prices)} points)")
    print(f"Test period: {test_prices.index.min()} to {test_prices.index.max()} ({len(test_prices)} points)")

    # Initialize and train portfolio and risk models
    portfolio = RegimeAdaptiveStrategyPortfolio(regime_mapping, config)
    portfolio.train_strategies(train_features, train_prices, train_regimes, synthetic_features, use_augmented)

    risk_forecaster = RiskForecaster(config)
    risk_forecaster.train_regarch_evt(train_returns, train_regimes)
    risk_forecaster.train_qrf(train_features, train_returns)

    # Backtesting loop
    print("Starting backtesting loop...")
    positions = pd.DataFrame(index=test_prices.index, columns=['signal', 'position'])
    risk_forecasts = pd.DataFrame(index=test_prices.index, columns=['regarch_var', 'qrf_var'])
    lookback = config['lookback_window']
    rebalance_freq = config['rebalance_frequency']
    last_position = 0.0
    last_trade_date = None

    for i in range(len(test_prices)):
        current_time = test_prices.index[i]
        current_date = current_time.date()

        # Get data available up to this point (t-1 for features/prices)
        # Need lookback + 1 points ending at index i-1 for feature sequence generation
        start_idx_features = max(0, i - lookback)
        current_feature_window = test_features.iloc[start_idx_features:i] if test_features is not None else None
        # Need prices up to i-1 for Kalman signal generation
        current_price_window = test_prices.iloc[:i]
        # Need returns up to i-1 for risk forecasting
        current_return_window = test_returns.iloc[:i]

        # Get current regime (known at time t)
        current_regime_num = test_regimes.iloc[i][config['regime_column']]

        # Generate signals (using data up to t-1)
        signal_series = portfolio.generate_signals(current_feature_window, current_price_window, current_regime_num)
        current_signal = signal_series['portfolio_signal']

        # --- One Trade Per Day Logic ---
        target_position = current_signal  # Ideal position based on signal
        final_position = last_position  # Default to holding previous position

        # Check if we can trade today
        can_trade_today = (last_trade_date is None) or (current_date > last_trade_date)

        if can_trade_today and target_position != last_position:
            final_position = target_position
            last_trade_date = current_date  # Update last trade date
        # If it's the same day, or signal hasn't changed, keep final_position = last_position

        positions.loc[current_time, 'signal'] = current_signal  # Record raw signal
        positions.loc[current_time, 'position'] = final_position
        last_position = final_position  # Update for next iteration

        # Forecast risk (using data up to t-1)
        risk_forecasts.loc[current_time, 'regarch_var'] = risk_forecaster.forecast_regarch_evt_var(
            current_return_window[config['asset_to_trade']], current_regime_num)
        risk_forecasts.loc[current_time, 'qrf_var'] = risk_forecaster.forecast_qrf_var(current_feature_window)

        if i % 1000 == 0:  # Print progress
            print(f"Backtest progress: {i + 1}/{len(test_prices)} ({current_time})")

    print("Backtesting loop finished.")
    positions = positions.fillna(0)
    risk_forecasts = risk_forecasts.fillna(method='ffill')

    # Calculate strategy returns (using daily resampling for one-trade-per-day)
    print("Calculating strategy returns...")
    # Get daily positions (end of day) - shift(1) to use previous day's position for today's return
    daily_positions = positions['position'].resample('D').last().ffill().shift(1).dropna()
    # Get daily market returns
    daily_market_returns = test_returns[config['asset_to_trade']].resample('D').apply(
        lambda x: (1 + x).prod() - 1).dropna()

    # Align daily positions and returns
    common_daily_idx = daily_positions.index.intersection(daily_market_returns.index)
    daily_positions = daily_positions.loc[common_daily_idx]
    daily_market_returns = daily_market_returns.loc[common_daily_idx]

    # Calculate daily strategy returns
    daily_strategy_returns = daily_positions * daily_market_returns
    daily_strategy_returns.name = f"{data_type}_strategy"

    # Save detailed results
    results_df = pd.concat([positions, risk_forecasts], axis=1).loc[test_prices.index]
    results_path = os.path.join(config['output_dir'], config['strategy_results_pattern'].format(data_type=data_type,
                                                                                                regime_method=config[
                                                                                                    'regime_method']))
    results_df.to_csv(results_path)
    print(f"Saved detailed results to {results_path}")

    # --- Generate QuantStats Report (Print Metrics & Save Plot) ---
    print(f"\n--- Performance Metrics ({data_type}) ---")
    try:
        # Ensure returns are pandas Series with DatetimeIndex
        if not isinstance(daily_strategy_returns.index, pd.DatetimeIndex):
            daily_strategy_returns.index = pd.to_datetime(daily_strategy_returns.index)

        # Print metrics to console
        qs.reports.metrics(daily_strategy_returns, mode='full', display=True)

        # Save cumulative returns plot
        plot_filename = f"{data_type}_{config['regime_method']}_cumulative_returns.png"
        plot_path = os.path.join(config['output_dir'], plot_filename)
        qs.plots.returns(daily_strategy_returns, benchmark=None, savefig=plot_path, show=False)
        print(f"Saved cumulative returns plot to {plot_path}")

    except Exception as e:
        print(f"Error generating QuantStats report/plot: {e}")
        # Print basic stats if quantstats fails
        print("Basic Stats:")
        print(f"Mean Daily Return: {daily_strategy_returns.mean():.6f}")
        print(f"Std Dev Daily Return: {daily_strategy_returns.std():.6f}")
        print(f"Cumulative Return: {(1 + daily_strategy_returns).prod() - 1:.4f}")

    return daily_strategy_returns, results_df


# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.time()

    # Load data
    price_data, returns, feature_data, regime_labels, regime_mapping = load_data(CONFIG)

    if price_data is None or returns is None or regime_labels is None or regime_mapping is None:
        print("Critical data loading failed. Exiting.")
        exit()

    # Load synthetic features (will be empty if files not found or columns mismatch)
    original_feature_cols = feature_data.columns if feature_data is not None else []
    synthetic_features = load_synthetic_features(CONFIG, original_feature_cols)

    # Run Baseline Backtest
    baseline_returns, baseline_results = run_backtest(
        CONFIG, price_data, returns, feature_data, regime_labels, regime_mapping,
        synthetic_features=None, use_augmented=False
    )

    # Run Augmented Backtest (only if synthetic features were loaded)
    augmented_returns, augmented_results = None, None
    if synthetic_features:  # Check if dictionary is not empty
        augmented_returns, augmented_results = run_backtest(
            CONFIG, price_data, returns, feature_data, regime_labels, regime_mapping,
            synthetic_features=synthetic_features, use_augmented=True
        )
    else:
        print("\nSkipping augmented backtest as no valid synthetic features were loaded.")

    # --- Comparison (if augmented run was performed) ---
    if augmented_returns is not None:
        print("\n--- Comparison: Baseline vs Augmented ---")
        comparison_df = pd.DataFrame({
            'Baseline': baseline_returns,
            'Augmented': augmented_returns
        }).dropna()

        print("\n--- Baseline Metrics ---")
        qs.reports.metrics(comparison_df['Baseline'], mode='full', display=True)

        print("\n--- Augmented Metrics ---")
        qs.reports.metrics(comparison_df['Augmented'], mode='full', display=True)

        # Plot comparison
        print("Plotting cumulative returns comparison...")
        fig, ax = plt.subplots(figsize=(12, 6)) # Create figure and axes
        try:
            # Use qs.plots.returns and pass the ax object
            # Note: qs.plots.returns plots cumulative returns from a daily returns series
            qs.plots.returns(comparison_df["Baseline"], show=False, subtitle=False)
            qs.plots.returns(comparison_df["Augmented"], show=False, subtitle=False)

            ax.set_title("Cumulative Returns Comparison: Baseline vs Augmented")
            ax.legend()
            ax.set_ylabel("Cumulative Return (%)") # QuantStats plots in %
            ax.set_xlabel("Date")
            ax.grid(True)

            comp_plot_path = os.path.join(CONFIG["output_dir"], f"{CONFIG["regime_method"]}_baseline_vs_augmented_returns.png")
            fig.savefig(comp_plot_path)
            print(f"Saved comparison plot to {comp_plot_path}") # Moved inside try block
            plt.close(fig) # Close the figure
        except Exception as plot_err:
            print(f"Error during plotting: {plot_err}")
            plt.close(fig) # Ensure figure is closed even if error occurs

