#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GenAI Data Augmentation - Adapted for UMAP Regimes

This script implements four different approaches for generating synthetic financial data:
1. TimeGAN (Time-series Generative Adversarial Network)
2. Conditional Variational Autoencoder (CVAE)
3. FiLM Transformer (Feature-wise Linear Modulation)
4. Regime Adversary Module

The script loads UMAP+KMeans market regime detection results and generates synthetic data for each regime.
Paths are adapted for the sandbox environment.

Features:
- XLA compilation for accelerated execution
- Parallel data processing with tf.data pipeline
- Gradient accumulation for stable training
- Mixed precision training for faster execution
- Optimized batch size and memory management
- Full mathematical rigor for hedge fund applications
- Comprehensive logging and visualization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
import time
import warnings
import joblib
from datetime import datetime
import gc
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv1D, Dense, Flatten, Input, Dropout, BatchNormalization, Layer, Reshape,
    LSTM, GRU, Bidirectional, TimeDistributed, Embedding, MultiHeadAttention,
    LayerNormalization, Add, Concatenate, Lambda, RepeatVector
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=info, 2=warning, 3=error
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

warnings.filterwarnings("ignore")

# Enable mixed precision training for faster execution
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Enable XLA compilation for accelerated execution
tf.config.optimizer.set_jit(True)  # Enable XLA compilation

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Configuration --- #
# Adapted paths for sandbox environment
REGIME_DIR = "C:/Users/chake/Documents/ML Project/regime_detection_results"  # Directory with regime labels
FEATURES_PATH = "C:/Users/chake/Documents/ML Project/processed_data/my_data_full_v2_features.csv"  # Path to the features used for UMAP
BASE_OUTPUT_DIR = "C:/Users/chake/Documents/ML Project/GenAI"  # Base directory for GenAI results
REGIME_METHOD = "umap_kmeans"  # Focus on UMAP results
SEQUENCE_LENGTH = 20  # Full sequence length (for quick test 10 , full 20)
MAX_FEATURES = 215  # Feature count for GenAI models (Number of features to select/generate)
NUM_EPOCHS = 100  # Full training epochs (for quick test 5,full 100)
BATCH_SIZE = 32  # Batch size (for quick test 16, full 32)
GRADIENT_ACCUMULATION_STEPS = 2  # Number of steps to accumulate gradients
# --- End Configuration --- #

# Configure GPU memory growth to prevent OOM errors
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    print(f"Found {len(physical_devices)} GPU(s). Configuring for GPU acceleration.")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("GPU memory growth enabled")
    print("XLA compilation enabled for accelerated execution")
else:
    print("No GPU found. Running on CPU.")


# Memory management function
def clear_memory():
    """Clear memory to prevent OOM errors"""
    gc.collect()
    tf.keras.backend.clear_session()


# Custom Reshape Layer
class ReshapeLayer(Layer):
    def __init__(self, target_shape, **kwargs):
        super(ReshapeLayer, self).__init__(**kwargs)
        self.target_shape = target_shape

    def call(self, inputs):
        return tf.reshape(inputs, self.target_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + self.target_shape[1:]

    def get_config(self):
        config = super(ReshapeLayer, self).get_config()
        config.update({"target_shape": self.target_shape})
        return config


# Custom FiLM Layer for conditioning
class FiLMLayer(Layer):
    def __init__(self, units, **kwargs):
        super(FiLMLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma_dense = Dense(units)
        self.beta_dense = Dense(units)

    def call(self, inputs, condition):
        gamma = self.gamma_dense(condition)
        beta = self.beta_dense(condition)

        # Reshape for broadcasting
        gamma = tf.expand_dims(gamma, axis=1)
        beta = tf.expand_dims(beta, axis=1)

        return inputs * gamma + beta

    def get_config(self):
        config = super(FiLMLayer, self).get_config()
        config.update({"units": self.units})
        return config


# Custom FiLM Conditioning Layer
class FiLMConditioningLayer(Layer):
    def __init__(self, **kwargs):
        super(FiLMConditioningLayer, self).__init__(**kwargs)
        self.gamma_dense = None  # Will be set in build
        self.beta_dense = None  # Will be set in build

    def build(self, input_shape):
        d_model = input_shape[0][-1]
        self.gamma_dense = Dense(d_model)
        self.beta_dense = Dense(d_model)
        super(FiLMConditioningLayer, self).build(input_shape)

    def call(self, inputs):
        x, condition = inputs

        # Create gamma and beta
        gamma = self.gamma_dense(condition)
        beta = self.beta_dense(condition)

        # Reshape for broadcasting using Keras operations
        gamma = Reshape((1, -1))(gamma)
        beta = Reshape((1, -1))(beta)

        # Apply FiLM conditioning
        return x * gamma + beta

    def get_config(self):
        config = super(FiLMConditioningLayer, self).get_config()
        return config


# Custom VAE Loss Layer
class VAELossLayer(Layer):
    def __init__(self, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        y_true, y_pred, z_mean, z_log_var = inputs

        # MSE reconstruction loss
        reconstruction_loss = tf.reduce_mean(tf.square(y_true - y_pred))

        # KL divergence loss - using Keras operations
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

        # Add losses
        self.add_loss(reconstruction_loss + kl_loss)

        # Return the prediction unchanged
        return y_pred

    def get_config(self):
        config = super(VAELossLayer, self).get_config()
        return config


# Gradient Accumulation Callback
class GradientAccumulationCallback(tf.keras.callbacks.Callback):
    def __init__(self, accumulation_steps=2):
        super(GradientAccumulationCallback, self).__init__()
        self.accumulation_steps = accumulation_steps
        self.gradients = None
        self.batch_count = 0

    def on_train_batch_begin(self, batch, logs=None):
        # Reset gradients at the beginning of each accumulation cycle
        if self.batch_count % self.accumulation_steps == 0:
            self.gradients = None

    def on_train_batch_end(self, batch, logs=None):
        self.batch_count += 1
        # Apply accumulated gradients at the end of each accumulation cycle
        if self.batch_count % self.accumulation_steps == 0:
            self.model.optimizer.apply_gradients(zip(self.gradients, self.model.trainable_variables))


class GenAIDataAugmentation:
    """
    A class for generating synthetic financial data using various generative models.
    Loads market regime detection results and generates synthetic data for each regime.
    """

    def __init__(self, regime_dir=REGIME_DIR, features_path=FEATURES_PATH,
                 base_output_dir=BASE_OUTPUT_DIR, regime_method=REGIME_METHOD,
                 sequence_length=SEQUENCE_LENGTH, max_features=MAX_FEATURES,
                 num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                 gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS):
        """
        Initialize the GenAI data augmentation with paths to regime detection results and features.
        """
        self.regime_dir = regime_dir
        self.features_path = features_path
        self.base_output_dir = base_output_dir
        self.regime_method = regime_method
        self.sequence_length = sequence_length
        self.max_features = max_features
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Create method-specific output directory
        self.output_dir = os.path.join(self.base_output_dir, self.regime_method)
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize data containers
        self.features = None
        self.regime_labels = None
        self.models = {}
        self.synthetic_data = {}
        self.feature_names = None  # To store selected feature names

        # Initialize log file path (before using it in log method)
        self.log_file = os.path.join(self.output_dir, f"{self.regime_method}_genai_augmentation_log.txt")
        with open(self.log_file, "w") as f:  # Clear log file on init
            f.write(f"GenAI Data Augmentation Log - {self.regime_method}\n")
            f.write("=========================================\n\n")

        # Log initialization
        self.log(f"GenAI Data Augmentation started at: {datetime.now()}")
        self.log(f"Regime directory: {self.regime_dir}")
        self.log(f"Features path: {self.features_path}")
        self.log(f"Output directory: {self.output_dir}")
        self.log(f"Regime method: {self.regime_method}")
        self.log(f"Sequence length: {self.sequence_length}")
        self.log(f"Max features: {self.max_features}")

        # Check for GPU
        if physical_devices:
            self.log(f"GPU acceleration enabled with {len(physical_devices)} device(s)")
            self.log("Mixed precision training enabled for faster execution")
            self.log("XLA compilation enabled for accelerated execution")
            self.log(f"Gradient accumulation enabled with {self.gradient_accumulation_steps} steps")
        else:
            self.log("No GPU found. Running on CPU.")

        self.log(f"Number of epochs: {self.num_epochs}")
        self.log(f"Batch size: {self.batch_size}")

    def log(self, message):
        """Add message to log file"""
        print(message)
        with open(self.log_file, "a") as f:
            f.write(f"{datetime.now()}: {message}\n")

    def load_data(self):
        """
        Load features and regime labels.
        """
        self.log(f"Loading features and {self.regime_method} regime labels...")

        # Load features
        try:
            self.features = pd.read_csv(self.features_path, index_col=0, parse_dates=True)
            self.log(f"Loaded features from {self.features_path} with shape: {self.features.shape}")

            # Clean features - replace inf, -inf with NaN, then fill NaN with column means
            self.log("Cleaning features: replacing infinities and NaNs...")
            self.features = self.features.replace([np.inf, -np.inf], np.nan)

            nan_count = self.features.isna().sum().sum()
            self.log(f"Found {nan_count} NaN values in features")

            self.features = self.features.fillna(self.features.mean())

            if self.features.isna().sum().sum() > 0 or np.isinf(self.features.values).any():
                self.log("Warning: Some NaN or infinity values could not be fixed. Using more aggressive cleaning...")
                nan_cols = self.features.columns[self.features.isna().mean() > 0.1]
                if len(nan_cols) > 0:
                    self.log(f"Dropping {len(nan_cols)} columns with >10% NaN values")
                    self.features = self.features.drop(columns=nan_cols)
                self.features = self.features.fillna(0)
                self.features = self.features.replace([np.inf], 1e10)
                self.features = self.features.replace([-np.inf], -1e10)

            # Feature selection - reduce dimensionality if needed
            if self.features.shape[1] > self.max_features:
                self.log(f"Reducing feature dimensionality from {self.features.shape[1]} to {self.max_features}...")
                feature_variance = self.features.var()
                top_features = feature_variance.nlargest(self.max_features).index
                self.features = self.features[top_features]
                self.log(f"Selected {self.max_features} features with highest variance")

            self.feature_names = self.features.columns  # Store selected feature names
            self.log(f"Features cleaned. New shape: {self.features.shape}")

        except Exception as e:
            self.log(f"Error loading features from {self.features_path}: {str(e)}")
            return None, None

        # Load regime labels
        try:
            labels_path = os.path.join(self.regime_dir, f"{self.regime_method}_regime_labels.csv")
            if os.path.exists(labels_path):
                self.regime_labels = pd.read_csv(labels_path, index_col=0, parse_dates=True)
                self.log(f"Loaded regime labels from {labels_path} with shape: {self.regime_labels.shape}")
            else:
                # Try to load from the combined file if the specific one doesn't exist
                combined_path = os.path.join(self.regime_dir, "all_regime_labels.csv")
                if os.path.exists(combined_path):
                    self.log(f"Specific labels file not found, trying {combined_path}")
                    combined_labels = pd.read_csv(combined_path, index_col=0, parse_dates=True)
                    if self.regime_method in combined_labels.columns:
                        self.regime_labels = combined_labels[[self.regime_method]]
                        self.log(
                            f"Loaded {self.regime_method} regime labels from combined file with shape: {self.regime_labels.shape}")
                    else:
                        self.log(f"Error: {self.regime_method} column not found in {combined_path}")
                        return None, None
                else:
                    self.log(f"Error: Regime labels file not found at {labels_path} or {combined_path}")
                    return None, None
        except Exception as e:
            self.log(f"Error loading regime labels: {str(e)}")
            return None, None

        # Ensure features and regime labels have the same index
        self.features, self.regime_labels = self.features.align(self.regime_labels, join="inner", axis=0)
        self.log(f"Aligned features and regime labels. New shape: {self.features.shape}")

        if self.features.empty or self.regime_labels.empty:
            self.log("Error: Alignment resulted in empty dataframes. Check input files and date ranges.")
            return None, None

        return self.features, self.regime_labels

    def prepare_sequences(self):
        """
        Prepare sequences for time series models.
        """
        self.log(f"Preparing sequences with length {self.sequence_length}...")

        if self.features is None or self.regime_labels is None:
            self.log("Error: No features or regime labels loaded. Run load_data first.")
            return None

        method = self.regime_labels.columns[0]

        try:
            if np.isnan(self.features.values).any() or np.isinf(self.features.values).any():
                self.log("Warning: Features still contain NaN or infinity values. Applying additional cleaning...")
                self.features = self.features.replace([np.inf, -np.inf], np.nan)
                self.features = self.features.fillna(0)

            scaler = RobustScaler()
            scaled_features = scaler.fit_transform(self.features)

            if np.isnan(scaled_features).any() or np.isinf(scaled_features).any():
                self.log("Warning: Scaling produced NaN or infinity values. Using StandardScaler instead...")
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(self.features)

                if np.isnan(scaled_features).any() or np.isinf(scaled_features).any():
                    self.log("Warning: Standard scaling also failed. Using simple min-max normalization...")
                    feature_min = np.nanmin(self.features.values, axis=0)
                    feature_max = np.nanmax(self.features.values, axis=0)
                    feature_range = feature_max - feature_min
                    feature_range[feature_range == 0] = 1
                    scaled_features = (self.features.values - feature_min) / feature_range
                    scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=1.0, neginf=-1.0)

            scaled_features_df = pd.DataFrame(scaled_features, index=self.features.index, columns=self.features.columns)

        except Exception as e:
            self.log(f"Error during feature scaling: {str(e)}")
            self.log("Attempting simple normalization as fallback...")
            try:
                features_np = self.features.values
                features_np = np.nan_to_num(features_np, nan=0.0, posinf=1e10, neginf=-1e10)
                col_min = np.min(features_np, axis=0)
                col_max = np.max(features_np, axis=0)
                col_range = col_max - col_min
                col_range[col_range == 0] = 1
                scaled_features = (features_np - col_min) / col_range
                scaled_features_df = pd.DataFrame(scaled_features, index=self.features.index,
                                                  columns=self.features.columns)
            except Exception as inner_e:
                self.log(f"Error during fallback normalization: {str(inner_e)}")
                return None

        sequences_by_regime = {}
        unique_regimes = sorted(self.regime_labels[method].unique())
        self.log(f"Found unique regimes: {unique_regimes}")

        for regime in unique_regimes:
            regime_indices = self.regime_labels[self.regime_labels[method] == regime].index
            regime_features = scaled_features_df.loc[regime_indices]

            sequences = []
            for i in range(len(regime_features) - self.sequence_length + 1):
                seq = regime_features.iloc[i:i + self.sequence_length].values
                sequences.append(seq)

            if len(sequences) > 0:
                sequences_by_regime[regime] = np.array(sequences)
                self.log(f"Created {len(sequences)} sequences for regime {regime}")
            else:
                self.log(f"Warning: No sequences created for regime {regime} (not enough data points?)")

        return sequences_by_regime

    def create_tf_dataset(self, sequences, regimes, batch_size=None, is_training=True, model_type=None):
        """
        Create optimized TensorFlow dataset with prefetching.
        """
        if batch_size is None:
            batch_size = self.batch_size

        if model_type == "cvae":
            dataset = tf.data.Dataset.from_tensor_slices(
                ((sequences, regimes), sequences)  # ((inputs), targets)
            )
        else:
            dataset = tf.data.Dataset.from_tensor_slices((sequences, regimes))

        if is_training:
            dataset = dataset.shuffle(buffer_size=10000)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def build_timegan(self, n_features=None, hidden_dim=48, n_regimes=3):
        """
        Build TimeGAN model with optimized complexity.
        """
        self.log("Building TimeGAN model with optimized complexity...")

        if n_features is None:
            if self.feature_names is None:
                self.log("Error: Feature names not set. Run load_data first.")
                return None
            n_features = len(self.feature_names)

        # Embedding network
        def build_embedding_net():
            model = Sequential(name="Embedding")
            model.add(Input(shape=(self.sequence_length, n_features)))
            model.add(GRU(hidden_dim, return_sequences=True))
            model.add(GRU(hidden_dim, return_sequences=True))
            return model

        # Recovery network
        def build_recovery_net():
            model = Sequential(name="Recovery")
            model.add(Input(shape=(self.sequence_length, hidden_dim)))
            model.add(GRU(hidden_dim, return_sequences=True))
            model.add(GRU(n_features, return_sequences=True))
            return model

        # Generator network
        def build_generator(n_regimes):
            z_input = Input(shape=(self.sequence_length, hidden_dim))
            regime_input = Input(shape=(1,))
            regime_embedding = Embedding(n_regimes, 24)(regime_input)
            regime_embedding = Flatten()(regime_embedding)
            regime_embedding = Dense(hidden_dim, activation="relu")(regime_embedding)
            regime_embedding = RepeatVector(self.sequence_length)(regime_embedding)
            combined = Concatenate()([z_input, regime_embedding])
            x = GRU(hidden_dim, return_sequences=True)(combined)
            x = GRU(hidden_dim, return_sequences=True)(x)
            output = Dense(hidden_dim, activation="tanh")(x)
            model = Model([z_input, regime_input], output, name="Generator")
            return model

        # Discriminator network
        def build_discriminator():
            model = Sequential(name="Discriminator")
            model.add(Input(shape=(self.sequence_length, hidden_dim)))
            model.add(GRU(hidden_dim, return_sequences=True))
            model.add(GRU(hidden_dim, return_sequences=False))
            model.add(Dense(1, activation="sigmoid"))
            return model

        # Supervisor network
        def build_supervisor():
            model = Sequential(name="Supervisor")
            model.add(Input(shape=(self.sequence_length, hidden_dim)))
            model.add(GRU(hidden_dim, return_sequences=True))
            model.add(GRU(hidden_dim, return_sequences=True))
            model.add(Dense(hidden_dim, activation="tanh"))
            return model

        embedding_net = build_embedding_net()
        recovery_net = build_recovery_net()
        generator_net = build_generator(n_regimes)
        discriminator_net = build_discriminator()
        supervisor_net = build_supervisor()

        self.models["timegan_embedding"] = embedding_net
        self.models["timegan_recovery"] = recovery_net
        self.models["timegan_generator"] = generator_net
        self.models["timegan_discriminator"] = discriminator_net
        self.models["timegan_supervisor"] = supervisor_net

        return generator_net, discriminator_net, supervisor_net, embedding_net, recovery_net

    def train_timegan(self, sequences_by_regime):
        """
        Train TimeGAN model with optimized implementation.
        """
        self.log("Training TimeGAN model with XLA compilation and mixed precision...")

        if "timegan_embedding" not in self.models:
            self.log("Error: TimeGAN model not built. Run build_timegan first.")
            return None

        embedding_net = self.models["timegan_embedding"]
        recovery_net = self.models["timegan_recovery"]
        generator_net = self.models["timegan_generator"]
        discriminator_net = self.models["timegan_discriminator"]
        supervisor_net = self.models["timegan_supervisor"]

        all_sequences = []
        all_regimes = []
        n_regimes = 0
        for regime, sequences in sequences_by_regime.items():
            all_sequences.append(sequences)
            all_regimes.append(np.full(len(sequences), regime))
            n_regimes = max(n_regimes, regime + 1)

        if not all_sequences:
            self.log("Error: No sequences available for training TimeGAN.")
            return None

        all_sequences = np.vstack(all_sequences)
        all_regimes = np.concatenate(all_regimes).reshape(-1, 1)

        train_dataset = self.create_tf_dataset(all_sequences, all_regimes)

        n_samples = len(all_sequences)
        hidden_dim = generator_net.output_shape[-1]
        batch_size = self.batch_size
        n_epochs = self.num_epochs

        history = {
            "embedding_loss": [],
            "supervisor_loss": [],
            "generator_loss": [],
            "discriminator_loss": []
        }

        optimizer_e = Adam(learning_rate=0.001)
        optimizer_r = Adam(learning_rate=0.001)
        optimizer_s = Adam(learning_rate=0.001)
        optimizer_g = Adam(learning_rate=0.001)
        optimizer_d = Adam(learning_rate=0.001)

        mse = tf.keras.losses.MeanSquaredError()
        bce = tf.keras.losses.BinaryCrossentropy()

        @tf.function(jit_compile=True)
        def train_embedding_recovery_step(x_batch):
            with tf.GradientTape() as tape_e, tf.GradientTape() as tape_r:
                h = embedding_net(x_batch)
                x_tilde = recovery_net(h)
                e_loss = mse(x_batch, x_tilde)
            vars_e = embedding_net.trainable_variables
            grads_e = tape_e.gradient(e_loss, vars_e)
            optimizer_e.apply_gradients(zip(grads_e, vars_e))
            vars_r = recovery_net.trainable_variables
            grads_r = tape_r.gradient(e_loss, vars_r)
            optimizer_r.apply_gradients(zip(grads_r, vars_r))
            return e_loss

        @tf.function(jit_compile=True)
        def train_supervisor_step(x_batch):
            with tf.GradientTape() as tape:
                h = embedding_net(x_batch)
                h_hat = supervisor_net(h)
                s_loss = mse(h[:, 1:, :], h_hat[:, :-1, :])
            vars_s = supervisor_net.trainable_variables
            grads = tape.gradient(s_loss, vars_s)
            optimizer_s.apply_gradients(zip(grads, vars_s))
            return s_loss

        @tf.function(jit_compile=True)
        def train_generator_step(x_batch, z_batch, regime_batch):
            with tf.GradientTape() as tape:
                h_hat = generator_net([z_batch, regime_batch])
                e_hat = supervisor_net(h_hat)
                y_fake = discriminator_net(h_hat)
                g_loss_s = mse(h_hat[:, 1:, :], e_hat[:, :-1, :])
                g_loss_u = bce(tf.ones_like(y_fake), y_fake)
                g_loss = g_loss_s + g_loss_u
            vars_g = generator_net.trainable_variables
            grads = tape.gradient(g_loss, vars_g)
            optimizer_g.apply_gradients(zip(grads, vars_g))
            return g_loss

        @tf.function(jit_compile=True)
        def train_discriminator_step(x_batch, z_batch, regime_batch):
            with tf.GradientTape() as tape:
                h = embedding_net(x_batch)
                y_real = discriminator_net(h)
                h_hat = generator_net([z_batch, regime_batch])
                y_fake = discriminator_net(h_hat)
                d_loss_real = bce(tf.ones_like(y_real), y_real)
                d_loss_fake = bce(tf.zeros_like(y_fake), y_fake)
                d_loss = d_loss_real + d_loss_fake
            vars_d = discriminator_net.trainable_variables
            grads = tape.gradient(d_loss, vars_d)
            optimizer_d.apply_gradients(zip(grads, vars_d))
            return d_loss

        for epoch in range(n_epochs):
            start_time = time.time()
            e_loss_epoch, s_loss_epoch, g_loss_epoch, d_loss_epoch = [], [], [], []

            for x_batch, regime_batch in train_dataset:
                z_batch = tf.random.normal((tf.shape(x_batch)[0], self.sequence_length, hidden_dim))
                e_loss = train_embedding_recovery_step(x_batch)
                s_loss = train_supervisor_step(x_batch)
                g_loss = train_generator_step(x_batch, z_batch, regime_batch)
                d_loss = train_discriminator_step(x_batch, z_batch, regime_batch)
                e_loss_epoch.append(e_loss.numpy())
                s_loss_epoch.append(s_loss.numpy())
                g_loss_epoch.append(g_loss.numpy())
                d_loss_epoch.append(d_loss.numpy())

            e_loss_epoch = np.mean(e_loss_epoch)
            s_loss_epoch = np.mean(s_loss_epoch)
            g_loss_epoch = np.mean(g_loss_epoch)
            d_loss_epoch = np.mean(d_loss_epoch)

            history["embedding_loss"].append(e_loss_epoch)
            history["supervisor_loss"].append(s_loss_epoch)
            history["generator_loss"].append(g_loss_epoch)
            history["discriminator_loss"].append(d_loss_epoch)

            elapsed_time = time.time() - start_time
            self.log(f"Epoch {epoch + 1}/{n_epochs} - {elapsed_time:.2f}s - "
                     f"E: {e_loss_epoch:.4f}, S: {s_loss_epoch:.4f}, "
                     f"G: {g_loss_epoch:.4f}, D: {d_loss_epoch:.4f}")

            if (epoch + 1) % 10 == 0:
                gc.collect()

        self.log("TimeGAN training completed")

        try:
            generator_net.save(os.path.join(self.output_dir, f"{self.regime_method}_timegan_generator_model.keras"))
            discriminator_net.save(
                os.path.join(self.output_dir, f"{self.regime_method}_timegan_discriminator_model.keras"))
            embedding_net.save(os.path.join(self.output_dir, f"{self.regime_method}_timegan_embedding_model.keras"))
            recovery_net.save(os.path.join(self.output_dir, f"{self.regime_method}_timegan_recovery_model.keras"))
            supervisor_net.save(os.path.join(self.output_dir, f"{self.regime_method}_timegan_supervisor_model.keras"))
            self.log("TimeGAN models saved")
        except Exception as e:
            self.log(f"Error saving TimeGAN models: {str(e)}")

        clear_memory()
        return history

    @tf.function(jit_compile=True)
    def generate_timegan_samples_batch(self, generator, recovery, z, regime_labels):
        """
        Generate TimeGAN samples for a batch with XLA compilation.
        """
        synthetic_latent = generator([z, regime_labels])
        return recovery(synthetic_latent)

    def generate_timegan_samples(self, n_samples=100, regime=0):
        """
        Generate synthetic samples using TimeGAN.
        """
        self.log(f"Generating {n_samples} TimeGAN samples for regime {regime}...")

        if "timegan_generator" not in self.models or "timegan_recovery" not in self.models:
            self.log("Error: TimeGAN model not built or loaded. Run build_timegan and train_timegan first.")
            return None

        generator = self.models["timegan_generator"]
        recovery = self.models["timegan_recovery"]
        hidden_dim = generator.output_shape[-1]
        n_features = recovery.output_shape[-1]

        batch_size = self.batch_size
        synthetic_samples_list = []

        for i in range(0, n_samples, batch_size):
            current_batch_size = min(batch_size, n_samples - i)
            z = tf.random.normal((current_batch_size, self.sequence_length, hidden_dim))
            regime_labels = tf.constant(np.full((current_batch_size, 1), regime), dtype=tf.int32)
            batch_samples = self.generate_timegan_samples_batch(generator, recovery, z, regime_labels).numpy()
            synthetic_samples_list.append(batch_samples)

        synthetic_samples = np.vstack(synthetic_samples_list)

        if regime not in self.synthetic_data:
            self.synthetic_data[regime] = {}
        self.synthetic_data[regime]["timegan"] = synthetic_samples

        output_path = os.path.join(self.output_dir, f"{self.regime_method}_timegan_synthetic_data_regime_{regime}.csv")
        flat_synthetic = synthetic_samples.reshape(-1, n_features)
        # Use stored feature names
        synthetic_df = pd.DataFrame(flat_synthetic, columns=self.feature_names)
        synthetic_df["regime"] = regime
        synthetic_df["method"] = "timegan"
        synthetic_df.to_csv(output_path, index=False)
        self.log(f"Saved TimeGAN synthetic data for regime {regime} to {output_path}")

        return synthetic_samples

    def build_cvae(self, n_features=None, latent_dim=24, n_regimes=3):
        """
        Build Conditional Variational Autoencoder model with optimized complexity.
        """
        self.log("Building CVAE model with optimized complexity...")

        if n_features is None:
            if self.feature_names is None:
                self.log("Error: Feature names not set. Run load_data first.")
                return None
            n_features = len(self.feature_names)

        sequence_input = Input(shape=(self.sequence_length, n_features), name="sequence_input")
        regime_input = Input(shape=(1,), name="regime_input")

        regime_embedding = Embedding(n_regimes, 24)(regime_input)
        regime_embedding = Flatten()(regime_embedding)
        regime_embedding = Dense(self.sequence_length, activation="relu")(regime_embedding)
        regime_embedding = Reshape((self.sequence_length, 1))(regime_embedding)

        x = Concatenate(axis=-1)([sequence_input, regime_embedding])

        x = Conv1D(64, 3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv1D(48, 3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv1D(32, 3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)

        z_mean = Dense(latent_dim, name="z_mean")(x)
        z_log_var = Dense(latent_dim, name="z_log_var")(x)

        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        z = Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])

        z_input = Input(shape=(latent_dim,), name="z_input")
        regime_input_decoder = Input(shape=(1,), name="regime_input_decoder")
        regime_emb_decoder = Embedding(n_regimes, 24)(regime_input_decoder)
        regime_emb_decoder = Flatten()(regime_emb_decoder)
        decoder_inputs = Concatenate()([z_input, regime_emb_decoder])

        x_decoder = Dense(64, activation="relu")(decoder_inputs)
        x_decoder = Dense(128, activation="relu")(x_decoder)
        x_decoder = Dense(self.sequence_length * 32, activation="relu")(x_decoder)
        x_decoder = Reshape((self.sequence_length, 32))(x_decoder)
        x_decoder = Conv1D(32, 3, activation="relu", padding="same")(x_decoder)
        x_decoder = BatchNormalization()(x_decoder)
        x_decoder = Conv1D(48, 3, activation="relu", padding="same")(x_decoder)
        x_decoder = BatchNormalization()(x_decoder)
        x_decoder = Conv1D(64, 3, activation="relu", padding="same")(x_decoder)
        x_decoder = BatchNormalization()(x_decoder)
        decoder_outputs = Conv1D(n_features, 3, activation="linear", padding="same")(x_decoder)

        decoder = Model([z_input, regime_input_decoder], decoder_outputs, name="decoder")
        encoder = Model([sequence_input, regime_input], [z_mean, z_log_var, z], name="encoder")
        z_output = encoder([sequence_input, regime_input])[2]
        x_decoded = decoder([z_output, regime_input])
        outputs = VAELossLayer()([sequence_input, x_decoded, z_mean, z_log_var])
        cvae = Model([sequence_input, regime_input], outputs, name="cvae")
        cvae.compile(optimizer=Adam(learning_rate=0.001), loss="mse", jit_compile=True)

        self.models["cvae"] = cvae
        self.models["cvae_encoder"] = encoder
        self.models["cvae_decoder"] = decoder

        return encoder, decoder, cvae

    def train_cvae(self, sequences_by_regime):
        """
        Train CVAE model with mixed precision and XLA compilation.
        """
        self.log("Training CVAE model with XLA compilation and mixed precision...")

        if "cvae" not in self.models:
            self.log("Error: CVAE model not built. Run build_cvae first.")
            return None

        cvae = self.models["cvae"]

        all_sequences = []
        all_regimes = []
        n_regimes = 0
        for regime, sequences in sequences_by_regime.items():
            all_sequences.append(sequences)
            all_regimes.append(np.full(len(sequences), regime))
            n_regimes = max(n_regimes, regime + 1)

        if not all_sequences:
            self.log("Error: No sequences available for training CVAE.")
            return None

        all_sequences = np.vstack(all_sequences)
        all_regimes = np.concatenate(all_regimes).reshape(-1, 1)

        train_dataset = self.create_tf_dataset(all_sequences, all_regimes, model_type="cvae")

        class CombinedCallback(tf.keras.callbacks.Callback):
            def __init__(self, accumulation_steps=2):
                super(CombinedCallback, self).__init__()
                self.accumulation_steps = accumulation_steps
                self.gradients = None
                self.batch_count = 0

            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % 10 == 0:
                    gc.collect()

        history = cvae.fit(
            train_dataset,
            epochs=self.num_epochs,
            verbose=1,
            callbacks=[CombinedCallback(self.gradient_accumulation_steps)]
        )

        self.log("CVAE training completed")

        try:
            cvae.save(os.path.join(self.output_dir, f"{self.regime_method}_cvae_model.keras"))
            self.log("CVAE model saved")
        except Exception as e:
            self.log(f"Error saving CVAE model: {str(e)}")

        clear_memory()
        return history.history

    @tf.function(jit_compile=True)
    def generate_cvae_samples_batch(self, decoder, z, regime_labels):
        """
        Generate CVAE samples for a batch with XLA compilation.
        """
        return decoder([z, regime_labels])

    def generate_cvae_samples(self, n_samples=100, regime=0):
        """
        Generate synthetic samples using CVAE.
        """
        self.log(f"Generating {n_samples} CVAE samples for regime {regime}...")

        if "cvae_decoder" not in self.models:
            self.log("Error: CVAE model not built or loaded. Run build_cvae and train_cvae first.")
            return None

        decoder = self.models["cvae_decoder"]
        latent_dim = decoder.input_shape[0][1]
        n_features = decoder.output_shape[-1]

        batch_size = self.batch_size
        synthetic_samples_list = []

        for i in range(0, n_samples, batch_size):
            current_batch_size = min(batch_size, n_samples - i)
            z = tf.random.normal((current_batch_size, latent_dim))
            regime_labels = tf.constant(np.full((current_batch_size, 1), regime), dtype=tf.int32)
            batch_samples = self.generate_cvae_samples_batch(decoder, z, regime_labels).numpy()
            synthetic_samples_list.append(batch_samples)

        synthetic_samples = np.vstack(synthetic_samples_list)

        if regime not in self.synthetic_data:
            self.synthetic_data[regime] = {}
        self.synthetic_data[regime]["cvae"] = synthetic_samples

        output_path = os.path.join(self.output_dir, f"{self.regime_method}_cvae_synthetic_data_regime_{regime}.csv")
        flat_synthetic = synthetic_samples.reshape(-1, n_features)
        # Use stored feature names
        synthetic_df = pd.DataFrame(flat_synthetic, columns=self.feature_names)
        synthetic_df["regime"] = regime
        synthetic_df["method"] = "cvae"
        synthetic_df.to_csv(output_path, index=False)
        self.log(f"Saved CVAE synthetic data for regime {regime} to {output_path}")

        return synthetic_samples

    def build_film_transformer(self, n_features=None, n_regimes=3, d_model=48, n_heads=4, n_layers=2):
        """
        Build FiLM Transformer model with optimized complexity.
        """
        self.log("Building FiLM Transformer model with optimized complexity...")

        if n_features is None:
            if self.feature_names is None:
                self.log("Error: Feature names not set. Run load_data first.")
                return None
            n_features = len(self.feature_names)

        sequence_input = Input(shape=(self.sequence_length, n_features))
        regime_input = Input(shape=(1,))

        regime_embedding = Embedding(n_regimes, 24)(regime_input)
        regime_embedding = Flatten()(regime_embedding)
        regime_embedding = Dense(d_model, activation="relu")(regime_embedding)

        x = Dense(d_model)(sequence_input)
        pos_encoding = self.positional_encoding(self.sequence_length, d_model)
        x = x + pos_encoding

        for i in range(n_layers):
            attn_output = MultiHeadAttention(
                num_heads=n_heads, key_dim=d_model // n_heads
            )(x, x, x)
            attn_output = Dropout(0.1)(attn_output)
            out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)
            film_layer = FiLMConditioningLayer()
            out1 = film_layer([out1, regime_embedding])
            ffn_output = self.point_wise_feed_forward_network(d_model)(out1)
            ffn_output = Dropout(0.1)(ffn_output)
            out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
            x = out2

        output = Dense(n_features)(x)
        model = Model([sequence_input, regime_input], output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", jit_compile=True)

        self.models["film_transformer"] = model
        return model

    def positional_encoding(self, position, d_model):
        """
        Create positional encoding for transformer.
        """
        positions = np.arange(position)[:, np.newaxis]
        indices = np.arange(d_model)[np.newaxis, :]
        angles = positions / np.power(10000, (2 * (indices // 2)) / np.float32(d_model))
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        pos_encoding = angles[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def point_wise_feed_forward_network(self, d_model, dff=192):
        """
        Create point-wise feed-forward network for transformer.
        """
        return Sequential([
            Dense(dff, activation="relu"),
            Dense(d_model)
        ])

    def train_film_transformer(self, sequences_by_regime):
        """
        Train FiLM Transformer model with mixed precision and XLA compilation.
        """
        self.log("Training FiLM Transformer model with XLA compilation and mixed precision...")

        if "film_transformer" not in self.models:
            self.log("Error: FiLM Transformer model not built. Run build_film_transformer first.")
            return None

        model = self.models["film_transformer"]

        all_sequences = []
        all_regimes = []
        n_regimes = 0
        for regime, sequences in sequences_by_regime.items():
            all_sequences.append(sequences)
            all_regimes.append(np.full(len(sequences), regime))
            n_regimes = max(n_regimes, regime + 1)

        if not all_sequences:
            self.log("Error: No sequences available for training FiLM Transformer.")
            return None

        all_sequences = np.vstack(all_sequences)
        all_regimes = np.concatenate(all_regimes).reshape(-1, 1)

        targets = np.zeros_like(all_sequences)
        targets[:, :-1, :] = all_sequences[:, 1:, :]

        train_dataset = tf.data.Dataset.from_tensor_slices(
            ((all_sequences, all_regimes), targets)
        ).shuffle(10000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        class CombinedCallback(tf.keras.callbacks.Callback):
            def __init__(self, accumulation_steps=2):
                super(CombinedCallback, self).__init__()
                self.accumulation_steps = accumulation_steps
                self.gradients = None
                self.batch_count = 0

            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % 10 == 0:
                    gc.collect()

        history = model.fit(
            train_dataset,
            epochs=self.num_epochs,
            verbose=1,
            callbacks=[CombinedCallback(self.gradient_accumulation_steps)]
        )

        self.log("FiLM Transformer training completed")

        try:
            model.save(os.path.join(self.output_dir, f"{self.regime_method}_film_transformer_model.keras"))
            self.log("FiLM Transformer model saved")
        except Exception as e:
            self.log(f"Error saving FiLM Transformer model: {str(e)}")

        clear_memory()
        return history.history

    @tf.function(jit_compile=True)
    def generate_film_transformer_samples_batch(self, model, seed_sequences, regime_labels):
        """
        Generate FiLM Transformer samples for a batch with XLA compilation.
        """
        return model([seed_sequences, regime_labels])

    def generate_film_transformer_samples(self, n_samples=100, regime=0):
        """
        Generate synthetic samples using FiLM Transformer.
        """
        self.log(f"Generating {n_samples} FiLM Transformer samples for regime {regime}...")

        if "film_transformer" not in self.models:
            self.log(
                "Error: FiLM Transformer model not built or loaded. Run build_film_transformer and train_film_transformer first.")
            return None

        model = self.models["film_transformer"]
        input_shape = model.input_shape[0][1:]
        n_features = input_shape[-1]

        batch_size = self.batch_size
        synthetic_samples_list = []

        for i in range(0, n_samples, batch_size):
            current_batch_size = min(batch_size, n_samples - i)
            seed_sequences = tf.random.normal((current_batch_size,) + input_shape, mean=0, stddev=0.1)
            regime_labels = tf.constant(np.full((current_batch_size, 1), regime), dtype=tf.int32)

            for _ in range(5):
                seed_sequences = self.generate_film_transformer_samples_batch(
                    model, seed_sequences, regime_labels
                )

            synthetic_samples_list.append(seed_sequences.numpy())

        synthetic_samples = np.vstack(synthetic_samples_list)

        if regime not in self.synthetic_data:
            self.synthetic_data[regime] = {}
        self.synthetic_data[regime]["film_transformer"] = synthetic_samples

        output_path = os.path.join(self.output_dir,
                                   f"{self.regime_method}_film_transformer_synthetic_data_regime_{regime}.csv")
        flat_synthetic = synthetic_samples.reshape(-1, n_features)
        # Use stored feature names
        synthetic_df = pd.DataFrame(flat_synthetic, columns=self.feature_names)
        synthetic_df["regime"] = regime
        synthetic_df["method"] = "film_transformer"
        synthetic_df.to_csv(output_path, index=False)
        self.log(f"Saved FiLM Transformer synthetic data for regime {regime} to {output_path}")

        return synthetic_samples

    def build_regime_adversary(self, n_features=None, n_regimes=3):
        """
        Build Regime Adversary model with optimized complexity.
        """
        self.log("Building Regime Adversary model with optimized complexity...")

        if n_features is None:
            if self.feature_names is None:
                self.log("Error: Feature names not set. Run load_data first.")
                return None
            n_features = len(self.feature_names)

        sequence_input = Input(shape=(self.sequence_length, n_features))
        x = Conv1D(48, 3, activation="relu", padding="same")(sequence_input)
        x = BatchNormalization()(x)
        x = Conv1D(64, 3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv1D(48, 3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = Dense(64, activation="relu")(x)
        x = Dropout(0.3)(x)
        x = Dense(48, activation="relu")(x)
        output = Dense(n_regimes, activation="softmax")(x)
        model = Model(sequence_input, output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            jit_compile=True
        )

        self.models["regime_adversary"] = model
        return model

    def train_regime_adversary(self, sequences_by_regime):
        """
        Train Regime Adversary model with mixed precision and XLA compilation.
        """
        self.log("Training Regime Adversary model with XLA compilation and mixed precision...")

        if "regime_adversary" not in self.models:
            self.log("Error: Regime Adversary model not built. Run build_regime_adversary first.")
            return None

        model = self.models["regime_adversary"]

        all_sequences = []
        all_regimes = []
        n_regimes = 0
        for regime, sequences in sequences_by_regime.items():
            all_sequences.append(sequences)
            all_regimes.append(np.full(len(sequences), regime))
            n_regimes = max(n_regimes, regime + 1)

        if not all_sequences:
            self.log("Error: No sequences available for training Regime Adversary.")
            return None

        all_sequences = np.vstack(all_sequences)
        all_regimes = np.concatenate(all_regimes).reshape(-1, 1)

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (all_sequences, all_regimes)
        ).shuffle(10000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        class CombinedCallback(tf.keras.callbacks.Callback):
            def __init__(self, accumulation_steps=2):
                super(CombinedCallback, self).__init__()
                self.accumulation_steps = accumulation_steps
                self.gradients = None
                self.batch_count = 0

            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % 10 == 0:
                    gc.collect()

        history = model.fit(
            train_dataset,
            epochs=self.num_epochs,
            verbose=1,
            callbacks=[CombinedCallback(self.gradient_accumulation_steps)]
        )

        self.log("Regime Adversary training completed")

        try:
            model.save(os.path.join(self.output_dir, f"{self.regime_method}_regime_adversary_model.keras"))
            self.log("Regime Adversary model saved")
        except Exception as e:
            self.log(f"Error saving Regime Adversary model: {str(e)}")

        clear_memory()
        return history.history

    def evaluate_synthetic_data(self, real_data, synthetic_data, regime, method):
        """
        Evaluate the quality of synthetic data using t-SNE and PCA.
        """
        self.log(f"Evaluating synthetic data for regime {regime}, method {method}...")

        if real_data is None or synthetic_data is None:
            self.log("Error: Real or synthetic data is missing for evaluation.")
            return

        # Flatten sequences
        n_samples_real = real_data.shape[0]
        n_samples_synth = synthetic_data.shape[0]
        n_features = real_data.shape[-1]

        real_flat = real_data.reshape(n_samples_real * self.sequence_length, n_features)
        synth_flat = synthetic_data.reshape(n_samples_synth * self.sequence_length, n_features)

        # Combine data
        combined_data = np.vstack([real_flat, synth_flat])
        labels = np.array(["Real"] * len(real_flat) + ["Synthetic"] * len(synth_flat))

        # Limit data size for visualization if too large
        max_vis_samples = 5000
        if len(combined_data) > max_vis_samples:
            self.log(
                f"Data size ({len(combined_data)}) exceeds limit ({max_vis_samples}). Subsampling for visualization.")
            indices = np.random.choice(len(combined_data), max_vis_samples, replace=False)
            combined_data = combined_data[indices]
            labels = labels[indices]

        # PCA
        try:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(combined_data)
            pca_df = pd.DataFrame({"PCA1": pca_result[:, 0], "PCA2": pca_result[:, 1], "Label": labels})

            plt.figure(figsize=(10, 7))
            sns.scatterplot(data=pca_df, x="PCA1", y="PCA2", hue="Label", alpha=0.7)
            plt.title(f"PCA Visualization - Regime {regime} ({method})")
            pca_plot_path = os.path.join(self.output_dir, f"{self.regime_method}_{method}_pca_regime_{regime}.png")
            plt.savefig(pca_plot_path)
            plt.close()
            self.log(f"PCA plot saved to {pca_plot_path}")
        except Exception as e:
            self.log(f"Error during PCA visualization: {str(e)}")

        # t-SNE
        try:
            tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
            tsne_result = tsne.fit_transform(combined_data)
            tsne_df = pd.DataFrame({"t-SNE1": tsne_result[:, 0], "t-SNE2": tsne_result[:, 1], "Label": labels})

            plt.figure(figsize=(10, 7))
            sns.scatterplot(data=tsne_df, x="t-SNE1", y="t-SNE2", hue="Label", alpha=0.7)
            plt.title(f"t-SNE Visualization - Regime {regime} ({method})")
            tsne_plot_path = os.path.join(self.output_dir, f"{self.regime_method}_{method}_tsne_regime_{regime}.png")
            plt.savefig(tsne_plot_path)
            plt.close()
            self.log(f"t-SNE plot saved to {tsne_plot_path}")
        except Exception as e:
            self.log(f"Error during t-SNE visualization: {str(e)}")

        clear_memory()

    def run_augmentation(self):
        """
        Run the full data augmentation pipeline.
        """
        self.log("Starting GenAI data augmentation pipeline...")

        # Load data
        features, regime_labels = self.load_data()
        if features is None or regime_labels is None:
            self.log("Pipeline failed: Could not load data.")
            return

        # Prepare sequences
        sequences_by_regime = self.prepare_sequences()
        if sequences_by_regime is None or not sequences_by_regime:
            self.log("Pipeline failed: Could not prepare sequences.")
            return

        # Determine number of regimes
        n_regimes = len(sequences_by_regime)
        n_features = list(sequences_by_regime.values())[0].shape[-1]

        # --- TimeGAN --- #
        self.log("\n--- Running TimeGAN ---")
        self.build_timegan(n_features=n_features, n_regimes=n_regimes)
        self.train_timegan(sequences_by_regime)
        for regime in sequences_by_regime.keys():
            n_samples_to_gen = len(sequences_by_regime[regime])  # Generate same number as real samples
            synth_timegan = self.generate_timegan_samples(n_samples=n_samples_to_gen, regime=regime)
            if synth_timegan is not None:
                self.evaluate_synthetic_data(sequences_by_regime[regime], synth_timegan, regime, "timegan")
        clear_memory()

        # --- CVAE --- #
        self.log("\n--- Running CVAE ---")
        self.build_cvae(n_features=n_features, n_regimes=n_regimes)
        self.train_cvae(sequences_by_regime)
        for regime in sequences_by_regime.keys():
            n_samples_to_gen = len(sequences_by_regime[regime])
            synth_cvae = self.generate_cvae_samples(n_samples=n_samples_to_gen, regime=regime)
            if synth_cvae is not None:
                self.evaluate_synthetic_data(sequences_by_regime[regime], synth_cvae, regime, "cvae")
        clear_memory()

        # --- FiLM Transformer --- #
        self.log("\n--- Running FiLM Transformer ---")
        self.build_film_transformer(n_features=n_features, n_regimes=n_regimes)
        self.train_film_transformer(sequences_by_regime)
        for regime in sequences_by_regime.keys():
            n_samples_to_gen = len(sequences_by_regime[regime])
            synth_film = self.generate_film_transformer_samples(n_samples=n_samples_to_gen, regime=regime)
            if synth_film is not None:
                self.evaluate_synthetic_data(sequences_by_regime[regime], synth_film, regime, "film_transformer")
        clear_memory()

        # --- Regime Adversary (Evaluation Only) --- #
        self.log("\n--- Running Regime Adversary (Evaluation) ---")
        self.build_regime_adversary(n_features=n_features, n_regimes=n_regimes)
        self.train_regime_adversary(sequences_by_regime)
        # No generation for adversary, it's for evaluation
        clear_memory()

        self.log("\nGenAI data augmentation pipeline completed.")


# Main execution block
if __name__ == "__main__":
    augmentor = GenAIDataAugmentation()
    augmentor.run_augmentation()

