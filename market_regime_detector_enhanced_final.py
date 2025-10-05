#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Regime Detector - Enhanced Version
Detects market regimes using multiple methods and creates consensus regimes.
"""

import os
import time
import warnings
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP # Corrected import for UMAP class
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from hmmlearn import hmm
from tensorflow.keras.layers import Input, Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Set TensorFlow logging level
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Suppress warnings
warnings.filterwarnings("ignore")

class MarketRegimeDetector:
    """
    Detects market regimes using multiple methods and creates consensus regimes.
    """

    def __init__(self, data_dir="C:/Users/chake/Documents/ML Project/processed_data",
                 output_dir="C:/Users/chake/Documents/ML Project/regime_detection_results"):
        """
        Initialize the MarketRegimeDetector.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing preprocessed data
        output_dir : str
            Directory to save results
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize data containers
        self.features = None
        self.aligned_prices = None
        self.returns = None
        self.targets = None
        self.clustering_features = None
        
        # Initialize results containers
        self.regime_labels = {}
        self.models = {}
        self.silhouette_scores = {}
        self.regime_characteristics = {}
        
        # Initialize logger
        self.log_file = os.path.join(self.output_dir, "regime_detection_log.txt")
        with open(self.log_file, "w") as f:
            f.write("Market Regime Detector Log\n")
            f.write("=========================\n\n")
    
    def log(self, message):
        """
        Log a message to the log file and print it.
        
        Parameters:
        -----------
        message : str
            Message to log
        """
        print(message)
        with open(self.log_file, "a") as f:
            f.write(f"{message}\n")
    
    def load_data(self, features_file="my_data_full_v2_features.csv", 
                 aligned_file="my_data_full_v2_aligned.csv",
                 targets_file="my_data_full_v2_targets.csv"):
        """
        Load preprocessed data.
        
        Parameters:
        -----------
        features_file : str
            Filename of features CSV
        aligned_file : str
            Filename of aligned prices CSV
        targets_file : str
            Filename of targets CSV
            
        Returns:
        --------
        bool
            True if data loaded successfully, False otherwise
        """
        self.log("Loading preprocessed data...")
        
        try:
            # Load features
            features_path = os.path.join(self.data_dir, features_file)
            self.features = pd.read_csv(features_path, index_col=0, parse_dates=True)
            self.log(f"Loaded features from {features_path} with shape: {self.features.shape}")
            
            # Load aligned prices
            aligned_path = os.path.join(self.data_dir, aligned_file)
            self.aligned_prices = pd.read_csv(aligned_path, index_col=0, parse_dates=True)
            self.log(f"Loaded aligned prices from {aligned_path} with shape: {self.aligned_prices.shape}")
            
            # Calculate returns
            self.returns = self.aligned_prices.pct_change().iloc[1:]
            self.log(f"Calculated returns with shape: {self.returns.shape}")
            
            # Load targets
            targets_path = os.path.join(self.data_dir, targets_file)
            self.targets = pd.read_csv(targets_path, index_col=0, parse_dates=True)
            self.log(f"Loaded targets from {targets_path} with shape: {self.targets.shape}")
            
            # Align features and targets
            self.features, self.targets = self.features.align(self.targets, join='inner', axis=0)
            self.log(f"Aligned features and targets. New shape: {self.features.shape}")
            
            return True
            
        except Exception as e:
            self.log(f"Error loading data: {str(e)}")
            return False
    
    def prepare_features_for_regime_detection(self, feature_selection='market_indicators', add_deepseek_features=True):
        """
        Prepare features for regime detection.
        
        Parameters:
        -----------
        feature_selection : str
            Method to select features:
            - 'all': Use all features
            - 'market_indicators': Use only market indicators
            - 'pca': Use PCA to reduce dimensionality
        add_deepseek_features : bool
            Whether to add enhanced features suggested by DeepSeek
            
        Returns:
        --------
        pandas.DataFrame
            Prepared features
        """
        self.log(f"Preparing features using {feature_selection} method with DeepSeek enhancements: {add_deepseek_features}...")
        
        if self.features is None:
            self.log("Error: No features loaded.")
            return None
        
        # Start with all features
        selected_features = self.features.copy()
        
        # Select features based on method
        if feature_selection == 'market_indicators':
            # Select only market indicators (technical indicators, etc.)
            # This is a simplified approach - in practice, you would have a more sophisticated selection
            indicator_cols = [col for col in selected_features.columns if 
                             any(ind in col.lower() for ind in ['rsi', 'macd', 'bb', 'atr', 'ema', 'sma', 'momentum', 'vol'])]
            
            if len(indicator_cols) > 0:
                selected_features = selected_features[indicator_cols]
            else:
                # Fallback if no indicator columns found
                self.log("Warning: No indicator columns found. Using all features.")
        
        elif feature_selection == 'pca':
            # Use PCA to reduce dimensionality
            from sklearn.decomposition import PCA
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(selected_features)
            
            # Apply PCA
            pca = PCA(n_components=min(5, selected_features.shape[1]))
            pca_features = pca.fit_transform(scaled_features)
            
            # Convert back to DataFrame
            selected_features = pd.DataFrame(
                pca_features, 
                index=selected_features.index,
                columns=[f'PC{i+1}' for i in range(pca_features.shape[1])]
            )
            
            self.log(f"Reduced features to {selected_features.shape[1]} principal components.")
        
        # Add enhanced features suggested by DeepSeek
        if add_deepseek_features:
            self.log("Adding enhanced features (volatility, correlation, momentum)...")
            
            # Calculate returns for enhanced features if not already done
            if self.returns is None and self.aligned_prices is not None:
                self.returns = self.aligned_prices.pct_change().iloc[1:]
                self.log(f"Calculating returns for enhanced features...")
            
            if self.returns is not None:
                # 1. Rolling volatility (standard deviation of returns)
                vol_window = 20
                rolling_vol = self.returns.rolling(window=vol_window).std()
                
                # 2. Cross-asset correlation
                # Calculate correlation between SPY and other assets
                if 'SPY' in self.returns.columns:
                    spy_returns = self.returns['SPY']
                    
                    # Calculate rolling correlation with SPY
                    corr_window = 30
                    rolling_corr = pd.DataFrame(index=self.returns.index)
                    
                    for col in self.returns.columns:
                        if col != 'SPY':
                            rolling_corr[f'corr_{col}_SPY'] = self.returns[col].rolling(window=corr_window).corr(spy_returns)
                    
                    # 3. Momentum features
                    momentum_windows = [5, 10, 20]
                    momentum_features = pd.DataFrame(index=self.returns.index)
                    
                    for window in momentum_windows:
                        momentum_features[f'momentum_{window}d'] = self.returns.mean(axis=1).rolling(window=window).sum()
                    
                    # Combine all enhanced features
                    enhanced_features = pd.DataFrame(index=self.returns.index)
                    enhanced_features['rolling_vol_mean'] = rolling_vol.mean(axis=1)
                    enhanced_features['rolling_vol_std'] = rolling_vol.std(axis=1)
                    
                    if not rolling_corr.empty:
                        enhanced_features['rolling_corr_mean'] = rolling_corr.mean(axis=1)
                        enhanced_features['rolling_corr_std'] = rolling_corr.std(axis=1)
                    
                    for col in momentum_features.columns:
                        enhanced_features[col] = momentum_features[col]
                    
                    # Align and combine with selected features
                    enhanced_features, selected_features = enhanced_features.align(selected_features, join='inner', axis=0)
                    
                    if not enhanced_features.empty:
                        combined_features = pd.concat([selected_features, enhanced_features], axis=1)
                        
                        # Drop rows with NaN values
                        combined_features = combined_features.dropna()
                        
                        # Log how many rows were dropped
                        rows_dropped = len(selected_features) - len(combined_features)
                        if rows_dropped > 0:
                            self.log(f"Dropped {rows_dropped} rows due to NaNs after adding enhanced features.")
                        
                        selected_features = combined_features
        
        # Store the prepared features for later use in consensus metrics
        # Note: This self.clustering_features will be overwritten by each method if they call prepare_features again
        # It might be better to pass features explicitly or store them per method
        self.clustering_features = selected_features
        
        self.log(f"Prepared features with shape: {selected_features.shape}")
        return selected_features
    
    def detect_regimes_hmm(self, n_regimes=3, selected_features=None):
        """
        Detect regimes using Hidden Markov Model.
        
        Parameters:
        -----------
        n_regimes : int
            Number of regimes to detect
        selected_features : pandas.DataFrame
            Features to use for regime detection. If None, uses prepare_features_for_regime_detection.
            
        Returns:
        --------
        numpy.ndarray
            Array of regime labels
        """
        start_time = time.time()
        self.log(f"Detecting regimes using HMM with {n_regimes} regimes...")
        
        # Force n_regimes to 3 as per user requirement
        n_regimes = 3
        
        if selected_features is None:
            # Use market indicators WITHOUT DeepSeek enhancements for HMM
            selected_features = self.prepare_features_for_regime_detection(
                feature_selection='market_indicators', 
                add_deepseek_features=True # Changed to False for HMM
            )
        
        if selected_features is None or selected_features.empty:
            self.log("Error: Could not prepare features for HMM.")
            return None
        
        try:
            # Scale features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(selected_features)
            
            # Create and fit HMM
            model = hmm.GaussianHMM(
                n_components=n_regimes,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )
            
            model.fit(scaled_data)
            
            # Predict hidden states
            hidden_states = model.predict(scaled_data)
            
            # Apply temporal smoothing to ensure regime continuity
            smoothed_states = self._apply_temporal_smoothing(hidden_states, window_size=5)
            
            # Store model and labels
            self.models['hmm'] = {
                'model': model,
                'scaler': scaler
            }
            self.regime_labels['hmm'] = pd.Series(smoothed_states, index=selected_features.index)
            
            # Calculate metrics
            avg_duration = self._calculate_regime_durations(smoothed_states)
            metrics = self._calculate_clustering_metrics(scaled_data, smoothed_states)
            self.silhouette_scores['hmm'] = metrics['silhouette']
            
            # Log results
            self.log(f"HMM training completed in {time.time() - start_time:.2f} seconds")
            self.log(f"Clustering metrics: {metrics}")
            self.log("Average Regime Durations:")
            for regime, duration in avg_duration.items():
                self.log(f"  Regime {regime}: {duration:.2f} periods")
            
            # Save model
            model_path = os.path.join(self.output_dir, 'hmm_model.joblib')
            joblib.dump({
                'model': model,
                'scaler': scaler
            }, model_path)
            self.log(f"HMM model saved to {model_path}")
            
            # Save regime labels to CSV
            labels_df = pd.DataFrame({'hmm': self.regime_labels['hmm']})
            labels_path = os.path.join(self.output_dir, 'hmm_regime_labels.csv')
            labels_df.to_csv(labels_path, index=True, index_label="Date", date_format="%Y-%m-%d %H:%M:%S")
            self.log(f"HMM regime labels saved to {labels_path}")
            
            # Characterize regimes
            self._characterize_regimes('hmm', smoothed_states)
            
            return smoothed_states
            
        except Exception as e:
            self.log(f"Error in HMM: {str(e)}")
            return None
    
    def detect_regimes_gmm(self, n_regimes=3, selected_features=None):
        """
        Detect regimes using Gaussian Mixture Model.
        
        Parameters:
        -----------
        n_regimes : int
            Number of regimes to detect
        selected_features : pandas.DataFrame
            Features to use for regime detection. If None, uses prepare_features_for_regime_detection.
            
        Returns:
        --------
        numpy.ndarray
            Array of regime labels
        """
        start_time = time.time()
        self.log(f"Detecting regimes using GMM with {n_regimes} regimes...")
        
        # Force n_regimes to 3 as per user requirement
        n_regimes = 3
        
        if selected_features is None:
            # Use market indicators WITHOUT DeepSeek enhancements for GMM
            selected_features = self.prepare_features_for_regime_detection(
                feature_selection='market_indicators', 
                add_deepseek_features=True # Changed to False for GMM
            )
        
        if selected_features is None or selected_features.empty:
            self.log("Error: Could not prepare features for GMM.")
            return None
        
        try:
            # Scale features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(selected_features)
            
            # Create and fit GMM
            model = GaussianMixture(
                n_components=n_regimes,
                covariance_type="full",
                n_init=10,
                random_state=42
            )
            
            model.fit(scaled_data)
            
            # Predict regimes
            regimes = model.predict(scaled_data)
            
            # Apply temporal smoothing
            smoothed_regimes = self._apply_temporal_smoothing(regimes, window_size=5)
            
            # Store model and labels
            self.models['gmm'] = {
                'model': model,
                'scaler': scaler
            }
            self.regime_labels['gmm'] = pd.Series(smoothed_regimes, index=selected_features.index)
            
            # Calculate metrics
            avg_duration = self._calculate_regime_durations(smoothed_regimes)
            metrics = self._calculate_clustering_metrics(scaled_data, smoothed_regimes)
            self.silhouette_scores['gmm'] = metrics['silhouette']
            
            # Log results
            self.log(f"GMM training completed in {time.time() - start_time:.2f} seconds")
            self.log(f"Clustering metrics: {metrics}")
            self.log("Average Regime Durations:")
            for regime, duration in avg_duration.items():
                self.log(f"  Regime {regime}: {duration:.2f} periods")
            
            # Save model
            model_path = os.path.join(self.output_dir, 'gmm_model.joblib')
            joblib.dump({
                'model': model,
                'scaler': scaler
            }, model_path)
            self.log(f"GMM model saved to {model_path}")
            
            # Save regime labels to CSV
            labels_df = pd.DataFrame({'gmm': self.regime_labels['gmm']})
            labels_path = os.path.join(self.output_dir, 'gmm_regime_labels.csv')
            labels_df.to_csv(labels_path, index=True, index_label="Date", date_format="%Y-%m-%d %H:%M:%S")
            self.log(f"GMM regime labels saved to {labels_path}")
            
            # Characterize regimes
            self._characterize_regimes('gmm', smoothed_regimes)
            
            return smoothed_regimes
            
        except Exception as e:
            self.log(f"Error in GMM: {str(e)}")
            return None
    
    def _build_tcn_encoder(self, input_shape, n_latent_dims=16):
        """
        Build a Temporal Convolutional Network (TCN) encoder.
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of the input data (sequence_length, n_features)
        n_latent_dims : int
            Number of dimensions for the latent space
            
        Returns:
        --------
        tensorflow.keras.models.Model
            TCN encoder model
        """
        input_layer = Input(shape=input_shape)
        
        # TCN layers with dilated convolutions
        x = Conv1D(filters=64, kernel_size=3, dilation_rate=1, activation='relu', padding='causal')(input_layer)
        x = Conv1D(filters=64, kernel_size=3, dilation_rate=2, activation='relu', padding='causal')(x)
        x = Conv1D(filters=64, kernel_size=3, dilation_rate=4, activation='relu', padding='causal')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        
        x = Conv1D(filters=128, kernel_size=3, dilation_rate=1, activation='relu', padding='causal')(x)
        x = Conv1D(filters=128, kernel_size=3, dilation_rate=2, activation='relu', padding='causal')(x)
        x = Conv1D(filters=128, kernel_size=3, dilation_rate=4, activation='relu', padding='causal')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.2)(x)
        
        x = Flatten()(x)
        latent_space = Dense(n_latent_dims, activation='relu')(x)
        
        encoder = Model(input_layer, latent_space, name="tcn_encoder")
        return encoder
    
    def _create_sequences(self, data, sequence_length):
        """
        Create sequences from time series data.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Input data (n_samples, n_features)
        sequence_length : int
            Length of each sequence
            
        Returns:
        --------
        numpy.ndarray
            Sequences (n_sequences, sequence_length, n_features)
        """
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)
    
    def detect_regimes_tcn_kmeans(self, n_regimes=3, sequence_length=20, n_latent_dims=16, epochs=50, batch_size=32, selected_features=None):
        """
        Detect regimes using TCN encoder + KMeans clustering.
        
        Parameters:
        -----------
        n_regimes : int
            Number of regimes to detect
        sequence_length : int
            Length of sequences for TCN input
        n_latent_dims : int
            Dimensionality of TCN latent space
        epochs : int
            Number of training epochs for TCN
        batch_size : int
            Batch size for TCN training
        selected_features : pandas.DataFrame
            Features to use for regime detection. If None, uses prepare_features_for_regime_detection.
            
        Returns:
        --------
        numpy.ndarray
            Array of regime labels
        """
        start_time = time.time()
        self.log(f"Detecting regimes using TCN+KMeans with {n_regimes} regimes...")
        
        # Force n_regimes to 3 as per user requirement
        n_regimes = 3
        
        if selected_features is None:
            # Use all features WITH DeepSeek enhancements for TCN
            selected_features = self.prepare_features_for_regime_detection(
                feature_selection='all', 
                add_deepseek_features=True # Keep True for TCN
            )
        
        if selected_features is None or selected_features.empty:
            self.log("Error: Could not prepare features for TCN+KMeans.")
            return None
        
        try:
            # Scale features
            scaler = RobustScaler() # Use RobustScaler for potentially non-Gaussian features
            scaled_data = scaler.fit_transform(selected_features)
            
            # Create sequences
            sequences = self._create_sequences(scaled_data, sequence_length)
            
            if sequences.shape[0] == 0:
                self.log("Error: Not enough data to create sequences for TCN.")
                return None
            
            # Build TCN encoder
            input_shape = (sequence_length, scaled_data.shape[1])
            encoder = self._build_tcn_encoder(input_shape, n_latent_dims)
            
            # Build Autoencoder (for unsupervised training)
            decoder_input = Input(shape=(n_latent_dims,))
            # Simple decoder for reconstruction (can be more complex)
            x = Dense(128, activation='relu')(decoder_input)
            x = Dense(input_shape[0] * input_shape[1], activation='linear')(x)
            x = tf.keras.layers.Reshape(input_shape)(x)
            decoder = Model(decoder_input, x, name="decoder")
            
            autoencoder = Model(encoder.input, decoder(encoder.output), name="tcn_autoencoder")
            autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train Autoencoder
            self.log("Training TCN Autoencoder...")
            autoencoder.fit(sequences, sequences, epochs=epochs, batch_size=batch_size, verbose=0)
            
            # Get latent representations
            latent_representations = encoder.predict(sequences)
            
            # Apply KMeans to latent space
            kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            regimes = kmeans.fit_predict(latent_representations)
            
            # Align regimes with original data index (adjust for sequence length)
            # The regimes correspond to the *end* of each sequence
            regime_index = selected_features.index[sequence_length - 1:]
            
            if len(regimes) != len(regime_index):
                self.log(f"Warning: Length mismatch between TCN regimes ({len(regimes)}) and index ({len(regime_index)}). Adjusting.")
                min_len = min(len(regimes), len(regime_index))
                regimes = regimes[:min_len]
                regime_index = regime_index[:min_len]
            
            # Apply temporal smoothing
            smoothed_regimes = self._apply_temporal_smoothing(regimes, window_size=5)
            
            # Store model and labels
            self.models['tcn_kmeans'] = {
                'encoder': encoder,
                'kmeans': kmeans,
                'scaler': scaler,
                'sequence_length': sequence_length
            }
            self.regime_labels['tcn_kmeans'] = pd.Series(smoothed_regimes, index=regime_index)
            
            # Calculate metrics (using latent space)
            avg_duration = self._calculate_regime_durations(smoothed_regimes)
            metrics = self._calculate_clustering_metrics(latent_representations, smoothed_regimes)
            self.silhouette_scores['tcn_kmeans'] = metrics['silhouette']
            
            # Log results
            self.log(f"TCN+KMeans training completed in {time.time() - start_time:.2f} seconds")
            self.log(f"Clustering metrics (latent space): {metrics}")
            self.log("Average Regime Durations:")
            for regime, duration in avg_duration.items():
                self.log(f"  Regime {regime}: {duration:.2f} periods")
            
            # Save models
            encoder_path = os.path.join(self.output_dir, 'tcn_encoder.keras') # Use .keras format
            encoder.save(encoder_path)
            kmeans_path = os.path.join(self.output_dir, 'tcn_kmeans_model.joblib')
            joblib.dump({
                'kmeans': kmeans,
                'scaler': scaler,
                'sequence_length': sequence_length
            }, kmeans_path)
            self.log(f"TCN encoder saved to {encoder_path}")
            self.log(f"TCN KMeans model saved to {kmeans_path}")
            
            # Save regime labels to CSV
            labels_df = pd.DataFrame({'tcn_kmeans': self.regime_labels['tcn_kmeans']})
            labels_path = os.path.join(self.output_dir, 'tcn_kmeans_regime_labels.csv')
            labels_df.to_csv(labels_path, index=True, index_label="Date", date_format="%Y-%m-%d %H:%M:%S")
            self.log(f"TCN+KMeans regime labels saved to {labels_path}")
            
            # Characterize regimes
            self._characterize_regimes('tcn_kmeans', smoothed_regimes)
            
            return smoothed_regimes
            
        except Exception as e:
            self.log(f"Error in TCN+KMeans: {str(e)}")
            return None
    
    def detect_regimes_umap_kmeans(self, n_regimes=3, umap_params=None, selected_features=None):
        """
        Detect regimes using UMAP + KMeans.

        Parameters:
        -----------
        n_regimes : int
            Number of regimes to detect
        umap_params : dict, optional
            Parameters for UMAP (e.g., {'n_neighbors': 15, 'min_dist': 0.1, 'n_components': 5})
            If None, performs a grid search over default parameters.
        selected_features : pandas.DataFrame
            Features to use for regime detection. If None, uses prepare_features_for_regime_detection.
            
        Returns:
        --------
        numpy.ndarray
            Array of regime labels
        """
        start_time = time.time()
        self.log(f"Detecting regimes using UMAP+KMeans with {n_regimes} regimes...")
        
        # Force n_regimes to 3 as per user requirement
        n_regimes = 3
        
        if selected_features is None:
            # Use all features WITH DeepSeek enhancements for UMAP
            selected_features = self.prepare_features_for_regime_detection(
                feature_selection='all', 
                add_deepseek_features=True # Keep True for UMAP
            )
        
        if selected_features is None or selected_features.empty:
            self.log("Error: Could not prepare features for UMAP+KMeans.")
            return None
        
        try:
            # Scale features
            scaler = RobustScaler()
            scaled_data = scaler.fit_transform(selected_features)
            
            best_score = -1
            best_params = None
            best_embedding = None
            best_labels = None
            
            if umap_params is None:
                # Define parameter grid for UMAP
                param_grid = {
                    'n_neighbors': [10,15,20,25, 30, 50],
                    'min_dist': [0.0, 0.1, 0.5],
                    'n_components': [3, 5, 10] # Test different embedding dimensions
                }
                self.log("Performing UMAP parameter grid search...")
            else:
                param_grid = [umap_params]
                self.log("Using provided UMAP parameters...")
            
            for params in ParameterGrid(param_grid):
                self.log(f"  Testing UMAP params: {params}")
                try:
                    # Apply UMAP
                    umap_reducer = UMAP(
                        n_neighbors=params['n_neighbors'],
                        min_dist=params['min_dist'],
                        n_components=params['n_components'],
                        random_state=42
                    )
                    embedding = umap_reducer.fit_transform(scaled_data)
                    
                    # Apply KMeans to UMAP embedding
                    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(embedding)
                    
                    # Calculate silhouette score
                    score = silhouette_score(embedding, labels)
                    self.log(f"    Silhouette Score: {score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_embedding = embedding
                        best_labels = labels
                        best_umap_reducer = umap_reducer
                        best_kmeans = kmeans
                        
                except Exception as e:
                    self.log(f"    Error with params {params}: {str(e)}")
                    continue
            
            if best_labels is None:
                self.log("Error: UMAP+KMeans failed for all parameter combinations.")
                return None
            
            self.log(f"Best UMAP params: {best_params} with Silhouette Score: {best_score:.4f}")
            
            # Apply temporal smoothing
            smoothed_regimes = self._apply_temporal_smoothing(best_labels, window_size=5)
            
            # Store model and labels
            self.models['umap_kmeans'] = {
                'umap': best_umap_reducer,
                'kmeans': best_kmeans,
                'scaler': scaler,
                'params': best_params
            }
            self.regime_labels['umap_kmeans'] = pd.Series(smoothed_regimes, index=selected_features.index)
            
            # Calculate metrics (using UMAP embedding)
            avg_duration = self._calculate_regime_durations(smoothed_regimes)
            metrics = self._calculate_clustering_metrics(best_embedding, smoothed_regimes)
            self.silhouette_scores['umap_kmeans'] = metrics['silhouette'] # Use the best score found
            
            # Log results
            self.log(f"UMAP+KMeans training completed in {time.time() - start_time:.2f} seconds")
            self.log(f"Clustering metrics (UMAP space): {metrics}")
            self.log("Average Regime Durations:")
            for regime, duration in avg_duration.items():
                self.log(f"  Regime {regime}: {duration:.2f} periods")
            
            # Save models
            model_path = os.path.join(self.output_dir, 'umap_kmeans_model.joblib')
            joblib.dump({
                'umap': best_umap_reducer,
                'kmeans': best_kmeans,
                'scaler': scaler,
                'params': best_params
            }, model_path)
            self.log(f"UMAP+KMeans model saved to {model_path}")
            
            # Save regime labels to CSV
            labels_df = pd.DataFrame({'umap_kmeans': self.regime_labels['umap_kmeans']})
            labels_path = os.path.join(self.output_dir, 'umap_kmeans_regime_labels.csv')
            labels_df.to_csv(labels_path, index=True, index_label="Date", date_format="%Y-%m-%d %H:%M:%S")
            self.log(f"UMAP+KMeans regime labels saved to {labels_path}")
            
            # Characterize regimes
            self._characterize_regimes('umap_kmeans', smoothed_regimes)
            
            return smoothed_regimes
            
        except Exception as e:
            self.log(f"Error in UMAP+KMeans: {str(e)}")
            return None
    
    def _apply_temporal_smoothing(self, labels, window_size=5):
        """
        Apply temporal smoothing using a rolling mode.
        
        Parameters:
        -----------
        labels : numpy.ndarray or pandas.Series
            Array of regime labels
        window_size : int
            Size of the rolling window
            
        Returns:
        --------
        numpy.ndarray
            Smoothed regime labels
        """
        self.log(f"Applying temporal smoothing with window size {window_size}...")
        if not isinstance(labels, pd.Series):
            labels_series = pd.Series(labels)
        else:
            labels_series = labels
            
        # Apply rolling mode
        smoothed_labels = labels_series.rolling(window=window_size, center=True, min_periods=1).apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan, raw=False)
        
        # Forward fill any NaNs introduced at the edges or by empty modes
        smoothed_labels = smoothed_labels.fillna(method='ffill').fillna(method='bfill')
        
        return smoothed_labels.astype(int).values
    
    def _calculate_regime_durations(self, labels):
        """
        Calculate the average duration of each regime.
        
        Parameters:
        -----------
        labels : numpy.ndarray
            Array of regime labels
            
        Returns:
        --------
        dict
            Dictionary mapping regime label to average duration
        """
        if labels is None or len(labels) == 0:
            return {}
            
        durations = {regime: [] for regime in np.unique(labels)}
        current_regime = labels[0]
        current_duration = 0
        
        for label in labels:
            if label == current_regime:
                current_duration += 1
            else:
                if current_regime in durations:
                    durations[current_regime].append(current_duration)
                current_regime = label
                current_duration = 1
        
        # Add the last duration
        if current_regime in durations:
            durations[current_regime].append(current_duration)
            
        avg_durations = {regime: np.mean(d) if d else 0 for regime, d in durations.items()}
        return avg_durations
    
    def _calculate_clustering_metrics(self, data, labels):
        """
        Calculate clustering evaluation metrics.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Data used for clustering
        labels : numpy.ndarray
            Cluster labels
            
        Returns:
        --------
        dict
            Dictionary containing silhouette, calinski_harabasz, and davies_bouldin scores
        """
        metrics = {
            'silhouette': np.nan,
            'calinski_harabasz': np.nan,
            'davies_bouldin': np.nan
        }
        
        try:
            # Ensure there are at least 2 unique labels and enough samples
            unique_labels = np.unique(labels)
            if len(unique_labels) > 1 and len(labels) > len(unique_labels):
                metrics['silhouette'] = silhouette_score(data, labels)
                metrics['calinski_harabasz'] = calinski_harabasz_score(data, labels)
                metrics['davies_bouldin'] = davies_bouldin_score(data, labels)
            else:
                self.log("Warning: Cannot calculate clustering metrics. Need at least 2 clusters and more samples than clusters.")
        except Exception as e:
            self.log(f"Error calculating clustering metrics: {str(e)}")
            
        return metrics
    
    def _characterize_regimes(self, method_name, labels):
        """
        Characterize regimes based on average returns and volatility.
        
        Parameters:
        -----------
        method_name : str
            Name of the regime detection method (e.g., 'hmm', 'gmm')
        labels : numpy.ndarray or pandas.Series
            Array or Series of regime labels
        """
        self.log(f"Characterizing regimes for {method_name}...")
        
        if self.returns is None:
            self.log("Error: Returns data not available for characterization.")
            return
        
        if labels is None:
            self.log(f"Error: No labels provided for {method_name} characterization.")
            return
        
        try:
            # Ensure labels are a Pandas Series with the correct index
            if not isinstance(labels, pd.Series):
                # For TCN+KMeans, we need to handle the sequence length offset
                if method_name == 'tcn_kmeans' and method_name in self.regime_labels:
                    # Use the existing index from the stored regime labels
                    labels_series = pd.Series(labels, index=self.regime_labels[method_name].index, name=method_name)
                elif self.clustering_features is None:
                    self.log(f"Error: Cannot align labels for {method_name}. Clustering features missing.")
                    return
                else:
                    # For other methods, use the clustering features index but handle length mismatch
                    if len(labels) != len(self.clustering_features):
                        self.log(f"Warning: Length mismatch for {method_name}. Labels length: {len(labels)}, Features length: {len(self.clustering_features)}")
                        # Use the last N elements of clustering_features index where N is the length of labels
                        if len(labels) < len(self.clustering_features):
                            index_to_use = self.clustering_features.index[-len(labels):]
                        else:
                            # If labels are somehow longer, truncate them
                            labels = labels[:len(self.clustering_features)]
                            index_to_use = self.clustering_features.index
                    else:
                        index_to_use = self.clustering_features.index
                    
                    labels_series = pd.Series(labels, index=index_to_use, name=method_name)
            else:
                labels_series = labels
            
            # Align returns with labels using intersection of indices
            common_index = self.returns.index.intersection(labels_series.index)
            if common_index.empty:
                self.log(f"Error: No common index between returns and labels for {method_name}.")
                return
                
            aligned_returns = self.returns.loc[common_index]
            aligned_labels = labels_series.loc[common_index]
            
            # Double-check alignment
            if len(aligned_labels) != len(aligned_returns):
                self.log(f"Warning: Length mismatch after aligning returns and labels for {method_name}.")
                self.log(f"Aligned labels length: {len(aligned_labels)}, Aligned returns length: {len(aligned_returns)}")
                # Find the intersection again to ensure perfect alignment
                final_index = aligned_returns.index.intersection(aligned_labels.index)
                aligned_returns = aligned_returns.loc[final_index]
                aligned_labels = aligned_labels.loc[final_index]
                
                if len(aligned_labels) == 0:
                    self.log(f"Error: No common dates left after strict alignment for {method_name}.")
                    return
            
            # Calculate average daily return and volatility for each regime
            regime_stats = pd.DataFrame(index=np.unique(aligned_labels))
            regime_stats['Avg Daily Return'] = aligned_returns.groupby(aligned_labels).mean().mean(axis=1)
            regime_stats['Avg Daily Volatility'] = aligned_returns.groupby(aligned_labels).std().mean(axis=1)
            
            # Define regime characteristics based on stats
            # Simple heuristic: High return = Bullish, Low return/High vol = Bearish, Mid = Risky/Neutral
            sorted_returns = regime_stats['Avg Daily Return'].sort_values()
            
            regime_mapping = {}
            if len(sorted_returns) == 3:
                regime_mapping[sorted_returns.index[0]] = 'Bearish'
                regime_mapping[sorted_returns.index[1]] = 'Risky/Neutral'
                regime_mapping[sorted_returns.index[2]] = 'Bullish'
            elif len(sorted_returns) > 0:
                # Fallback for different number of regimes
                median_ret = regime_stats['Avg Daily Return'].median()
                median_vol = regime_stats['Avg Daily Volatility'].median()
                for regime in regime_stats.index:
                    ret = regime_stats.loc[regime, 'Avg Daily Return']
                    vol = regime_stats.loc[regime, 'Avg Daily Volatility']
                    if ret > median_ret and vol < median_vol:
                        regime_mapping[regime] = 'Bullish'
                    elif ret < median_ret and vol > median_vol:
                        regime_mapping[regime] = 'Bearish'
                    else:
                        regime_mapping[regime] = 'Risky/Neutral'
            else:
                 self.log(f"Warning: Could not determine regime characteristics for {method_name} due to empty stats.")
                 return # Cannot proceed if stats are empty

            regime_stats['Label'] = regime_stats.index.map(regime_mapping)
            
            self.regime_characteristics[method_name] = regime_stats
            self.log(f"Regime Characteristics for {method_name}:\n{regime_stats}")
            
            # Save characteristics to CSV
            char_path = os.path.join(self.output_dir, f'{method_name}_regime_characteristics.csv')
            regime_stats.to_csv(char_path, index=True, index_label="Regime")
            self.log(f"Regime characteristics saved to {char_path}")
            
        except Exception as e:
            self.log(f"Error characterizing regimes for {method_name}: {str(e)}")
            # Add more detailed error logging if needed
            import traceback
            self.log(traceback.format_exc())
    
    def create_consensus_regimes(self, methods=['hmm', 'gmm', 'umap_kmeans']): # Removed tcn_kmeans temporarily if it has issues
        """
        Create consensus regimes using majority voting.
        
        Parameters:
        -----------
        methods : list
            List of method names to include in the consensus
            
        Returns:
        --------
        pandas.Series
            Series of consensus regime labels
        """
        start_time = time.time()
        self.log("Creating consensus regimes...")
        
        valid_labels = {m: self.regime_labels[m] for m in methods if m in self.regime_labels and self.regime_labels[m] is not None and not self.regime_labels[m].empty}
        
        if len(valid_labels) < 2:
            self.log("Error: Need at least two valid regime label sets for consensus.")
            return None
        
        try:
            # Align all label series to a common index (intersection)
            labels_df = pd.DataFrame(valid_labels)
            # Align using outer join first to keep all dates, then maybe intersect?
            # Let's try aligning pairs first, might be more robust
            
            # Alternative: Use pd.concat and align
            aligned_df = pd.concat(valid_labels, axis=1, join='inner') # Use inner join for common dates
            
            if aligned_df.empty:
                 self.log("Error: No common dates found among the selected methods for consensus.")
                 return None
            
            # Calculate the mode (most frequent label) for each time step
            # Ensure mode calculation handles potential NaNs or ties appropriately
            consensus_labels = aligned_df.mode(axis=1)
            
            # Handle cases where mode returns multiple values (ties) - take the first one
            if consensus_labels.shape[1] > 1:
                consensus_labels = consensus_labels[0]
            else:
                consensus_labels = consensus_labels.squeeze()
                
            # Handle potential NaNs if mode was empty for some rows (shouldn't happen with inner join?)
            consensus_labels = consensus_labels.fillna(method='ffill').fillna(method='bfill')
            
            if consensus_labels.empty:
                 self.log("Error: Consensus labels calculation resulted in an empty Series.")
                 return None
            
            consensus_labels = consensus_labels.astype(int)
            
            # Apply temporal smoothing
            smoothed_consensus = self._apply_temporal_smoothing(consensus_labels, window_size=5)
            smoothed_consensus_series = pd.Series(smoothed_consensus, index=aligned_df.index)
            
            self.regime_labels['consensus'] = smoothed_consensus_series
            
            # Calculate metrics (using an average or representative feature space? Difficult.)
            # For simplicity, we won't calculate metrics for consensus directly here.
            avg_duration = self._calculate_regime_durations(smoothed_consensus)
            
            # Log results
            self.log(f"Consensus creation completed in {time.time() - start_time:.2f} seconds")
            self.log("Average Consensus Regime Durations:")
            for regime, duration in avg_duration.items():
                self.log(f"  Regime {regime}: {duration:.2f} periods")
            
            # Save consensus labels to CSV
            labels_df = pd.DataFrame({'consensus': smoothed_consensus_series})
            labels_path = os.path.join(self.output_dir, 'consensus_regime_labels.csv')
            labels_df.to_csv(labels_path, index=True, index_label="Date", date_format="%Y-%m-%d %H:%M:%S")
            self.log(f"Consensus regime labels saved to {labels_path}")
            
            # Characterize consensus regimes
            self._characterize_regimes('consensus', smoothed_consensus_series)
            
            return smoothed_consensus_series
            
        except Exception as e:
            self.log(f"Error creating consensus regimes: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return None
    
    def plot_regimes(self, method_name, price_data=None, returns_data=None):
        """
        Plot price or returns data overlaid with detected regimes.
        
        Parameters:
        -----------
        method_name : str
            Name of the method whose regimes to plot (e.g., 'hmm', 'consensus')
        price_data : pandas.DataFrame, optional
            Price data to plot (e.g., self.aligned_prices['SPY'])
        returns_data : pandas.DataFrame, optional
            Returns data to plot (e.g., self.returns['SPY'])
        """
        if method_name not in self.regime_labels or self.regime_labels[method_name] is None:
            self.log(f"No regime labels found for method: {method_name}")
            return
            
        labels = self.regime_labels[method_name]
        n_regimes = len(labels.unique())
        
        # Use a predefined colormap suitable for discrete categories
        # Using 'viridis', 'plasma', 'inferno', 'magma', 'cividis' might not be ideal for regimes
        # Let's try 'tab10' or 'Set1'
        try:
            colors = plt.get_cmap('tab10', n_regimes)
        except ValueError:
             self.log(f"Warning: Colormap 'tab10' might not support {n_regimes} distinct colors. Using default.")
             colors = plt.get_cmap(None, n_regimes) # Use default if specific map fails
        
        fig, ax = plt.subplots(figsize=(15, 7))
        
        data_to_plot = None
        y_label = ""
        if price_data is not None:
            data_to_plot = price_data
            y_label = "Price"
        elif returns_data is not None:
            data_to_plot = returns_data
            y_label = "Returns"
        else:
            self.log("No price or returns data provided for plotting.")
            return
            
        # Align data with labels
        common_index = data_to_plot.index.intersection(labels.index)
        if common_index.empty:
            self.log(f"Error: No common index between data and labels for {method_name} plotting.")
            return
            
        aligned_data = data_to_plot.loc[common_index]
        aligned_labels = labels.loc[common_index]
        
        ax.plot(aligned_data.index, aligned_data.values, label=y_label, color='black', alpha=0.7)
        
        # Add shaded regions for regimes
        for i in range(n_regimes):
            regime_indices = aligned_data.index[aligned_labels == i]
            # Plot vertical spans for each continuous block in the regime
            start_date = None
            for j, date in enumerate(regime_indices):
                if start_date is None:
                    start_date = date
                # Check if the next index is consecutive or if it's the last one
                is_last = (j == len(regime_indices) - 1)
                next_date_is_not_consecutive = not is_last and (regime_indices[j+1] - date).days > 1 # Adjust threshold if needed
                
                if is_last or next_date_is_not_consecutive:
                    end_date = date
                    ax.axvspan(start_date, end_date, color=colors(i), alpha=0.3, label=f'Regime {i}' if start_date == regime_indices[0] else "")
                    start_date = None # Reset for the next block
        
        ax.set_title(f'{y_label} with {method_name.upper()} Regimes')
        ax.set_xlabel("Date")
        ax.set_ylabel(y_label)
        
        # Improve legend handling
        handles, labels_legend = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_legend, handles)) # Remove duplicate labels
        ax.legend(by_label.values(), by_label.keys())
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f'{method_name}_regimes_plot.png')
        plt.savefig(plot_path)
        self.log(f"Regime plot saved to {plot_path}")
        plt.close(fig)
    
    def run_all_methods(self):
        """
        Run all implemented regime detection methods.
        """
        self.log("\n--- Running HMM ---")
        self.detect_regimes_hmm()
        
        self.log("\n--- Running GMM ---")
        self.detect_regimes_gmm()
        
        self.log("\n--- Running TCN+KMeans ---")
        self.detect_regimes_tcn_kmeans()
        
        self.log("\n--- Running UMAP+KMeans ---")
        self.detect_regimes_umap_kmeans()
        
        self.log("\n--- Creating Consensus ---")
        self.create_consensus_regimes()
        
        self.log("\n--- Generating Plots ---")
        spy_price = self.aligned_prices['SPY'] if self.aligned_prices is not None and 'SPY' in self.aligned_prices.columns else None
        if spy_price is not None:
            for method in self.regime_labels.keys():
                if self.regime_labels[method] is not None:
                    self.plot_regimes(method, price_data=spy_price)
        else:
            self.log("SPY price data not found, skipping plotting.")
            
        self.log("\n--- Final Silhouette Scores ---")
        for method, score in self.silhouette_scores.items():
            self.log(f"  {method}: Silhouette Score = {score:.4f}")
            
        # Save all labels together
        all_labels_df = pd.DataFrame(self.regime_labels)
        all_labels_path = os.path.join(self.output_dir, 'all_regime_labels.csv')
        all_labels_df.to_csv(all_labels_path, index=True, index_label="Date", date_format="%Y-%m-%d %H:%M:%S")
        self.log(f"\nAll regime labels saved to {all_labels_path}")

# Main execution block
if __name__ == "__main__":
    detector = MarketRegimeDetector()
    
    if detector.load_data():
        detector.run_all_methods()
        detector.log("\nMarket regime detection process completed.")
    else:
        detector.log("\nMarket regime detection failed due to data loading errors.")

