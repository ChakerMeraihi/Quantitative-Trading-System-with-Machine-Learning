#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Financial Data Preprocessing Pipeline (Fixed for Data Leakage with Performance Optimization - Version 3)

This script implements a comprehensive preprocessing pipeline for financial time series data,
including data loading, cleaning, feature engineering, and preparation for market regime detection.
All features are calculated with proper time lag to prevent data leakage.
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
from scipy import stats
import time

import json # Added for metadata saving

warnings.filterwarnings("ignore")


class FinancialDataPreprocessor:
    """
    A class for preprocessing financial time series data for market regime detection
    and algorithmic trading strategies, with strict prevention of data leakage.
    """

    def __init__(self, data_dir, output_dir=None):
        """
        Initialize the preprocessor with data directory paths.

        Parameters:
        -----------
        data_dir : str
            Directory containing the raw CSV files
        output_dir : str, optional
            Directory for saving processed data
        """
        self.data_dir = data_dir
        self.output_dir = output_dir if output_dir else os.path.join(os.path.dirname(data_dir), "processed_data")

        # Create output directory if it doesn"t exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Initialize data containers
        self.raw_data = {}
        self.cleaned_data = {}
        self.aligned_data = None
        self.features = None
        self.target = None

        # Initialize metadata attributes (as per feedback)
        self.symbols_processed = None
        self.sample_size_processed = None
        self.window_sizes_used = None
        self.outlier_method_used = None
        self.outlier_threshold_used = None
        self.outlier_detection_method_used = None # Added for consistency
        self.alignment_strategy_used = None # Added for consistency

        # Logging
        self.log_file = os.path.join(self.output_dir, "preprocessing_log.txt")
        with open(self.log_file, "w") as f:
            f.write(f"Preprocessing started at: {datetime.now()}\n")
            f.write(f"Data directory: {self.data_dir}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")
            f.write("NOTE: All features are calculated with proper time lag to prevent data leakage\n\n")

    def log(self, message):
        """Add message to log file"""
        with open(self.log_file, "a") as f:
            f.write(f"{datetime.now()}: {message}\n")
        print(message)

    def load_data(self, symbols=None, sample_size=None):
        """
        Load CSV files from the data directory.

        Parameters:
        -----------
        symbols : list, optional
            List of specific symbols to load. If None, load all available files.
        sample_size : int, optional
            Number of random symbols to load for testing. If None, load all symbols.

        Returns:
        --------
        dict
            Dictionary with symbols as keys and DataFrames as values
        """
        self.log("Loading data files...")

        # Get list of all CSV files
        all_files = glob.glob(os.path.join(self.data_dir, "*.csv"))

        # Filter by symbols if provided
        if symbols:
            files_to_load = [f for f in all_files if os.path.basename(f).replace(".csv", "") in symbols]
        else:
            files_to_load = all_files

        # Take a random sample if requested
        if sample_size and sample_size < len(files_to_load):
            np.random.seed(42)  # For reproducibility
            files_to_load = np.random.choice(files_to_load, sample_size, replace=False)

        # Load each file
        for file_path in files_to_load:
            symbol = os.path.basename(file_path).replace(".csv", "")
            try:
                df = pd.read_csv(file_path)
                # Convert timestamp to datetime
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                # Set timestamp as index
                df.set_index("timestamp", inplace=True)
                # Sort by timestamp to ensure chronological order
                df = df.sort_index()
                # Store in raw_data dictionary
                self.raw_data[symbol] = df

            except Exception as e:
                self.log(f"Error loading {symbol}: {str(e)}")

        self.log(f"Loaded {len(self.raw_data)} symbols")
        return self.raw_data

    def detect_missing_values(self):
        """
        Detect missing values in the loaded data.

        Returns:
        --------
        dict
            Dictionary with symbols as keys and missing value counts as values
        """
        self.log("Detecting missing values...")
        missing_values = {}

        for symbol, df in self.raw_data.items():
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                missing_values[symbol] = missing_count

        if missing_values:
            self.log(f"Found missing values in {len(missing_values)} symbols")
            for symbol, count in missing_values.items():
                self.log(f"  {symbol}: {count} missing values")
        else:
            self.log("No missing values found in the dataset")

        return missing_values

    def handle_missing_values(self):
        """
        Handle missing values in the loaded data using forward fill only.
        This is time-series safe as it only uses past data.
        Includes check for leading NaNs as suggested by DeepSeek analysis.

        Returns:
        --------
        dict
            Dictionary with symbols as keys and cleaned DataFrames as values
        """
        self.log("Handling missing values...")
        self.cleaned_data = {}

        for symbol, df in self.raw_data.items():
            # Make a copy to avoid modifying the original
            cleaned_df = df.copy()

            # Check for and log leading NaNs
            if cleaned_df.iloc[0].isnull().any():
                self.log(f"  {symbol}: Warning - Data starts with missing values. Initial period might be dropped after ffill/dropna.")
                # Optional: Could implement logic to find first valid index across all columns and slice
                # first_valid = cleaned_df.apply(pd.Series.first_valid_index)
                # if first_valid.notna().all():
                #     cleaned_df = cleaned_df.loc[first_valid.max():]
                # else:
                #     self.log(f"  {symbol}: Warning - Some columns are entirely NaN.")

            # Check for missing values
            missing_before = cleaned_df.isnull().sum().sum()

            # Forward fill missing values (appropriate for time series)
            if missing_before > 0:
                cleaned_df.fillna(method="ffill", inplace=True)
                
                # Drop any remaining missing values (includes leading NaNs if ffill didn"t cover them)
                cleaned_df.dropna(inplace=True)

                missing_after = cleaned_df.isnull().sum().sum()
                self.log(f"  {symbol}: Filled/dropped {missing_before - missing_after} missing values")
                
                # Verify no NaNs remain
                if not cleaned_df.empty:
                    assert not cleaned_df.isnull().any().any(), f"NaNs remain in {symbol} after handling missing values"
                else:
                    self.log(f"  {symbol}: Warning - DataFrame became empty after handling missing values.")

            self.cleaned_data[symbol] = cleaned_df

        self.log(f"Completed missing value handling for {len(self.cleaned_data)} symbols")
        return self.cleaned_data

    def detect_outliers(self, method="zscore", threshold=3.0, min_periods=20):
        """
        Detect outliers in the price data using vectorized expanding windows.

        Parameters:
        -----------
        method : str
            Method for outlier detection ("zscore" or "iqr")
        threshold : float
            Threshold for outlier detection
        min_periods : int
            Minimum number of periods for expanding window calculations

        Returns:
        --------
        dict
            Dictionary with symbols as keys and outlier counts as values
        """
        self.log(f"Detecting outliers using vectorized {method} method with threshold {threshold}...")
        outliers = {}

        for symbol, df in self.cleaned_data.items():
            # Calculate returns for outlier detection
            returns = df["close"].pct_change().dropna()
            
            if len(returns) < min_periods:
                self.log(f"  {symbol}: Not enough data points ({len(returns)}) for outlier detection (min_periods={min_periods})")
                continue

            outlier_mask = pd.Series(False, index=returns.index)

            if method == "zscore":
                # Z-score method - using expanding window to prevent look-ahead bias
                expanding_mean = returns.expanding(min_periods=min_periods).mean()
                expanding_std = returns.expanding(min_periods=min_periods).std()
                
                # Calculate z-scores, handle potential division by zero
                z_scores = (returns - expanding_mean) / expanding_std.replace(0, np.nan) # Replace 0 std with NaN to avoid division error
                
                # Identify outliers
                outlier_mask = (z_scores.abs() > threshold)

            elif method == "iqr":
                # IQR method - using expanding window to prevent look-ahead bias
                q1 = returns.expanding(min_periods=min_periods).quantile(0.25)
                q3 = returns.expanding(min_periods=min_periods).quantile(0.75)
                iqr = q3 - q1
                
                # Calculate bounds, handle potential division by zero (iqr=0)
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                
                # Identify outliers
                outlier_mask = (returns < lower_bound) | (returns > upper_bound)

            # Count outliers (excluding initial NaNs from expanding window)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                outliers[symbol] = outlier_count

        if outliers:
            self.log(f"Found outliers in {len(outliers)} symbols")
            for symbol, count in outliers.items():
                self.log(f"  {symbol}: {count} outliers detected")
        else:
            self.log("No significant outliers found using the specified method and threshold")

        return outliers

    def handle_outliers(self, method="winsorize", threshold=3.0, min_periods=20, winsorize_limits=(0.01, 0.99)):
        """
        Handle outliers in the price data using vectorized expanding windows.
        Optimized version based on DeepSeek analysis.

        Parameters:
        -----------
        method : str
            Method for outlier handling ("winsorize" or "clip")
        threshold : float
            Threshold for outlier detection (used with z-score)
        min_periods : int
            Minimum number of periods for expanding window calculations
        winsorize_limits : tuple
            Lower and upper quantiles for winsorization (e.g., (0.01, 0.99))

        Returns:
        --------
        dict
            Dictionary with symbols as keys and cleaned DataFrames as values
        """
        start_time = time.time()
        self.log(f"Handling outliers using vectorized {method} method...")

        for symbol, df in self.cleaned_data.items():
            # Calculate returns
            returns = df["close"].pct_change()
            
            if len(returns) < min_periods:
                self.log(f"  {symbol}: Not enough data points ({len(returns)}) for outlier handling (min_periods={min_periods})")
                continue

            # --- Vectorized Outlier Detection (using expanding window) ---
            expanding_mean = returns.expanding(min_periods=min_periods).mean()
            expanding_std = returns.expanding(min_periods=min_periods).std()
            z_scores = (returns - expanding_mean) / expanding_std.replace(0, np.nan)
            outlier_mask = (z_scores.abs() > threshold).fillna(False) # Fill NaNs from initial periods
            
            outlier_count = outlier_mask.sum()
            if outlier_count == 0:
                self.log(f"  {symbol}: No outliers detected to handle.")
                continue
                
            self.log(f"  {symbol}: Detected {outlier_count} outliers to handle.")

            # --- Vectorized Outlier Handling --- 
            if method == "winsorize":
                # Calculate expanding quantiles
                lower_quantile = returns.expanding(min_periods=min_periods).quantile(winsorize_limits[0])
                upper_quantile = returns.expanding(min_periods=min_periods).quantile(winsorize_limits[1])
                
                # Apply winsorization only to outliers
                # Create a copy of returns to modify
                handled_returns = returns.copy()
                handled_returns[outlier_mask] = np.clip(returns[outlier_mask], lower_quantile[outlier_mask], upper_quantile[outlier_mask])
                
                # Recalculate close prices based on handled returns
                # Get the first valid close price to start reconstruction
                first_valid_index = df["close"].first_valid_index()
                if first_valid_index is None:
                    self.log(f"  {symbol}: Cannot handle outliers, no valid start price.")
                    continue
                    
                # Initialize new close price series
                new_close = pd.Series(index=df.index, dtype=float)
                new_close.loc[first_valid_index] = df["close"].loc[first_valid_index]
                
                # Reconstruct prices iteratively (vectorization is tricky here due to dependency)
                # Use numpy arrays for faster iteration
                close_values = new_close.values
                return_values = handled_returns.values
                valid_indices = np.where(df.index >= first_valid_index)[0]
                
                for i in range(1, len(valid_indices)):
                    current_idx_loc = valid_indices[i]
                    prev_idx_loc = valid_indices[i-1]
                    # Check if previous close price is valid
                    if not np.isnan(close_values[prev_idx_loc]):
                         # Check if current return is valid
                        if not np.isnan(return_values[current_idx_loc]):
                            close_values[current_idx_loc] = close_values[prev_idx_loc] * (1 + return_values[current_idx_loc])
                        else:
                             # If return is NaN (e.g., first return), keep previous close
                            close_values[current_idx_loc] = close_values[prev_idx_loc]
                    else:
                        # If previous close is NaN, propagate NaN
                        close_values[current_idx_loc] = np.nan
                        
                df["close"] = close_values
                self.log(f"  {symbol}: Winsorized {outlier_count} outliers.")

            elif method == "clip": # Alternative: Simple clipping based on Z-score threshold
                 # Calculate clip bounds based on expanding mean/std
                lower_bound = expanding_mean - threshold * expanding_std
                upper_bound = expanding_mean + threshold * expanding_std
                
                # Apply clipping only to outliers
                handled_returns = returns.copy()
                handled_returns[outlier_mask] = np.clip(returns[outlier_mask], lower_bound[outlier_mask], upper_bound[outlier_mask])
                
                # Recalculate close prices (same reconstruction logic as winsorize)
                first_valid_index = df["close"].first_valid_index()
                if first_valid_index is None:
                    self.log(f"  {symbol}: Cannot handle outliers, no valid start price.")
                    continue
                
                new_close = pd.Series(index=df.index, dtype=float)
                new_close.loc[first_valid_index] = df["close"].loc[first_valid_index]
                
                close_values = new_close.values
                return_values = handled_returns.values
                valid_indices = np.where(df.index >= first_valid_index)[0]
                
                for i in range(1, len(valid_indices)):
                    current_idx_loc = valid_indices[i]
                    prev_idx_loc = valid_indices[i-1]
                    if not np.isnan(close_values[prev_idx_loc]):
                        if not np.isnan(return_values[current_idx_loc]):
                            close_values[current_idx_loc] = close_values[prev_idx_loc] * (1 + return_values[current_idx_loc])
                        else:
                            close_values[current_idx_loc] = close_values[prev_idx_loc]
                    else:
                        close_values[current_idx_loc] = np.nan
                        
                df["close"] = close_values
                self.log(f"  {symbol}: Clipped {outlier_count} outliers based on Z-score threshold.")

            # Drop the temporary returns column if it exists
            if "returns" in df.columns:
                 df.drop("returns", axis=1, inplace=True)
                 
            # Final check for NaNs introduced during handling
            if df["close"].isnull().any():
                nan_count = df["close"].isnull().sum()
                self.log(f"  {symbol}: Warning - {nan_count} NaNs found in close prices after outlier handling. Applying ffill.")
                df["close"].fillna(method='ffill', inplace=True)
                df.dropna(subset=["close"], inplace=True) # Drop any remaining leading NaNs

        elapsed_time = time.time() - start_time
        self.log(f"Completed outlier handling in {elapsed_time:.2f} seconds")
        return self.cleaned_data

    def align_time_series(self, strategy="union"):
        """
        Align all time series to a common time index.
        
        Parameters:
        -----------
        strategy : str
            Alignment strategy ('union' or 'intersection')
            - 'union': Keep all timestamps (more data, more NaNs)
            - 'intersection': Keep only common timestamps (less data, no NaNs)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with aligned time series
        """
        self.log("Aligning time series...")
        
        if not self.cleaned_data:
            self.log("Error: No cleaned data available. Run handle_missing_values first.")
            return None
        
        # Get all unique timestamps
        all_timestamps = set()
        common_timestamps = None
        
        for symbol, df in self.cleaned_data.items():
            if common_timestamps is None:
                common_timestamps = set(df.index)
            else:
                common_timestamps = common_timestamps.intersection(set(df.index))
            all_timestamps.update(df.index)
        
        # Create master time index based on strategy
        if strategy == "intersection":
            self.log(f"Using intersection strategy with {len(common_timestamps)} common timestamps")
            master_index = pd.DatetimeIndex(sorted(common_timestamps))
        else:  # "union"
            self.log(f"Using union strategy with {len(all_timestamps)} total timestamps")
            master_index = pd.DatetimeIndex(sorted(all_timestamps))
        
        # Create empty DataFrame with master index
        aligned_data = pd.DataFrame(index=master_index)
        
        # Add each symbol's close price as a column
        for symbol, df in self.cleaned_data.items():
            aligned_data[symbol] = df["close"]
        
        # Check for any remaining missing values
        missing_after_align = aligned_data.isnull().sum().sum()
        if missing_after_align > 0:
            self.log(f"Warning: {missing_after_align} missing values remain after alignment")
            # Drop rows with missing values introduced during alignment (time-series safe)
            aligned_data.dropna(inplace=True)
            self.log(f"Dropped rows with missing values. New shape: {aligned_data.shape}")
        
        # Verify no NaNs remain
        assert not aligned_data.isnull().any().any(), "NaNs remain in aligned data after handling"
        
        self.aligned_data = aligned_data
        self.log(f"Time series alignment complete. Shape: {aligned_data.shape}")
        
        return aligned_data

    def calculate_returns(self):
        """
        Calculate percentage returns from the aligned price data.

        Returns:
        --------
        pandas.DataFrame
            DataFrame with returns for all symbols
        """
        self.log("Calculating returns...")

        if self.aligned_data is None:
            self.log("Error: Aligned data not available. Run align_time_series first.")
            return None

        # Calculate returns using pct_change()
        returns = self.aligned_data.pct_change()

        # Drop the first row which will have NaN returns
        returns = returns.iloc[1:]

        # Align aligned_data index with returns index (important if first row was dropped)
        self.aligned_data = self.aligned_data.loc[returns.index]

        self.log(f"Returns calculated for {len(returns.columns)} symbols. Shape: {returns.shape}")
        return returns

    def engineer_features(self, window_sizes=[5, 10, 20, 50]):
        """
        Engineer features for market regime detection and trading strategies.
        All features are calculated with proper time lag to prevent data leakage.

        Parameters:
        -----------
        window_sizes : list
            List of window sizes for rolling calculations

        Returns:
        --------
        pandas.DataFrame
            DataFrame with engineered features
        """
        self.log("Engineering features...")

        # Start with returns
        returns = self.calculate_returns()

        # Initialize features DataFrame
        features = pd.DataFrame(index=returns.index)

        # Select a subset of major symbols for market indicators
        major_symbols = ["SPY", "QQQ", "IWM", "VXX", "AGG"] if all(
            s in self.aligned_data.columns for s in ["SPY", "QQQ", "IWM", "VXX", "AGG"]) else list(self.aligned_data.columns[:5])
        self.log(f"Using {major_symbols} as major market indicators")

        # For each major symbol, calculate various features
        for symbol in major_symbols:
            if symbol in self.aligned_data.columns:
                price_series = self.aligned_data[symbol]
                return_series = returns[symbol]

                # Price-based features
                for window in window_sizes:
                    # Moving averages - use closed="left" to exclude current point
                    features[f"{symbol}_MA_{window}"] = price_series.rolling(window=window, closed="left").mean()

                    # Price relative to moving average
                    features[f"{symbol}_Price_to_MA_{window}"] = price_series / features[f"{symbol}_MA_{window}"]

                    # Bollinger Bands - use closed="left" to exclude current point
                    ma = price_series.rolling(window=window, closed="left").mean()
                    std = price_series.rolling(window=window, closed="left").std()
                    features[f"{symbol}_BB_Upper_{window}"] = ma + 2 * std
                    features[f"{symbol}_BB_Lower_{window}"] = ma - 2 * std
                    features[f"{symbol}_BB_Position_{window}"] = (price_series - ma) / (2 * std)

                # Return-based features
                for window in window_sizes:
                    # Volatility (rolling standard deviation of returns) - use closed="left"
                    features[f"{symbol}_Volatility_{window}"] = return_series.rolling(window=window, closed="left").std()

                    # Momentum (rolling sum of returns) - use closed="left"
                    features[f"{symbol}_Momentum_{window}"] = return_series.rolling(window=window, closed="left").sum()

                    # RSI (Relative Strength Index) - use closed="left"
                    delta = return_series
                    gain = delta.clip(lower=0)
                    loss = -delta.clip(upper=0)
                    avg_gain = gain.rolling(window=window, closed="left").mean()
                    avg_loss = loss.rolling(window=window, closed="left").mean()
                    rs = avg_gain / avg_loss
                    features[f"{symbol}_RSI_{window}"] = 100 - (100 / (1 + rs))

                # MACD - use shift(1) to prevent look-ahead bias as per DeepSeek analysis
                if 26 < len(price_series):
                    ema12 = price_series.shift(1).ewm(span=12, adjust=False).mean()
                    ema26 = price_series.shift(1).ewm(span=26, adjust=False).mean()
                    features[f"{symbol}_MACD"] = ema12 - ema26
                    features[f"{symbol}_MACD_Signal"] = features[f"{symbol}_MACD"].ewm(span=9, adjust=False).mean()
                    features[f"{symbol}_MACD_Hist"] = features[f"{symbol}_MACD"] - features[f"{symbol}_MACD_Signal"]

        # Cross-asset features
        if len(major_symbols) > 1:
            # Correlation matrix (rolling window)
            for window in window_sizes:
                if window < len(returns):
                    # Calculate pairwise correlations for major symbols
                    for i, sym1 in enumerate(major_symbols):
                        if sym1 in returns.columns:
                            for sym2 in major_symbols[i+1:]:
                                if sym2 in returns.columns:
                                    # Calculate rolling correlation - use closed="left"
                                    corr = returns[sym1].rolling(window=window, closed="left").corr(returns[sym2])
                                    features[f"Corr_{sym1}_{sym2}_{window}"] = corr

        # Drop rows with NaN values (due to rolling windows or initial NaNs)
        features = features.dropna()

        # --- Target Variable Calculation (Post-Feature Engineering) ---
        # Calculate target variable (next period"s return) AFTER feature engineering
        # This prevents target leakage as suggested by DeepSeek analysis
        self.log("Calculating target variable (next period return)...")
        target_returns = pd.DataFrame(index=self.aligned_data.index) # Use aligned_data index initially
        for symbol in self.aligned_data.columns:
            # Next period"s return (shifted by -1)
            target_returns[f"{symbol}_next_return"] = self.aligned_data[symbol].pct_change().shift(-1)
            
        # Align features and target to ensure they have the same index
        # Use inner join to keep only timestamps present in both
        features, target_returns = features.align(target_returns, join="inner", axis=0)
        
        # Verify alignment and check for NaNs in target (last row will be NaN)
        if not features.index.equals(target_returns.index):
             self.log("Warning: Feature/target index mismatch after alignment. Re-aligning...")
             features, target_returns = features.align(target_returns, join="inner", axis=0)
             assert features.index.equals(target_returns.index), "Feature/target index mismatch persists!"
             
        # Drop the last row where target is NaN
        if not target_returns.empty and target_returns.iloc[-1].isnull().all():
            self.log("Dropping last row due to NaN target values.")
            features = features.iloc[:-1]
            target_returns = target_returns.iloc[:-1]
        
        # Final check for NaNs in target
        if target_returns.isnull().any().any():
            nan_target_cols = target_returns.columns[target_returns.isnull().any()].tolist()
            self.log(f"Warning: NaNs found in target columns: {nan_target_cols}. Consider investigation.")
            # Option: Drop rows with any NaN target? Or fill?
            # For now, we keep them but log the warning.

        self.features = features
        self.target = target_returns
        
        self.log(f"Feature engineering complete. Shape: {features.shape}")
        self.log(f"Target variable created. Shape: {target_returns.shape}")
        
        return features, target_returns

    def save_processed_data(self, filename=None):
        """
        Save processed data to CSV files and metadata to JSON.
        Includes metadata saving as suggested by DeepSeek analysis.

        Parameters:
        -----------
        filename : str, optional
            Base filename for saving data. If None, use default names.

        Returns:
        --------
        dict
            Dictionary with data types as keys and saved file paths as values
        """
        self.log("Saving processed data...")
        saved_files = {}

        # Define base filename if not provided
        base_filename = filename if filename else "processed_data"

        # Save aligned price data
        if self.aligned_data is not None:
            aligned_file = os.path.join(self.output_dir, f"{base_filename}_aligned.csv")
            self.aligned_data.to_csv(aligned_file)
            saved_files["aligned_prices"] = aligned_file
            self.log(f"Saved aligned price data to {aligned_file}")

        # Save features
        if self.features is not None:
            features_file = os.path.join(self.output_dir, f"{base_filename}_features.csv")
            self.features.to_csv(features_file)
            saved_files["features"] = features_file
            self.log(f"Saved features to {features_file}")
            
        # Save target variables
        if self.target is not None:
            target_file = os.path.join(self.output_dir, f"{base_filename}_targets.csv")
            self.target.to_csv(target_file)
            saved_files["targets"] = target_file
            self.log(f"Saved target variables to {target_file}")

        # Save metadata
        metadata = {
            "preprocessing_date": datetime.now().isoformat(),
            "data_directory": self.data_dir,
            "output_directory": self.output_dir,
            "symbols_processed": self.symbols_processed,
            "sample_size_processed": self.sample_size_processed,
            "window_sizes_used": self.window_sizes_used,
            "outlier_detection_method_used": self.outlier_detection_method_used, # Added
            "outlier_handling_method_used": self.outlier_method_used, # Renamed for clarity
            "outlier_threshold_used": self.outlier_threshold_used,
            "alignment_strategy_used": self.alignment_strategy_used, # Added
            "aligned_data_shape": self.aligned_data.shape if self.aligned_data is not None else None,
            "features_shape": self.features.shape if self.features is not None else None,
            "targets_shape": self.target.shape if self.target is not None else None,
            "feature_columns": self.features.columns.tolist() if self.features is not None else None
        }
        metadata_file = os.path.join(self.output_dir, f"{base_filename}_metadata.json")
        try:
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=4)
            saved_files["metadata"] = metadata_file
            self.log(f"Saved metadata to {metadata_file}")
        except Exception as e:
            self.log(f"Error saving metadata: {str(e)}")

        return saved_files

    def train_test_split(self, test_size=0.2):
        """
        Perform time-based train-test split.
        
        Parameters:
        -----------
        test_size : float
            Proportion of data to use for testing
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        if self.features is None or self.target is None:
            self.log("Error: Features and target must be created before splitting")
            return None, None, None, None
            
        self.log(f"Performing time-based train-test split with test_size={test_size}...")
        
        # Calculate split index
        split_idx = int(len(self.features) * (1 - test_size))
        
        # Split data chronologically
        X_train = self.features.iloc[:split_idx]
        X_test = self.features.iloc[split_idx:]
        
        y_train = self.target.iloc[:split_idx]
        y_test = self.target.iloc[split_idx:]
        
        self.log(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test

    def run_pipeline(self, symbols=None, sample_size=None, window_sizes=[5, 10, 20, 50], 
                    outlier_detection_method="zscore", # Added based on feedback
                    outlier_handling_method="winsorize", # Renamed for clarity
                    outlier_threshold=3.0, 
                    alignment_strategy="union", # Added based on feedback
                    save_filename=None):
        """
        Run the complete preprocessing pipeline.

        Parameters:
        -----------
        symbols : list, optional
            List of specific symbols to process
        sample_size : int, optional
            Number of random symbols to process
        window_sizes : list, optional
            List of window sizes for feature engineering
        outlier_detection_method : str, optional
            Method for outlier detection ("zscore" or "iqr")
        outlier_handling_method : str, optional
            Method for outlier handling ("winsorize" or "clip")
        outlier_threshold : float, optional
            Threshold for outlier detection
        alignment_strategy : str, optional
            Method for time series alignment ("union" or "intersection")
        save_filename : str, optional
            Base filename for saving processed data

        Returns:
        --------
        dict
            Dictionary with processed data types as keys and DataFrames as values
        """
        start_time = time.time()
        self.log("Starting preprocessing pipeline...")

        # Store parameters for metadata saving
        self.symbols_processed = symbols
        self.sample_size_processed = sample_size
        self.window_sizes_used = window_sizes
        self.outlier_detection_method_used = outlier_detection_method # Updated
        self.outlier_method_used = outlier_handling_method # Updated
        self.outlier_threshold_used = outlier_threshold
        self.alignment_strategy_used = alignment_strategy # Added

        # Load data
        self.load_data(symbols=symbols, sample_size=sample_size)

        # Handle missing values
        self.detect_missing_values()
        self.handle_missing_values()

        # Handle outliers (using improved methods)
        self.detect_outliers(method=outlier_detection_method, threshold=outlier_threshold) # Use parameter
        self.handle_outliers(method=outlier_handling_method, threshold=outlier_threshold) # Use parameter

        # Align time series
        self.align_time_series(strategy=alignment_strategy) # Use parameter

        # Engineer features and create target variables
        self.engineer_features(window_sizes=window_sizes)

        # Save processed data
        if save_filename:
            self.save_processed_data(filename=save_filename)

        elapsed_time = time.time() - start_time
        self.log(f"Preprocessing pipeline completed in {elapsed_time:.2f} seconds")

        return {
            "aligned_prices": self.aligned_data,
            "features": self.features,
            "targets": self.target
        }


# Example usage
if __name__ == "__main__":
    # Set data directory (Modified for sandbox)
    data_dir = "C:/Users/chake/Documents/ML Project/towardsai-x-whitebox-startup-challenge/History"
    output_dir = "C:/Users/chake/Documents/ML Project/processed_data"  # Changed output dir
    # Initialize preprocessor
    preprocessor = FinancialDataPreprocessor(data_dir, output_dir)

    # Run pipeline with sample (Using all data for now, remove sample_size later if needed)
    # results = preprocessor.run_pipeline(sample_size=10, save_filename="my_data") # Use sample for faster testing initially
    results = preprocessor.run_pipeline(save_filename="my_data_full_v2") # Run on all data
    
    # Perform time-based train-test split (Optional, not strictly needed for regime detection)
    # X_train, X_test, y_train, y_test = preprocessor.train_test_split(test_size=0.2)
