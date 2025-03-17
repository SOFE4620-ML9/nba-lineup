"""
Data loader module for NBA lineup prediction project.
This module handles the loading and preprocessing of NBA game data.
"""

import os
import pandas as pd
import numpy as np
from glob import glob
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('nba_data_loader')

class NBADataLoader:
    """
    Class responsible for loading and preprocessing NBA game data.
    
    Attributes:
        data_dir (str): Directory containing the dataset
        allowed_features (list): List of features allowed for model training
    """
    
    def __init__(self, data_dir='dataset', allowed_features=None):
        """
        Initialize the data loader.
        
        Args:
            data_dir (str): Path to the data directory
            allowed_features (list): List of allowed features for the model
        """
        self.data_dir = data_dir
        self.training_dir = os.path.join(data_dir, 'training')
        self.eval_dir = os.path.join(data_dir, 'eval')
        self.allowed_features = allowed_features
        self.training_data = None
        self.test_data = None
        self.test_labels = None
        
    def load_training_data(self, years=None):
        """
        Load training data for specified years.
        
        Args:
            years (list): List of years to load data for. If None, loads all years.
        
        Returns:
            pd.DataFrame: Combined training data
        """
        logger.info("Loading training data...")
        
        if years is None:
            # Load all years
            file_pattern = os.path.join(self.training_dir, 'matchups-*.csv')
            files = sorted(glob(file_pattern))
        else:
            # Load specific years
            files = [os.path.join(self.training_dir, f'matchups-{year}.csv') for year in years]
        
        if not files:
            raise FileNotFoundError(f"No training data files found in {self.training_dir}")
        
        # Load and concatenate all data files
        dataframes = []
        for file in files:
            logger.info(f"Loading file: {file}")
            year = os.path.basename(file).split('-')[1].split('.')[0]
            df = pd.read_csv(file)
            df['year'] = year
            dataframes.append(df)
        
        self.training_data = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Loaded {len(self.training_data)} training samples")
        
        return self.training_data
    
    def load_test_data(self):
        """
        Load test data and labels.
        
        Returns:
            tuple: (test_data, test_labels)
        """
        logger.info("Loading test data...")
        
        test_file = os.path.join(self.eval_dir, 'NBA_test.csv')
        label_file = os.path.join(self.eval_dir, 'NBA_test_labels.csv')
        
        if not os.path.exists(test_file) or not os.path.exists(label_file):
            raise FileNotFoundError(f"Test data files not found in {self.eval_dir}")
        
        self.test_data = pd.read_csv(test_file)
        self.test_labels = pd.read_csv(label_file)
        
        logger.info(f"Loaded {len(self.test_data)} test samples")
        
        return self.test_data, self.test_labels
    
    def preprocess_training_data(self):
        """
        Preprocess the training data for model training.
        
        Returns:
            pd.DataFrame: Preprocessed training data
        """
        if self.training_data is None:
            raise ValueError("Training data must be loaded before preprocessing")
        
        logger.info("Preprocessing training data...")
        
        # Create a copy to avoid modifying the original
        df = self.training_data.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Filter by allowed features if specified
        if self.allowed_features:
            df = self._filter_allowed_features(df)
        
        logger.info("Training data preprocessing complete")
        
        return df
    
    def preprocess_test_data(self):
        """
        Preprocess the test data for prediction.
        
        Returns:
            pd.DataFrame: Preprocessed test data
        """
        if self.test_data is None:
            raise ValueError("Test data must be loaded before preprocessing")
        
        logger.info("Preprocessing test data...")
        
        # Create a copy to avoid modifying the original
        df = self.test_data.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Filter by allowed features if specified
        if self.allowed_features:
            df = self._filter_allowed_features(df)
        
        logger.info("Test data preprocessing complete")
        
        return df
    
    def _handle_missing_values(self, df):
        """
        Handle missing values in the data.
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        # Check for missing values
        missing_values = df.isnull().sum()
        logger.info(f"Missing values before handling: {missing_values[missing_values > 0]}")
        
        # For numeric columns, fill missing values with median
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # For categorical columns, fill missing values with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                # Skip player columns with '?' as these indicate the missing player we need to predict
                if col in ['home_0', 'home_1', 'home_2', 'home_3', 'home_4']:
                    continue
                df[col] = df[col].fillna(df[col].mode()[0])
        
        # Check remaining missing values
        missing_values = df.isnull().sum()
        logger.info(f"Missing values after handling: {missing_values[missing_values > 0]}")
        
        return df
    
    def _filter_allowed_features(self, df):
        """
        Filter dataframe to include only allowed features.
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        # Get all columns in the dataframe
        all_columns = set(df.columns)
        
        # Add player columns as they are always needed
        player_cols = [col for col in all_columns if col.startswith('home_') or col.startswith('away_')]
        allowed_features_set = set(self.allowed_features) | set(player_cols)
        
        # Keep only allowed features
        allowed_columns = list(all_columns.intersection(allowed_features_set))
        
        logger.info(f"Keeping {len(allowed_columns)} features out of {len(all_columns)} total columns")
        
        return df[allowed_columns]
    
    def get_player_candidates(self, season=None):
        """
        Get a list of all player candidates for a specific season or all seasons.
        
        Args:
            season (str, optional): Season to get players for. If None, gets players from all seasons.
        
        Returns:
            list: List of unique player names
        """
        if self.training_data is None:
            raise ValueError("Training data must be loaded first")
        
        df = self.training_data
        
        # Filter by season if specified
        if season:
            df = df[df['season'] == season]
        
        # Get all home team players
        home_players = []
        for i in range(5):
            col = f'home_{i}'
            if col in df.columns:
                home_players.extend(df[col].dropna().unique())
        
        # Get all away team players
        away_players = []
        for i in range(5):
            col = f'away_{i}'
            if col in df.columns:
                away_players.extend(df[col].dropna().unique())
        
        # Combine and get unique players
        all_players = list(set(home_players) | set(away_players))
        all_players = [p for p in all_players if p != '?']
        
        logger.info(f"Found {len(all_players)} unique players")
        
        return sorted(all_players)

if __name__ == "__main__":
    # Example usage
    loader = NBADataLoader()
    
    # Load training data
    training_data = loader.load_training_data()
    print(f"Training data shape: {training_data.shape}")
    
    # Load test data
    test_data, test_labels = loader.load_test_data()
    print(f"Test data shape: {test_data.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    
    # Get player candidates
    players = loader.get_player_candidates()
    print(f"Number of unique players: {len(players)}") 