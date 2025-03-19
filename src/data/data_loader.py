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

# Column mappings to match actual data columns
COLUMN_MAPPING = {
    'game': 'game_id',
    'pts_home': 'home_score', 
    'pts_visitor': 'visitor_score',
    # Player columns are embedded in lineup data rather than a single column
}

class NBADataLoader:
    def __init__(self, data_path, metadata_path):
        """
        Initialize the data loader.
        Args:
            data_path (str): Path to directory containing training/evaluation data
            metadata_path (str): Path to the player statistics metadata file
        """
        self.data_path = data_path
        self.training_dir = os.path.join(data_path, 'training')  # Changed to use data_path
        self.eval_dir = os.path.join(data_path, 'evaluation')
        self.metadata_path = metadata_path
        self._player_candidates = None
        self.player_stats = {}
        self.training_data = None
        self._load_player_stats()

    def _load_player_stats(self):
        """Load player statistics from metadata with validation."""
        try:
            stats_df = pd.read_csv(self.metadata_path, sep='\t')
            
            # Add default stats for missing players
            default_stats = {
                'ppg': 0.0,
                'rpg': 0.0,
                'apg': 0.0,
                # ... other default stats ...
            }
            
            for _, row in stats_df.iterrows():
                self.player_stats[row['player_id']] = row.to_dict()
                
            # Add fallback for unknown players
            self.player_stats['default'] = default_stats
            
        except Exception as e:
            logger.error(f"Failed to load player stats: {str(e)}")
            raise

    def get_player_stats(self, player_id):
        """Get stats for a player with fallback to defaults."""
        stats = self.player_stats.get(player_id, self.player_stats['default'])
        if stats is self.player_stats['default']:
            logger.warning(f"Using default stats for missing player: {player_id}")
        return stats

    def load_training_data(self, years=None):
        """Load training data from annual CSV files."""
        logger.info(f"Loading training data from {self.training_dir}")
        
        # Find all training files
        training_files = glob(os.path.join(self.training_dir, 'matchups-*.csv'))
        
        # Filter by years if specified
        if years:
            year_pattern = r'matchups-(\d{4})\.csv'
            training_files = [
                f for f in training_files
                if int(re.search(year_pattern, f).group(1)) in years
            ]
        
        if not training_files:
            raise FileNotFoundError(f"No training data files found in {self.training_dir}")
        
        # Load and concatenate all data files
        dataframes = []
        for file in training_files:
            logger.info(f"Loading file: {file}")
            year = os.path.basename(file).split('-')[1].split('.')[0]
            df = pd.read_csv(file)
            df['year'] = year
            dataframes.append(df)
        
        self.training_data = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Loaded {len(self.training_data)} training samples")
        
        # Validate the loaded training data
        self.validate_data(self.training_data)
        
        # Preprocess the training data
        self.training_data = self.preprocess_data(self.training_data)
        
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
        
        # Validate the loaded test data
        self.validate_data(self.test_data)
        
        # Preprocess the test data
        self.test_data = self.preprocess_data(self.test_data)
        
        return self.test_data, self.test_labels
    
    def validate_data(self, df):
        required = ['game', 'season', 'home_team', 'away_team', 
                    'pts_home', 'pts_visitor', 'outcome']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def preprocess_data(self, df):
        # Check for different column naming conventions
        if 'home_lineup' in df.columns:
            # Handle comma-separated lineup format
            logger.info("Splitting lineup strings into individual player columns")
            df[['home_player_1', 'home_player_2', 'home_player_3', 'home_player_4', 'home_player_5']] = (
                df['home_lineup'].str.split(',', expand=True)
            )
            df[['away_player_1', 'away_player_2', 'away_player_3', 'away_player_4', 'away_player_5']] = (
                df['away_lineup'].str.split(',', expand=True)
            )
        elif 'home_0' in df.columns:  # Handle numeric index format
            logger.info("Renaming numeric player columns to positional names")
            column_mapping = {
                f'home_{i}': f'home_player_{i+1}' 
                for i in range(5)
            }
            column_mapping.update({
                f'away_{i}': f'away_player_{i+1}' 
                for i in range(5)
            })
            df = df.rename(columns=column_mapping)
        
        # Now create combined players list
        home_players = df[[f'home_player_{i}' for i in range(1,6)]].values.tolist()
        away_players = df[[f'away_player_{i}' for i in range(1,6)]].values.tolist()
        
        df['players'] = [h + a for h, a in zip(home_players, away_players)]
        return df
    
    def preprocess_training_data(self, training_data):
        # Extract player features
        player_features_df = self._extract_player_features(training_data)
        
        # Merge player features with the original dataframe
        training_data = pd.concat([training_data.reset_index(drop=True), player_features_df.reset_index(drop=True)], axis=1)
        
        # Add missing score columns if needed
        if COLUMN_MAPPING['pts_home'] not in training_data.columns and 'pts_home' in training_data.columns:
            training_data = training_data.rename(columns={
                'pts_home': COLUMN_MAPPING['pts_home'],
                'pts_away': COLUMN_MAPPING['pts_visitor']
            })
        
        # Add debug logging
        logging.info(f"Columns after preprocessing: {training_data.columns.tolist()}")
        return training_data
    
    def preprocess_test_data(self, test_data):
        """
        Preprocess the test data for prediction.
        
        Args:
            test_data (pd.DataFrame): The test data to preprocess
        
        Returns:
            pd.DataFrame: Preprocessed test data
        """
        if test_data is None:
            raise ValueError("Test data must be provided for preprocessing")
        
        logger.info("Preprocessing test data...")
        
        # Create a copy to avoid modifying the original
        df = test_data.copy()
        
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
        
        # Validate required columns
        required_columns = [f'home_{i}' for i in range(5)] + [f'away_{i}' for i in range(5)]
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Missing required player position columns after preprocessing")
        
        return df[allowed_columns]
    
    def _extract_player_features(self, df):
        # Create empty container for player features
        player_features = []
        
        # Process each player in the lineup
        for players in df['players']:
            # Ensure we have exactly 10 players (5 from each team)
            if len(players) != 10:
                raise ValueError(f"Invalid number of players in lineup: {len(players)}")
                
            # Split into home and away players
            home_players = players[:5]
            away_players = players[5:]
            
            # Calculate features for each player group
            home_features = self._calculate_group_features(home_players, 'home')
            away_features = self._calculate_group_features(away_players, 'away')
            
            player_features.append({**home_features, **away_features})
            
        return pd.DataFrame(player_features)
    
    def _calculate_group_features(self, players, prefix):
        features = {}
        valid_players = [p for p in players if p != '?']  # Filter out missing players
        
        # Handle case where all players are missing
        if not valid_players:
            return {
                f'{prefix}_avg_height': 0,
                f'{prefix}_avg_weight': 0,
                f'{prefix}_total_points': 0
            }
        
        # Add missing players with default stats
        for p in valid_players:
            if p not in self.player_stats:
                self.player_stats[p] = {
                    'height': 200,  # Default average height in cm
                    'weight': 100,   # Default average weight in kg
                    'points': 10     # Default average points
                }
                logger.warning(f"Using default stats for missing player: {p}")

        # Calculate features with fallback values
        try:
            features[f'{prefix}_avg_height'] = np.mean([self.player_stats[p].get('height', 200) for p in valid_players])
            features[f'{prefix}_avg_weight'] = np.mean([self.player_stats[p].get('weight', 100) for p in valid_players])
            features[f'{prefix}_total_points'] = np.sum([self.player_stats[p].get('points', 10) for p in valid_players])
        except KeyError as e:
            logger.error(f"Missing stats for player {e} even after fallback")
            raise
        
        return features
    
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
        
        # Get all players from the 'players' column
        all_players = set()
        for players in df['players']:
            all_players.update(players)
        
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