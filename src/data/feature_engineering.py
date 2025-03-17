"""
Feature engineering module for NBA lineup prediction project.
This module handles the transformation of raw data into features for model training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('nba_feature_engineering')

class NBAFeatureEngineer:
    """
    Class responsible for engineering features for the NBA lineup prediction model.
    
    Attributes:
        player_encoder (OneHotEncoder): Encoder for player names
        team_encoder (OneHotEncoder): Encoder for team names
        scaler (StandardScaler): Scaler for numerical features
    """
    
    def __init__(self):
        """Initialize the feature engineer with encoders and transformers."""
        self.player_encoder = None
        self.team_encoder = None
        self.scaler = StandardScaler()
        self.player_stats_cache = {}
        self.team_stats_cache = {}
        
    def fit(self, df):
        """
        Fit the feature engineering transformations on training data.
        
        Args:
            df (pd.DataFrame): Training data to fit transformers on
        """
        logger.info("Fitting feature engineering transformations...")
        
        # Extract all player names
        player_cols = [col for col in df.columns if col.startswith('home_') or col.startswith('away_')]
        all_players = []
        for col in player_cols:
            all_players.extend(df[col].dropna().unique())
        all_players = list(set(all_players) - {'?'})
        
        # Fit one-hot encoder for players
        self.player_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.player_encoder.fit(np.array(all_players).reshape(-1, 1))
        
        # Fit one-hot encoder for teams
        all_teams = list(set(df['home_team'].unique()) | set(df['away_team'].unique()))
        self.team_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.team_encoder.fit(np.array(all_teams).reshape(-1, 1))
        
        # Fit scaler for numerical features
        numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
        numerical_features = [col for col in numerical_features if not col.startswith('outcome')]
        self.scaler.fit(df[numerical_features].fillna(0))
        
        logger.info("Feature engineering transformations fitted successfully")
        
    def transform_training_data(self, df):
        """
        Transform training data into features for model training.
        
        Args:
            df (pd.DataFrame): Training data to transform
            
        Returns:
            tuple: (X, y) where X is the feature matrix and y is the target vector
        """
        logger.info("Transforming training data...")
        
        # Extract features
        X = self._extract_features(df)
        
        # Extract target (outcome)
        y = df['outcome'].values if 'outcome' in df.columns else None
        
        logger.info(f"Transformed training data shape: {X.shape}")
        
        return X, y
    
    def transform_test_data(self, df):
        """
        Transform test data into features for prediction.
        
        Args:
            df (pd.DataFrame): Test data to transform
            
        Returns:
            pd.DataFrame: Transformed features
        """
        logger.info("Transforming test data...")
        
        # Extract features
        X = self._extract_features(df)
        
        logger.info(f"Transformed test data shape: {X.shape}")
        
        return X
    
    def _extract_features(self, df):
        """
        Extract features from raw data.
        
        Args:
            df (pd.DataFrame): Raw data
            
        Returns:
            pd.DataFrame: Extracted features
        """
        features = []
        
        # Team features
        team_features = self._extract_team_features(df)
        features.append(team_features)
        
        # Player features
        player_features = self._extract_player_features(df)
        features.append(player_features)
        
        # Game context features
        game_features = self._extract_game_features(df)
        features.append(game_features)
        
        # Concatenate all features
        X = pd.concat(features, axis=1)
        
        return X
    
    def _extract_team_features(self, df):
        """
        Extract team-related features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Team features
        """
        # Encode home and away teams
        home_team_encoded = self.team_encoder.transform(df['home_team'].values.reshape(-1, 1))
        away_team_encoded = self.team_encoder.transform(df['away_team'].values.reshape(-1, 1))
        
        # Create feature dataframe
        home_team_cols = [f'home_team_{i}' for i in range(home_team_encoded.shape[1])]
        away_team_cols = [f'away_team_{i}' for i in range(away_team_encoded.shape[1])]
        
        team_features = pd.DataFrame(
            np.hstack([home_team_encoded, away_team_encoded]),
            columns=home_team_cols + away_team_cols,
            index=df.index
        )
        
        return team_features
    
    def _extract_player_features(self, df):
        """
        Extract player-related features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Player features
        """
        # Get player columns (excluding the missing player in test data)
        home_player_cols = [col for col in df.columns if col.startswith('home_') and col != 'home_team']
        away_player_cols = [col for col in df.columns if col.startswith('away_') and col != 'away_team']
        
        # Create empty feature matrix
        all_player_features = []
        
        # Process home team players
        for col in home_player_cols:
            # Skip missing players (marked with '?')
            players = df[col].replace('?', np.nan).dropna()
            if players.empty:
                continue
                
            # Encode players
            players_encoded = self._encode_players(players)
            players_encoded.columns = [f'{col}_{i}' for i in range(players_encoded.shape[1])]
            all_player_features.append(players_encoded)
        
        # Process away team players
        for col in away_player_cols:
            players = df[col].replace('?', np.nan).dropna()
            if players.empty:
                continue
                
            # Encode players
            players_encoded = self._encode_players(players)
            players_encoded.columns = [f'{col}_{i}' for i in range(players_encoded.shape[1])]
            all_player_features.append(players_encoded)
        
        # Concatenate all player features
        if all_player_features:
            player_features = pd.concat(all_player_features, axis=1)
            player_features.index = df.index
        else:
            # Create empty dataframe if no player features
            player_features = pd.DataFrame(index=df.index)
        
        return player_features
    
    def _encode_players(self, players):
        """
        Encode player names using one-hot encoding.
        
        Args:
            players (pd.Series): Series of player names
            
        Returns:
            pd.DataFrame: Encoded player features
        """
        if self.player_encoder is None:
            raise ValueError("Player encoder must be fitted before encoding players")
        
        # Reshape to 2D array for encoder
        players_2d = players.values.reshape(-1, 1)
        
        # Encode players
        encoded = self.player_encoder.transform(players_2d)
        
        return pd.DataFrame(encoded, index=players.index)
    
    def _extract_game_features(self, df):
        """
        Extract game context features.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Game context features
        """
        # Create empty dataframe for game features
        game_features = pd.DataFrame(index=df.index)
        
        # Add basic features that should be available in both training and test data
        if 'season' in df.columns:
            game_features['season'] = df['season'].astype(int)
            
        if 'starting_min' in df.columns:
            game_features['starting_min'] = df['starting_min'].fillna(0).astype(float)
            
        # For test data, we may not have all the game statistics that were available during training
        # Check if we're dealing with test data (which has fewer columns)
        is_test_data = len(df.columns) < 30  # Training data has many more columns
        
        if is_test_data:
            logger.info("Processing test data with limited features")
            # For test data, we'll use only the lineup information since game stats aren't available
            return game_features
        
        # If we're here, we're processing training data with all features
        # Select numerical features
        numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
        numerical_features = [col for col in numerical_features 
                             if not col.startswith('outcome') and 
                             not col.startswith('home_') and 
                             not col.startswith('away_')]
        
        # Scale numerical features if we have the scaler fitted
        if len(numerical_features) > 0 and self.scaler is not None:
            # Check if scaler has been fitted
            if hasattr(self.scaler, 'mean_'):
                try:
                    features_scaled = self.scaler.transform(df[numerical_features].fillna(0))
                    game_features = pd.concat([
                        game_features,
                        pd.DataFrame(
                            features_scaled,
                            columns=numerical_features,
                            index=df.index
                        )
                    ], axis=1)
                except ValueError as e:
                    logger.warning(f"Could not scale features: {e}")
        
        return game_features
    
    def calculate_player_statistics(self, df):
        """
        Calculate statistics for each player based on historical performance.
        
        Args:
            df (pd.DataFrame): Training data with player performance
            
        Returns:
            dict: Dictionary mapping player names to their statistics
        """
        logger.info("Calculating player statistics...")
        
        # Initialize player stats dictionary
        player_stats = {}
        
        # Get player columns
        player_cols = [col for col in df.columns if col.startswith('home_') or col.startswith('away_')]
        player_cols = [col for col in player_cols if col != 'home_team' and col != 'away_team']
        
        # Get all unique players
        all_players = set()
        for col in player_cols:
            all_players.update(df[col].dropna().unique())
        all_players.discard('?')
        
        # Initialize stats for each player
        for player in all_players:
            player_stats[player] = {
                'total_games': 0,
                'wins': 0,
                'points_contribution': 0,
                'rebounds_contribution': 0,
                'assists_contribution': 0,
            }
        
        # Calculate statistics for each player
        for _, row in df.iterrows():
            # Process home team players
            home_outcome = 1 if row.get('outcome', 0) == 1 else 0
            home_points = row.get('pts_home', 0)
            home_rebounds = row.get('reb_home', 0)
            home_assists = row.get('ast_home', 0)
            
            home_players = []
            for i in range(5):
                col = f'home_{i}'
                if col in row and row[col] != '?' and not pd.isna(row[col]):
                    home_players.append(row[col])
            
            # Update home player stats
            if home_players:
                for player in home_players:
                    if player in player_stats:
                        player_stats[player]['total_games'] += 1
                        player_stats[player]['wins'] += home_outcome
                        player_stats[player]['points_contribution'] += home_points / len(home_players)
                        player_stats[player]['rebounds_contribution'] += home_rebounds / len(home_players)
                        player_stats[player]['assists_contribution'] += home_assists / len(home_players)
            
            # Process away team players
            away_outcome = 1 if row.get('outcome', 0) == 0 else 0  # Away team wins when outcome is 0
            away_points = row.get('pts_visitor', 0)
            away_rebounds = row.get('reb_visitor', 0)
            away_assists = row.get('ast_visitor', 0)
            
            away_players = []
            for i in range(5):
                col = f'away_{i}'
                if col in row and row[col] != '?' and not pd.isna(row[col]):
                    away_players.append(row[col])
            
            # Update away player stats
            if away_players:
                for player in away_players:
                    if player in player_stats:
                        player_stats[player]['total_games'] += 1
                        player_stats[player]['wins'] += away_outcome
                        player_stats[player]['points_contribution'] += away_points / len(away_players)
                        player_stats[player]['rebounds_contribution'] += away_rebounds / len(away_players)
                        player_stats[player]['assists_contribution'] += away_assists / len(away_players)
        
        # Calculate averages
        for player in player_stats:
            games = player_stats[player]['total_games']
            if games > 0:
                player_stats[player]['win_rate'] = player_stats[player]['wins'] / games
                player_stats[player]['avg_points'] = player_stats[player]['points_contribution'] / games
                player_stats[player]['avg_rebounds'] = player_stats[player]['rebounds_contribution'] / games
                player_stats[player]['avg_assists'] = player_stats[player]['assists_contribution'] / games
            else:
                player_stats[player]['win_rate'] = 0
                player_stats[player]['avg_points'] = 0
                player_stats[player]['avg_rebounds'] = 0
                player_stats[player]['avg_assists'] = 0
        
        # Cache player statistics
        self.player_stats_cache = player_stats
        
        logger.info(f"Calculated statistics for {len(player_stats)} players")
        
        return player_stats
    
    def get_player_statistics(self, player_name):
        """
        Get statistics for a specific player.
        
        Args:
            player_name (str): Name of the player
            
        Returns:
            dict: Dictionary of player statistics
        """
        if not self.player_stats_cache:
            raise ValueError("Player statistics must be calculated before retrieval")
        
        return self.player_stats_cache.get(player_name, {
            'total_games': 0,
            'wins': 0,
            'win_rate': 0,
            'avg_points': 0,
            'avg_rebounds': 0,
            'avg_assists': 0,
        })

if __name__ == "__main__":
    # Example usage with dummy data
    import pandas as pd
    
    # Create dummy data
    data = {
        'home_team': ['LAL', 'GSW', 'BOS'],
        'away_team': ['MIA', 'CHI', 'NYK'],
        'home_0': ['LeBron James', 'Stephen Curry', 'Jayson Tatum'],
        'home_1': ['Anthony Davis', 'Klay Thompson', 'Jaylen Brown'],
        'home_2': ['Russell Westbrook', 'Draymond Green', 'Marcus Smart'],
        'home_3': ['Carmelo Anthony', 'Andrew Wiggins', 'Al Horford'],
        'home_4': ['Dwight Howard', 'Kevin Looney', 'Robert Williams'],
        'away_0': ['Jimmy Butler', 'Zach LaVine', 'RJ Barrett'],
        'away_1': ['Bam Adebayo', 'DeMar DeRozan', 'Julius Randle'],
        'away_2': ['Kyle Lowry', 'Lonzo Ball', 'Evan Fournier'],
        'away_3': ['Tyler Herro', 'Nikola Vucevic', 'Mitchell Robinson'],
        'away_4': ['PJ Tucker', 'Patrick Williams', 'Alec Burks'],
        'season': [2022, 2022, 2022],
        'pts_home': [110, 120, 105],
        'pts_visitor': [95, 118, 110],
        'reb_home': [45, 40, 38],
        'reb_visitor': [40, 35, 42],
        'ast_home': [25, 30, 22],
        'ast_visitor': [20, 25, 18],
        'outcome': [1, 1, 0]  # 1 = home team wins, 0 = away team wins
    }
    
    df = pd.DataFrame(data)
    
    # Create feature engineer
    engineer = NBAFeatureEngineer()
    
    # Fit transformations
    engineer.fit(df)
    
    # Transform data
    X, y = engineer.transform_training_data(df)
    
    # Calculate player statistics
    player_stats = engineer.calculate_player_statistics(df)
    
    # Print sample results
    print("Feature matrix shape:", X.shape)
    print("Target vector shape:", y.shape)
    print("\nPlayer statistics for LeBron James:")
    print(engineer.get_player_statistics('LeBron James')) 