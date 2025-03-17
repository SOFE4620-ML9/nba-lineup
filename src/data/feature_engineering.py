"""
Feature engineering module for NBA lineup prediction project.
This module handles the transformation of raw data into features for model training.
"""

from collections import defaultdict  # Add this import at the top
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import logging
import yaml  # Add import for yaml

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
    
    def __init__(self, config_path='config/feature_config.yaml'):
        self.config = self._load_config(config_path)
        self.scaler = StandardScaler() if self.config.get('scale_features') else None
        self.logger = logging.getLogger('nba_feature_engineering')
        self.logger.addHandler(logging.NullHandler())  # Avoid duplicate logs
        self._debug_counter = 0  # Track processed players
        self.player_encoder = None
        self.team_encoder = None
        self.player_stats_cache = {}
        self.team_stats_cache = {}
        self.position_mapping = {
            'PG': 0,  # Point Guard
            'SG': 1,  # Shooting Guard
            'SF': 2,  # Small Forward
            'PF': 3,  # Power Forward
            'C': 4    # Center
        }
        
    def _load_config(self, config_path):
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults")
            config = {
                'scale_features': True,
                'target_player': 'home_player_5',
                'numeric_features': [
                    'pts_home', 'pts_visitor',
                    'reb_home', 'reb_visitor',
                    'ast_home', 'ast_visitor'
                ],
                'categorical_features': [
                    'home_team', 'away_team'
                ]
            }
        return config

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
        
    def transform(self, df):
        """
        Transform the input data into features for model training.
        
        Args:
            df (pd.DataFrame): Input data to transform
            
        Returns:
            pd.DataFrame: Transformed features
        """
        logger.info("Transforming input data...")
        
        # Store the dataframe for player statistics calculation
        self.df = df
        
        # Initialize feature arrays
        features = []
        
        # Process each game
        for _, row in df.iterrows():
            # Get home team lineup (first 4 players)
            home_lineup = []
            for i in range(4):  # We only want the first 4 players
                col = f'home_{i}'
                if col in row and row[col] != '?' and not pd.isna(row[col]):
                    home_lineup.append(row[col])
            
            # Skip if we don't have enough players
            if len(home_lineup) < 4:
                logger.warning(f"Skipping game due to insufficient home lineup: {row.get('game_id', 'unknown')}")
                continue
            
            # Get team
            home_team = row.get('home_team')
            if not home_team:
                logger.warning(f"Skipping game due to missing home team: {row.get('game_id', 'unknown')}")
                continue
            
            # Extract lineup features
            game_features = self.extract_lineup_features(home_lineup, home_team)
            
            # Add game context features if available
            if 'game_date' in row:
                game_features['days_rest'] = row.get('days_rest_home', 0)
                game_features['is_back_to_back'] = 1 if row.get('days_rest_home', 0) == 0 else 0
                game_features['is_home_game'] = 1  # Always 1 for home team
            
            # Add team performance features if available
            if 'home_team_win_pct' in row:
                game_features['team_win_pct'] = row['home_team_win_pct']
            
            # Add opponent features if available
            if 'away_team' in row:
                away_team = row['away_team']
                game_features['opponent_win_pct'] = row.get('away_team_win_pct', 0.5)
                
                # Get opponent lineup
                away_lineup = []
                for i in range(5):
                    col = f'away_{i}'
                    if col in row and row[col] != '?' and not pd.isna(row[col]):
                        away_lineup.append(row[col])
                
                if away_lineup:
                    # Calculate opponent lineup strength
                    opp_features = self.extract_lineup_features(away_lineup, away_team)
                    for key, value in opp_features.items():
                        game_features[f'opp_{key}'] = value
            
            features.append(game_features)
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(features)
        
        # Fill missing values
        feature_df = feature_df.fillna(0)
        
        # Scale numerical features
        numerical_cols = feature_df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_cols) > 0:
            feature_df[numerical_cols] = self.scaler.fit_transform(feature_df[numerical_cols])
        
        logger.info(f"Transformed data shape: {feature_df.shape}")
        return feature_df
    
    def transform_training_data(self, df):
        """
        Transform training data and extract labels.
        
        Args:
            df (pd.DataFrame): Training data
            
        Returns:
            tuple: (X, y) where X is features and y is labels
        """
        logger.info("Transforming training data...")
        
        # Extract features
        X = self.transform(df)
        
        # Extract labels (actual fifth player)
        y = []
        for _, row in df.iterrows():
            fifth_player = row.get('home_player_5', '?')  # Changed from 'home_4'
            if fifth_player != '?' and not pd.isna(fifth_player):
                y.append(fifth_player)
        
        # Remove rows where label is missing
        valid_mask = [y_i is not None for y_i in y]
        X = X[valid_mask]
        y = [y_i for i, y_i in enumerate(y) if valid_mask[i]]
        
        logger.info(f"Transformed training data shape: X={X.shape}, y={len(y)}")
        return X, y
    
    def transform_test_data(self, df):
        """
        Transform test data.
        
        Args:
            df (pd.DataFrame): Test data
            
        Returns:
            pd.DataFrame: Transformed features
        """
        logger.info("Transforming test data...")
        
        # Extract features
        X = self.transform(df)
        
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
        features = []
        for players in df['players']:
            # Filter out placeholder '?' values
            valid_players = [p for p in players if p != '?']
            
            # Calculate features for valid players
            player_stats = [self.player_stats.get(p, {}) for p in valid_players]
            # ... rest of feature calculation logic ...
        
        return pd.DataFrame(features)
    
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
    
    def calculate_player_statistics(self, player, team, df):
        try:
            self._debug_counter += 1
            if self._debug_counter % 100 == 0:
                self.logger.debug(f"Processed {self._debug_counter} players...")
            
            # Optimized player presence check using vectorized operations
            home_players = df[[f'home_player_{i}' for i in range(1,6)]]
            away_players = df[[f'away_player_{i}' for i in range(1,6)]]
            
            # Create boolean masks using numpy for faster computation
            home_mask = (df['home_team'] == team) & (home_players == player).any(axis=1)
            away_mask = (df['away_team'] == team) & (away_players == player).any(axis=1)
            
            player_games = df[home_mask | away_mask].copy()
            
            # Convert categorical columns to reduce memory usage
            categorical_cols = ['home_team', 'away_team'] + \
                              [f'{loc}_player_{i}' for loc in ['home', 'away'] for i in range(1,6)]
            for col in categorical_cols:
                if col in player_games.columns:
                    player_games[col] = player_games[col].astype('category')

            # Calculate basic stats using correct column names
            stats = defaultdict(float)
            if not player_games.empty:
                # Use vectorized operations instead of slow apply/lambda
                home_mask = (player_games['home_team'] == team)
                away_mask = (player_games['away_team'] == team)
                
                # Points calculation
                stats['total_points'] = player_games.loc[home_mask, 'home_score'].sum() + \
                                      player_games.loc[away_mask, 'pts_visitor'].sum()
                
                # Other stats using correct column names
                stats['assists'] = player_games.loc[home_mask, 'ast_home'].sum() + \
                                 player_games.loc[away_mask, 'ast_visitor'].sum()
                                 
                stats['rebounds'] = player_games.loc[home_mask, 'reb_home'].sum() + \
                                  player_games.loc[away_mask, 'reb_visitor'].sum()
                
                stats['blocks'] = player_games.loc[home_mask, 'blk_home'].sum() + \
                                player_games.loc[away_mask, 'blk_visitor'].sum()
                
                # Calculate percentages using safe division
                total_games = len(player_games)
                stats['win_rate'] = (player_games['outcome'] == 1).sum() / total_games if total_games > 0 else 0
                
                # Convert defaultdict to regular dict to avoid serialization issues
                return dict(stats)
            
        except Exception as e:
            logger.error(f"Error calculating stats for {player}: {str(e)}")
            return defaultdict(float)  # Return default values

    def calculate_team_compatibility(self, lineup, team):
        """
        Calculate compatibility score for a given lineup with a team.
        
        Args:
            lineup (list): List of player names
            team (str): Team name
            
        Returns:
            float: Compatibility score between 0 and 1
        """
        if not lineup:
            return 0.0
        
        # Calculate position balance
        positions = []
        for player in lineup:
            stats = self.player_stats.get(player, {})
            positions.append(stats.get('primary_position', 'Unknown'))
        
        unique_positions = len(set(positions))
        position_balance = unique_positions / 5  # Normalize by max possible positions
        
        # Calculate team chemistry
        chemistry_scores = []
        for player in lineup:
            stats = self.player_stats.get(player, {})
            chemistry_scores.append(stats.get('team_chemistry', 0.0))
        team_chemistry = np.mean(chemistry_scores)
        
        # Calculate experience balance
        experience_scores = []
        for player in lineup:
            stats = self.player_stats.get(player, {})
            experience_scores.append(stats.get('total_games', 0))
        experience_std = np.std(experience_scores)
        experience_balance = 1 / (1 + experience_std)  # Lower std = higher balance
        
        # Calculate skill complementarity
        scoring_scores = []
        playmaking_scores = []
        defense_scores = []
        rebounding_scores = []
        
        for player in lineup:
            stats = self.player_stats.get(player, {})
            scoring_scores.append(stats.get('points_per_game', 0.0))
            playmaking_scores.append(stats.get('assists_per_game', 0.0))
            defense_scores.append((stats.get('steals_per_game', 0.0) + stats.get('blocks_per_game', 0.0)) / 2)
            rebounding_scores.append(stats.get('rebounds_per_game', 0.0))
        
        # Calculate skill diversity
        skill_diversity = (
            np.std(scoring_scores) +
            np.std(playmaking_scores) +
            np.std(defense_scores) +
            np.std(rebounding_scores)
        ) / 4
        
        # Combine all factors with weights
        weights = {
            'position_balance': 0.3,
            'team_chemistry': 0.2,
            'experience_balance': 0.2,
            'skill_diversity': 0.3
        }
        
        compatibility_score = (
            weights['position_balance'] * position_balance +
            weights['team_chemistry'] * team_chemistry +
            weights['experience_balance'] * experience_balance +
            weights['skill_diversity'] * (1 - skill_diversity)  # Lower diversity = higher complementarity
        )
        
        return compatibility_score

    def extract_lineup_features(self, lineup, team):
        """
        Extract features for a given lineup and team.
        
        Args:
            lineup (list): List of player names
            team (str): Team name
            
        Returns:
            dict: Lineup features
        """
        total_points = 0.0
        total_rebounds = 0.0
        total_assists = 0.0
        
        for player in lineup:
            stats = self.player_stats.get(player, {})
            # Use get() with default values
            total_points += stats.get('points_per_game', 0.0)
            total_rebounds += stats.get('rebounds_per_game', 0.0) 
            total_assists += stats.get('assists_per_game', 0.0)
        
        # Add fallback for missing team stats
        team_stats = self.team_stats.get(team, {
            'avg_points': 0.0,
            'avg_rebounds': 0.0,
            'win_rate': 0.0
        })
        
        return {
            'total_lineup_points': total_points,
            'total_lineup_rebounds': total_rebounds,
            'team_win_rate': team_stats['win_rate'],
            # ... other features ...
        }

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