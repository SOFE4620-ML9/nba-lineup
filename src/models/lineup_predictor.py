"""
Lineup predictor module for NBA lineup prediction project.
This module contains the model for predicting the optimal fifth player for a home team.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
import logging
import joblib
import os
import random
from data.data_loader import load_preprocessed_data  # Instead of 'from src.data...'

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('nba_lineup_predictor')

class LineupPredictor:
    """
    Model for predicting the optimal fifth player for a home team in NBA games.
    
    Attributes:
        model (BaseEstimator): Scikit-learn model for prediction
        player_encoder: Encoder for player names
        feature_engineer: Feature engineering transformer
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the lineup predictor with a specified model.
        
        Args:
            model_type (str): Type of model to use ('random_forest', 'gradient_boosting', or 'logistic')
        """
        self.model_type = model_type
        self.model = None
        self.feature_engineer = None
        self.player_candidates = None
        self.player_stats = None
        
    def _initialize_model(self):
        """Initialize the underlying prediction model based on the specified type."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,  # Increased from 100
                max_depth=10,      # Added max_depth
                min_samples_split=5,  # Increased from 2
                min_samples_leaf=2,   # Increased from 1
                n_jobs=-1,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=200,  # Increased from 100
                learning_rate=0.05,  # Decreased from 0.1
                max_depth=5,      # Increased from 3
                subsample=0.8,    # Added subsampling
                random_state=42
            )
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(
                C=0.1,            # Decreased from 1.0
                max_iter=2000,    # Increased from 1000
                solver='saga',    # Changed from liblinear
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def set_feature_engineer(self, feature_engineer):
        """
        Set the feature engineer for the predictor.
        
        Args:
            feature_engineer: Feature engineering transformer
        """
        self.feature_engineer = feature_engineer
    
    def set_player_candidates(self, player_candidates):
        """
        Set the list of player candidates for prediction.
        
        Args:
            player_candidates (list): List of player names
        """
        self.player_candidates = player_candidates
    
    def set_player_stats(self, player_stats):
        """
        Set player statistics for prediction.
        
        Args:
            player_stats (dict): Dictionary of player statistics
        """
        self.player_stats = player_stats
    
    def train(self, X, y):
        """
        Train the model on the provided data.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (np.ndarray): Target vector
            
        Returns:
            self: Trained model
        """
        logger.info(f"Training {self.model_type} model...")
        
        # Initialize the model if not already done
        if self.model is None:
            self._initialize_model()
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_preds = self.model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_preds)
        
        logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        logger.info(f"Classification report:\n{classification_report(y_val, val_preds)}")
        
        # Retrain on all data
        self.model.fit(X, y)
        
        logger.info("Model training complete")
        
        return self
    
    def predict_optimal_player(self, lineup_data):
        """
        Predict the optimal fifth player for the given lineup.
        
        Args:
            lineup_data (pd.DataFrame): Data for the lineup
            
        Returns:
            list: List of predicted player names
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        if self.feature_engineer is None:
            raise ValueError("Feature engineer must be set before prediction")
        
        if self.player_candidates is None:
            raise ValueError("Player candidates must be set before prediction")
        
        logger.info("Predicting optimal fifth player...")
        
        # Copy the data to avoid modifying the original
        data = lineup_data.copy()
        
        # Get the indices of rows with missing home player (marked with '?')
        missing_player_mask = data.apply(
            lambda row: any(row[f'home_{i}'] == '?' for i in range(5) if f'home_{i}' in row),
            axis=1
        )
        missing_player_indices = data[missing_player_mask].index
        
        if len(missing_player_indices) == 0:
            logger.warning("No missing home players found in the data")
            return []
        
        # Get the optimal player for each row with a missing player
        optimal_players = []
        
        for idx in missing_player_indices:
            optimal_player = self._find_optimal_player(data.loc[idx:idx])
            optimal_players.append(optimal_player)
        
        logger.info(f"Predicted {len(optimal_players)} optimal players")
        
        return optimal_players
    
    def _find_optimal_player(self, lineup_row):
        """
        Find the optimal fifth player for a single lineup.
        
        Args:
            lineup_row (pd.DataFrame): Single row dataframe with lineup data
            
        Returns:
            str: Name of the optimal player
        """
        # Find which home player position is missing
        missing_position = None
        for i in range(5):
            col = f'home_{i}'
            if col in lineup_row and lineup_row[col].iloc[0] == '?':
                missing_position = i
                logger.info(f"Found missing player at position home_{i}")
                break
        
        if missing_position is None:
            logger.warning("No missing home player found in the lineup")
            return None
        
        # Get existing home players
        existing_home_players = []
        for i in range(5):
            col = f'home_{i}'
            if i != missing_position and col in lineup_row:
                player = lineup_row[col].iloc[0]
                if player != '?' and not pd.isna(player):
                    existing_home_players.append(player)
        
        logger.info(f"Existing home players: {existing_home_players}")
        
        # Get away team players
        away_players = []
        for i in range(5):
            col = f'away_{i}'
            if col in lineup_row:
                player = lineup_row[col].iloc[0]
                if player != '?' and not pd.isna(player):
                    away_players.append(player)
        
        logger.info(f"Away players: {away_players}")
        logger.info(f"Total candidate players: {len(self.player_candidates)}")
        
        # Get home team
        home_team = lineup_row['home_team'].iloc[0]
        
        # Score each candidate player
        player_scores = {}
        candidates_evaluated = 0
        
        for candidate in self.player_candidates:
            # Skip players already in the lineup
            if candidate in existing_home_players or candidate in away_players:
                continue
            
            # Get player statistics
            player_stats = self.feature_engineer.calculate_player_statistics(candidate, home_team, self.feature_engineer.df)
            
            # Calculate player score based on multiple factors
            score = 0.0
            
            # 1. Performance score (25%)
            performance_score = (
                player_stats['points_per_game'] * 0.4 +
                player_stats['assists_per_game'] * 0.2 +
                player_stats['rebounds_per_game'] * 0.2 +
                (player_stats['steals_per_game'] + player_stats['blocks_per_game']) * 0.2
            )
            score += 0.25 * performance_score
            
            # 2. Team needs analysis (25%)
            # Get existing players' stats
            existing_stats = []
            for player in existing_home_players:
                stats = self.feature_engineer.calculate_player_statistics(player, home_team, self.feature_engineer.df)
                existing_stats.append(stats)
            
            # Calculate team averages
            team_avg_points = np.mean([s['points_per_game'] for s in existing_stats])
            team_avg_assists = np.mean([s['assists_per_game'] for s in existing_stats])
            team_avg_rebounds = np.mean([s['rebounds_per_game'] for s in existing_stats])
            team_avg_defense = np.mean([s['steals_per_game'] + s['blocks_per_game'] for s in existing_stats])
            
            # Calculate team needs
            needs_score = 0.0
            if player_stats['points_per_game'] > team_avg_points * 1.2:
                needs_score += 0.25
            if player_stats['assists_per_game'] > team_avg_assists * 1.2:
                needs_score += 0.25
            if player_stats['rebounds_per_game'] > team_avg_rebounds * 1.2:
                needs_score += 0.25
            if (player_stats['steals_per_game'] + player_stats['blocks_per_game']) > team_avg_defense * 1.2:
                needs_score += 0.25
            
            score += 0.25 * needs_score
            
            # 3. Position compatibility (20%)
            # Get existing positions
            existing_positions = []
            for stats in existing_stats:
                existing_positions.append(stats['primary_position'])
            
            # Modern NBA position mapping with more granular positions
            position_groups = {
                'PG': ['PG', 'Combo Guard', 'Point Forward'],
                'SG': ['SG', 'Combo Guard', 'Wing', '3&D'],
                'SF': ['SF', 'Wing', 'Forward', '3&D', 'Point Forward'],
                'PF': ['PF', 'Forward', 'Big', 'Stretch 4', 'Small Ball 5'],
                'C': ['C', 'Big', 'Center', 'Small Ball 5']
            }
            
            # Calculate position compatibility with position versatility bonus
            candidate_position = player_stats['primary_position']
            position_compatibility = 0.0
            versatility_bonus = 0.0
            
            for group in position_groups.values():
                if candidate_position in group:
                    for existing_pos in existing_positions:
                        if existing_pos in group:
                            position_compatibility += 1.0
                            break
                    versatility_bonus += 0.1
            
            position_compatibility = min(1.0, position_compatibility / len(existing_positions))
            versatility_bonus = min(0.2, versatility_bonus)  # Cap the versatility bonus
            score += 0.2 * (position_compatibility + versatility_bonus)
            
            # 4. Team chemistry (15%)
            # Calculate chemistry based on shared minutes and success rate
            chemistry_score = player_stats['team_chemistry']
            
            # Calculate player combination success rate
            combination_success = 0.0
            for existing_player in existing_home_players:
                existing_stats = self.feature_engineer.calculate_player_statistics(existing_player, home_team, self.feature_engineer.df)
                # Check if players have played together successfully
                if (existing_stats.get('shared_minutes', 0) > 100 and 
                    existing_stats.get('success_rate', 0) > 0.5):
                    combination_success += 1.0
            
            combination_success = min(1.0, combination_success / len(existing_home_players))
            chemistry_score = (chemistry_score + combination_success) / 2
            
            # Adjust chemistry based on player roles and playstyles
            role_compatibility = 0.0
            for existing_player in existing_home_players:
                existing_stats = self.feature_engineer.calculate_player_statistics(existing_player, home_team, self.feature_engineer.df)
                # Check for complementary playstyles
                if (existing_stats['primary_position'] in position_groups[candidate_position] and
                    existing_stats.get('playstyle', '') != player_stats.get('playstyle', '')):
                    role_compatibility += 1.0
            
            role_compatibility = min(1.0, role_compatibility / len(existing_home_players))
            chemistry_score = (chemistry_score + role_compatibility) / 2
            
            score += 0.15 * chemistry_score
            
            # 5. Recent performance (15%)
            # Calculate recent performance with more metrics
            recent_games = player_stats.get('recent_games', [])
            if recent_games:
                recent_score = 0.0
                for game in recent_games[-5:]:  # Last 5 games
                    game_score = (
                        game.get('points', 0) * 0.4 +
                        game.get('assists', 0) * 0.2 +
                        game.get('rebounds', 0) * 0.2 +
                        (game.get('steals', 0) + game.get('blocks', 0)) * 0.2
                    )
                    recent_score += game_score
                recent_score = min(1.0, recent_score / 100)  # Normalize to 0-1
            else:
                recent_score = 0.5  # Default score if no recent games
            
            score += 0.15 * recent_score
            
            # Store the score
            player_scores[candidate] = score
            candidates_evaluated += 1
            
            if candidates_evaluated % 100 == 0:
                logger.info(f"Evaluated {candidates_evaluated} candidate players...")
        
        # Select the player with the highest score
        if player_scores:
            optimal_player = max(player_scores.items(), key=lambda x: x[1])[0]
            logger.info(f"Selected optimal player: {optimal_player} (score: {player_scores[optimal_player]:.4f})")
            return optimal_player
        else:
            logger.warning("No valid candidate players found")
            return None
    
    def predict_optimal_player_stats_only(self, lineup_data):
        """
        Predict the optimal fifth player using only player statistics.
        
        Args:
            lineup_data (pd.DataFrame): Data for the lineup
            
        Returns:
            list: List of predicted player names
        """
        if self.player_stats is None:
            raise ValueError("Player statistics must be set before prediction")
            
        logger.info("Predicting optimal fifth player using statistics...")
        
        # Copy the data to avoid modifying the original
        data = lineup_data.copy()
        
        # Get the indices of rows with missing home player (marked with '?')
        missing_player_mask = data.apply(
            lambda row: any(row[f'home_{i}'] == '?' for i in range(5) if f'home_{i}' in row),
            axis=1
        )
        missing_player_indices = data[missing_player_mask].index
        
        if len(missing_player_indices) == 0:
            logger.warning("No missing home players found in the data")
            return []
        
        # Get the optimal player for each row with a missing player
        optimal_players = []
        
        for idx in missing_player_indices:
            lineup_row = data.loc[idx:idx]
            
            # Score all available players
            player_scores = {}
            for candidate in self.player_candidates:
                # Skip players already in the lineup
                if any(candidate == lineup_row[f'home_{i}'].iloc[0] 
                      for i in range(5) if f'home_{i}' in lineup_row):
                    continue
                if any(candidate == lineup_row[f'away_{i}'].iloc[0] 
                      for i in range(5) if f'away_{i}' in lineup_row):
                    continue
                
                # Calculate player score
                score = self._calculate_player_score(candidate, lineup_row)
                player_scores[candidate] = score
            
            # Select the player with the highest score
            if player_scores:
                optimal_player = max(player_scores.items(), key=lambda x: x[1])[0]
                optimal_players.append(optimal_player)
                logger.info(f"Selected {optimal_player} with score {player_scores[optimal_player]:.4f}")
            else:
                logger.warning("No valid candidates found for lineup")
                optimal_players.append(None)
        
        logger.info(f"Predicted {len(optimal_players)} optimal players")
        return optimal_players
    
    def _calculate_player_compatibility(self, player, existing_players):
        """
        Calculate compatibility score between a player and existing lineup.
        
        Args:
            player (str): Player to evaluate
            existing_players (list): List of players already in the lineup
            
        Returns:
            float: Compatibility score between 0 and 1
        """
        if not self.player_stats or player not in self.player_stats:
            return 0.0
            
        player_stats = self.player_stats[player]
        compatibility_score = 0.0
        
        # Calculate position diversity score
        positions = set(player_stats.get('position', '').split('/'))
        existing_positions = set()
        for existing_player in existing_players:
            if existing_player in self.player_stats:
                existing_positions.update(
                    self.player_stats[existing_player].get('position', '').split('/')
                )
        position_diversity = len(positions - existing_positions) / max(1, len(positions))
        
        # Calculate skill complementarity
        player_skills = {
            'scoring': player_stats.get('points_per_game', 0),
            'assists': player_stats.get('assists_per_game', 0),
            'rebounds': player_stats.get('rebounds_per_game', 0),
            'defense': player_stats.get('steals_per_game', 0) + player_stats.get('blocks_per_game', 0)
        }
        
        existing_skills = {skill: 0.0 for skill in player_skills}
        for existing_player in existing_players:
            if existing_player in self.player_stats:
                existing_stats = self.player_stats[existing_player]
                existing_skills['scoring'] += existing_stats.get('points_per_game', 0)
                existing_skills['assists'] += existing_stats.get('assists_per_game', 0)
                existing_skills['rebounds'] += existing_stats.get('rebounds_per_game', 0)
                existing_skills['defense'] += (
                    existing_stats.get('steals_per_game', 0) + 
                    existing_stats.get('blocks_per_game', 0)
                )
        
        # Normalize existing skills
        num_players = len(existing_players) or 1
        for skill in existing_skills:
            existing_skills[skill] /= num_players
        
        # Calculate skill complementarity score
        skill_needs = {}
        for skill, value in existing_skills.items():
            if value < 10:  # Below average, need this skill
                skill_needs[skill] = (10 - value) / 10
            else:  # Above average, less important
                skill_needs[skill] = 0.2
        
        skill_score = sum(
            player_skills[skill] * importance 
            for skill, importance in skill_needs.items()
        ) / sum(skill_needs.values())
        
        # Calculate experience compatibility
        player_experience = player_stats.get('experience', 0)
        existing_experience = sum(
            self.player_stats[p].get('experience', 0) 
            for p in existing_players 
            if p in self.player_stats
        ) / num_players
        
        experience_diff = abs(player_experience - existing_experience)
        experience_score = 1 - min(1, experience_diff / 10)
        
        # Combine scores with weights
        compatibility_score = (
            0.3 * position_diversity +
            0.5 * skill_score +
            0.2 * experience_score
        )
        
        return compatibility_score
    
    def _calculate_player_score(self, player, lineup_row):
        """
        Calculate overall score for a player in a given lineup context.
        
        Args:
            player (str): Player to evaluate
            lineup_row (pd.DataFrame): Single row dataframe with lineup data
            
        Returns:
            float: Player score between 0 and 1
        """
        if not self.player_stats or player not in self.player_stats:
            return 0.0
            
        player_stats = self.player_stats[player]
        
        # Get existing home players
        existing_home_players = []
        for i in range(5):
            col = f'home_{i}'
            if col in lineup_row:
                p = lineup_row[col].iloc[0]
                if p != '?' and not pd.isna(p):
                    existing_home_players.append(p)
        
        # Calculate base performance score
        performance_score = (
            0.4 * player_stats.get('win_rate', 0) +
            0.3 * min(1, player_stats.get('points_per_game', 0) / 30) +
            0.15 * min(1, player_stats.get('assists_per_game', 0) / 10) +
            0.15 * min(1, player_stats.get('rebounds_per_game', 0) / 15)
        )
        
        # Calculate compatibility score
        compatibility_score = self._calculate_player_compatibility(
            player, existing_home_players
        )
        
        # Calculate recency score based on recent performance
        recency_score = player_stats.get('recent_performance', 0.5)
        
        # Calculate final score with weights
        final_score = (
            0.4 * performance_score +
            0.4 * compatibility_score +
            0.2 * recency_score
        )
        
        return final_score
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test (pd.DataFrame): Test feature matrix
            y_test (np.ndarray): Test target vector
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info("Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def save(self, model_path):
        """
        Save the trained model to disk.
        
        Args:
            model_path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        logger.info(f"Saving model to {model_path}...")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        joblib.dump(self.model, model_path)
        
        logger.info("Model saved successfully")
    
    def load(self, model_path):
        """
        Load a trained model from disk.
        
        Args:
            model_path (str): Path to the saved model
        """
        logger.info(f"Loading model from {model_path}...")
        
        # Load the model
        self.model = joblib.load(model_path)
        
        logger.info("Model loaded successfully")
        
        return self

class TeamChemistryModel(BaseEstimator, ClassifierMixin):
    """
    Custom model that incorporates team chemistry into the prediction.
    This model considers player combinations and their historical performance.
    """
    
    def __init__(self, base_model=None):
        """
        Initialize the team chemistry model.
        
        Args:
            base_model (BaseEstimator): Base model to use for predictions
        """
        self.base_model = base_model or RandomForestClassifier(n_estimators=100, random_state=42)
        self.player_pair_outcomes = {}  # Dict to store outcomes of player pairs
    
    def fit(self, X, y):
        """
        Fit the model to the training data.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (np.ndarray): Target vector
            
        Returns:
            self: Fitted model
        """
        # Train the base model
        self.base_model.fit(X, y)
        
        # Additional logic for team chemistry could be added here
        # For example, calculating success rates for player combinations
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Args:
            X (pd.DataFrame): Test samples
            
        Returns:
            np.ndarray: Predicted class labels
        """
        # Get base model predictions
        base_preds = self.base_model.predict(X)
        
        # Additional logic for team chemistry could be added here
        # For example, adjusting predictions based on player combinations
        
        return base_preds
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Args:
            X (pd.DataFrame): Test samples
            
        Returns:
            np.ndarray: Predicted class probabilities
        """
        # Get base model probability predictions
        base_probs = self.base_model.predict_proba(X)
        
        # Additional logic for team chemistry could be added here
        # For example, adjusting probabilities based on player combinations
        
        return base_probs

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np
    
    # Create dummy data
    X = pd.DataFrame(np.random.randn(100, 10))
    y = np.random.randint(0, 2, 100)
    
    # Create the model
    model = LineupPredictor(model_type='random_forest')
    
    # Train the model
    model.train(X, y)
    
    # Evaluate the model
    X_test = pd.DataFrame(np.random.randn(20, 10))
    y_test = np.random.randint(0, 2, 20)
    metrics = model.evaluate(X_test, y_test)
    
    print(f"Test accuracy: {metrics['accuracy']:.4f}") 