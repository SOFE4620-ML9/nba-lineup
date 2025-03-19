"""
Main script for NBA lineup prediction project.
This script ties together all the modules for data loading, feature engineering, 
model training, and prediction.
"""

import os
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import joblib

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

from data.data_loader import NBADataLoader
from data.feature_engineering import NBAFeatureEngineer
from models.lineup_predictor import LineupPredictor
from visualization.visualizer import NBAVisualizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='NBA Lineup Prediction')
    
    # Changed default data path to point to dataset directory
    parser.add_argument('--data-path', type=str, default='dataset',
                        help='Path to directory containing training/evaluation data')
    
    # Keep other arguments
    parser.add_argument('--full', action='store_true',
                        help='Run on full dataset (2007-2015)')
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Output directory for results and logs')
    parser.add_argument('--model-type', type=str, default='random_forest',
                        choices=['random_forest', 'gradient_boosting', 'logistic'],
                        help='Type of model to use')
    parser.add_argument('--save-model', action='store_true',
                        help='Save the trained model')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to load a trained model from')
    parser.add_argument('--years', type=str, default=None,
                        help='Comma-separated list of years to use for training (e.g., "2007,2008,2009")')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--predict-only', action='store_true',
                        help='Only perform prediction without training')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'predict'],
                        help='Mode of operation: train or predict')
    parser.add_argument('--team', type=str, default=None,
                        help='Team to calculate player statistics for')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    return args

def main():
    """Main function for the NBA lineup prediction project."""
    args = parse_args()
    
    # Add early validation
    if not os.path.exists(args.test_data):
        raise FileNotFoundError(f"Test data file {args.test_data} not found")
        
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Convert to absolute path early
    args.output_dir = os.path.abspath(args.output_dir)
    print(f"DEBUG: Writing outputs to {args.output_dir}")  # For verification
    
    # Validate arguments early
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path {args.data_path} does not exist")
        
    # Setup logging FIRST
    log_file = os.path.join(args.output_dir, f"nba_lineup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('nba_main')

    try:
        # Rest of main logic
        loader = NBADataLoader(data_path=args.data_path)
        engineer = NBAFeatureEngineer()
        
        # Load and preprocess data
        training_data = loader.load_training_data(years=args.years if not args.full else list(range(2007, 2016)))
        processed_data = loader.preprocess_training_data(training_data=training_data)
        if args.mode == 'train':
            # Track processing progress
            players = list(processed_data['home_player_1'].unique()) + \
                     list(processed_data['away_player_1'].unique())
            players = list(set(players))
            
            if tqdm is not None:
                with tqdm(total=len(players), desc="Processing players") as pbar:
                    player_stats = {}
                    for player in players:
                        player_stats[player] = engineer.calculate_player_statistics(
                            player, team=args.team, df=processed_data
                        )
                        pbar.update(1)
            else:
                player_stats = {}
                for player in players:
                    player_stats[player] = engineer.calculate_player_statistics(
                        player, team=args.team, df=processed_data
                    )
        else:
            # Get home team from the first available game
            home_team = processed_data.iloc[0]['home_team']
            
            # Filter home team players
            home_team_players = processed_data[processed_data['home_team'] == home_team][
                ['home_player_1', 'home_player_2', 'home_player_3', 'home_player_4', 'home_player_5']
            ]
            
            # Remove any placeholder values
            home_team_players = home_team_players.values.flatten()
            home_team_players = [p for p in home_team_players if p not in ('?', None)]
            
            if home_team_players:
                player_stats = {}
                for player in home_team_players:
                    player_stats[player] = engineer.calculate_player_statistics(
                        player=player,
                        team=home_team,
                        df=processed_data
                    )
            else:
                player_stats = {}
        
        # Fit feature engineering transformations
        engineer.fit(processed_data)
        
        # Transform data for model training
        X, y = engineer.transform_training_data(processed_data)
        
        # Initialize model
        model = LineupPredictor(model_type=args.model_type)
        
        # Set feature engineer, player candidates, and stats
        model.set_feature_engineer(engineer)
        model.set_player_candidates(loader.get_player_candidates())
        model.set_player_stats(player_stats)
        
        # Load model if specified, otherwise train
        if args.load_model:
            logger.info(f"Loading model from {args.load_model}...")
            model.load(args.load_model)
        else:
            logger.info("Training model...")
            model.train(X, y)
            
            # Save model if specified
            if args.save_model:
                model_path = os.path.join(args.output_dir, f"{args.model_type}_model.pkl")
                model.save(model_path)
                
                # Also save player statistics separately
                stats_path = os.path.join(args.output_dir, "player_stats.pkl")
                joblib.dump(player_stats, stats_path)
                logger.info(f"Saved player statistics to {stats_path}")
        # Load test data
        logger.info("Loading test data...")
        test_data, test_labels = loader.load_test_data()
        logger.info(f"Test data has {len(test_data.columns)} columns: {list(test_data.columns)}")
        
        # Preprocess test data
        logger.info("Preprocessing test data...")
        processed_test = loader.preprocess_test_data(test_data)
        
        # Check if test data has the same structure as training data would have
        # If not, use the stats-only prediction method
        if len(test_data.columns) < 30:  # Test data has fewer columns
            logger.info("Test data has different structure than training data")
            logger.info("Using statistics-only prediction method")
            
            # Predict optimal fifth player using player statistics only
            optimal_players = model.predict_optimal_player_stats_only(processed_test)
            
            # Debug: Direct print statement to see what's in optimal_players
            print("DEBUGGING IN MAIN.PY - AFTER RECEIVING PREDICTIONS:")
            print(f"Type of optimal_players: {type(optimal_players)}")
            print(f"Length of optimal_players: {len(optimal_players)}")
            print(f"First 20 predictions: {optimal_players[:20]}")
            unique_players = set(optimal_players)
            print(f"Number of unique players: {len(unique_players)}")
            print(f"Unique players: {list(unique_players)[:20]}")
            
            # Debug: Check the diversity of predictions
            unique_players = set(optimal_players)
            logger.info(f"Number of unique players predicted: {len(unique_players)}")
            logger.info(f"Top 10 most common predictions: {pd.Series(optimal_players).value_counts().head(10).to_dict()}")
        else:
            # Use the standard prediction method
            logger.info("Using standard prediction method")
            optimal_players = model.predict_optimal_player(processed_test)
            
            # Debug: Check the diversity of predictions
            unique_players = set(optimal_players)
            logger.info(f"Number of unique players predicted: {len(unique_players)}")
            logger.info(f"Top 10 most common predictions: {pd.Series(optimal_players).value_counts().head(10).to_dict()}")
        
        # Verify that optimal_players list is populated correctly
        logger.info(f"Length of optimal_players: {len(optimal_players)}")
        if len(optimal_players) > 0:
            logger.info(f"First few predictions: {optimal_players[:5]}")
        
        # Ensure predictions are generated even if empty
        if len(optimal_players) == 0:
            logger.warning("No optimal players predicted - generating placeholder")
            optimal_players = ["Unknown"] * len(processed_test)

        # Save predictions with absolute path
        predictions_df = pd.DataFrame({
            'Game_ID': processed_test.get('game', [f'GAME_{i:04d}' for i in range(len(optimal_players))]),
            'Home_Team': processed_test.get('home_team', ['UNKNOWN_TEAM']*len(optimal_players)),
            'Fifth_Player': optimal_players
        })
        
        pred_path = os.path.join(args.output_dir, "predictions.csv")
        print(f"DEBUG: Saving predictions to {pred_path}")
        predictions_df.to_csv(pred_path, index=False)
        logger.info(f"Saved predictions to {pred_path}")

    except Exception as e:
        logger.error(f"Critical failure: {str(e)}", exc_info=True)
        raise SystemExit(1) from e