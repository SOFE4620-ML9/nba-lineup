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

from src.data.data_loader import NBADataLoader
from src.data.feature_engineering import NBAFeatureEngineer
from src.models.lineup_predictor import LineupPredictor
from src.visualization.visualizer import NBAVisualizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"nba_lineup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('nba_main')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='NBA Lineup Prediction')
    
    # Changed default data path to point to dataset directory
    parser.add_argument('--data-path', type=str, default='dataset',
                        help='Path to directory containing training/evaluation data')
    
    # Remove any existing --data-dir arguments
    # parser.add_argument('--data-dir', ...)  # Delete this if exists
    
    # Keep other arguments
    parser.add_argument('--full', action='store_true',
                        help='Run on full dataset (2007-2015)')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save output files')
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
    
    return parser.parse_args()

def main():
    """Main function for the NBA lineup prediction project."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize components with corrected data path
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
    
    # Save predictions to file
    predictions_file = os.path.join(args.output_dir, 'predictions.csv')
    
    # Debug: Check what's going into the DataFrame
    print("DEBUGGING - CREATING DATAFRAME FOR PREDICTIONS:")
    print(f"Game_ID length: {len(processed_test.get('game', [f'Game_{i}' for i in range(len(optimal_players))]))}")
    print(f"Home_Team length: {len(processed_test['home_team'])}")
    print(f"Fifth_Player length: {len(optimal_players)}")
    
    predictions_df = pd.DataFrame({
        'Game_ID': processed_test.get('game', [f'Game_{i}' for i in range(len(optimal_players))]),
        'Home_Team': processed_test['home_team'],
        'Fifth_Player': optimal_players
    })
    
    # Verify data before saving
    print("DEBUGGING - DATAFRAME BEFORE SAVING:")
    print(f"DataFrame shape: {predictions_df.shape}")
    print(f"DataFrame head:\n{predictions_df.head(20).to_string()}")
    print(f"Fifth_Player value counts:\n{predictions_df['Fifth_Player'].value_counts().head(10)}")
    
    # Verify data before saving
    logger.info(f"Predictions DataFrame shape: {predictions_df.shape}")
    logger.info(f"Predictions DataFrame head:\n{predictions_df.head().to_string()}")
    
    predictions_df.to_csv(predictions_file, index=False)
    logger.info(f"Saved predictions to {predictions_file}")
    
    # Calculate prediction accuracy if we have ground truth
    if 'removed_value' in test_labels.columns:
        logger.info("Calculating prediction accuracy...")
        ground_truth = test_labels['removed_value'].values
        correct = sum(p == gt for p, gt in zip(optimal_players, ground_truth))
        accuracy = correct / len(ground_truth)
        logger.info(f"Prediction accuracy: {accuracy:.4f} ({correct}/{len(ground_truth)})")
        
        # Save accuracy to file
        accuracy_file = os.path.join(args.output_dir, 'accuracy.txt')
        with open(accuracy_file, 'w') as f:
            f.write(f"Prediction accuracy: {accuracy:.4f} ({correct}/{len(ground_truth)})\n")
        logger.info(f"Saved accuracy to {accuracy_file}")
    
    # Generate visualizations if specified
    if args.visualize:
        logger.info("Generating visualizations...")
        visualizer = NBAVisualizer(output_dir=os.path.join(args.output_dir, 'figures'))
        
        # Plot test matches per year
        if 'season' in processed_test.columns:
            visualizer.plot_test_matches_per_year(processed_test)
        
        # Plot player statistics
        if 'player_stats' in locals():
            visualizer.plot_player_stats(player_stats, metric='win_rate')
            visualizer.plot_player_stats(player_stats, metric='avg_points')
        
        # Plot prediction examples
        if 'ground_truth' in locals():
            visualizer.plot_prediction_examples(
                processed_test, 
                optimal_players,
                ground_truth, 
                n=5
            )
        
        # Plot feature importance if available
        if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
            feature_names = model.model.feature_names_in_ if hasattr(model.model, 'feature_names_in_') else X.columns if 'X' in locals() else None
            if feature_names is not None:
                visualizer.plot_feature_importance(model.model, feature_names)
    
    logger.info("NBA Lineup Prediction complete!")

if __name__ == "__main__":
    main() 