"""
Visualization module for NBA lineup prediction project.
This module contains functions for generating plots and charts for data analysis and results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('nba_visualizer')

# Set seaborn style
sns.set_style('whitegrid')

class NBAVisualizer:
    """
    Class for generating visualizations for the NBA lineup prediction project.
    """
    
    def __init__(self, output_dir='figures'):
        """
        Initialize the visualizer.
        
        Args:
            output_dir (str): Directory to save output figures
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_team_win_rates(self, data, season=None):
        """
        Plot win rates for NBA teams.
        
        Args:
            data (pd.DataFrame): NBA game data
            season (str, optional): Season to filter by
        
        Returns:
            plt.Figure: The generated figure
        """
        logger.info("Generating team win rates plot...")
        
        # Filter by season if specified
        if season:
            data = data[data['season'] == season]
        
        # Calculate win rates for home teams
        home_teams = data['home_team'].unique()
        win_rates = []
        
        for team in home_teams:
            team_games = data[data['home_team'] == team]
            wins = len(team_games[team_games['outcome'] == 1])
            total = len(team_games)
            win_rate = wins / total if total > 0 else 0
            win_rates.append((team, win_rate))
        
        # Sort by win rate
        win_rates.sort(key=lambda x: x[1], reverse=True)
        
        # Create the plot
        plt.figure(figsize=(14, 8))
        teams, rates = zip(*win_rates)
        
        bars = plt.bar(teams, rates, color='royalblue')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.2f}', ha='center', va='bottom')
        
        plt.title(f'Home Team Win Rates{" for " + season if season else ""}', fontsize=16)
        plt.xlabel('Team', fontsize=12)
        plt.ylabel('Win Rate', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the figure
        filename = f'team_win_rates{"_" + season if season else ""}.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        
        logger.info(f"Saved team win rates plot as {filename}")
        
        return plt.gcf()
    
    def plot_player_stats(self, player_stats, top_n=20, metric='win_rate'):
        """
        Plot statistics for top players.
        
        Args:
            player_stats (dict): Dictionary of player statistics
            top_n (int): Number of top players to display
            metric (str): Metric to sort by ('win_rate', 'avg_points', 'avg_rebounds', 'avg_assists')
        
        Returns:
            plt.Figure: The generated figure
        """
        logger.info(f"Generating player stats plot for top {top_n} players by {metric}...")
        
        # Convert player stats to DataFrame
        stats_df = pd.DataFrame.from_dict(player_stats, orient='index')
        
        # Filter players with at least 10 games
        stats_df = stats_df[stats_df['total_games'] >= 10]
        
        # Sort by the specified metric
        stats_df = stats_df.sort_values(by=metric, ascending=False)
        
        # Get top N players
        top_players = stats_df.head(top_n)
        
        # Create the plot
        plt.figure(figsize=(14, 8))
        
        bars = plt.bar(top_players.index, top_players[metric], color='royalblue')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.2f}', ha='center', va='bottom')
        
        # Set plot title and labels
        metric_labels = {
            'win_rate': 'Win Rate',
            'avg_points': 'Average Points per Game',
            'avg_rebounds': 'Average Rebounds per Game',
            'avg_assists': 'Average Assists per Game'
        }
        
        plt.title(f'Top {top_n} Players by {metric_labels.get(metric, metric)}', fontsize=16)
        plt.xlabel('Player', fontsize=12)
        plt.ylabel(metric_labels.get(metric, metric), fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the figure
        filename = f'top_players_{metric}.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        
        logger.info(f"Saved player stats plot as {filename}")
        
        return plt.gcf()
    
    def plot_feature_importance(self, model, feature_names):
        """
        Plot feature importances for the model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names (list): Names of features
        
        Returns:
            plt.Figure: The generated figure
        """
        logger.info("Generating feature importance plot...")
        
        # Check if model has feature_importances_ attribute
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return None
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort feature importances
        indices = np.argsort(importances)[::-1]
        
        # Take top 20 features
        indices = indices[:20]
        
        # Create the plot
        plt.figure(figsize=(14, 8))
        
        bars = plt.bar(range(len(indices)), importances[indices], color='royalblue')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                     f'{height:.3f}', ha='center', va='bottom')
        
        plt.title('Feature Importances', fontsize=16)
        plt.xlabel('Feature', fontsize=12)
        plt.ylabel('Importance', fontsize=12)
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the figure
        filename = 'feature_importance.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        
        logger.info(f"Saved feature importance plot as {filename}")
        
        return plt.gcf()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot confusion matrix for model predictions.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
        
        Returns:
            plt.Figure: The generated figure
        """
        logger.info("Generating confusion matrix plot...")
        
        # Create confusion matrix
        cm = pd.crosstab(
            pd.Series(y_true, name='Actual'),
            pd.Series(y_pred, name='Predicted')
        )
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        
        plt.title('Confusion Matrix', fontsize=16)
        plt.tight_layout()
        
        # Save the figure
        filename = 'confusion_matrix.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        
        logger.info(f"Saved confusion matrix plot as {filename}")
        
        return plt.gcf()
    
    def plot_test_matches_per_year(self, test_data):
        """
        Plot number of matches per year in the test dataset.
        
        Args:
            test_data (pd.DataFrame): Test dataset
        
        Returns:
            plt.Figure: The generated figure
        """
        logger.info("Generating test matches per year plot...")
        
        # Count matches per year
        year_counts = test_data['season'].value_counts().sort_index()
        
        # Calculate average
        avg_matches = year_counts.mean()
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        bars = plt.bar(year_counts.index, year_counts.values, color='royalblue')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'{height:.0f}', ha='center', va='bottom')
        
        # Add average line
        plt.axhline(y=avg_matches, color='red', linestyle='--', label=f'Average: {avg_matches:.2f}')
        
        plt.title('Number of Matches per Year in Test Dataset', fontsize=16)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Number of Matches', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        # Save the figure
        filename = 'test_matches_per_year.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        
        logger.info(f"Saved test matches per year plot as {filename}")
        
        return plt.gcf()
    
    def plot_player_chemistry(self, player1, player2, outcome_data):
        """
        Plot chemistry between two players.
        
        Args:
            player1 (str): Name of first player
            player2 (str): Name of second player
            outcome_data (pd.DataFrame): Data with game outcomes
        
        Returns:
            plt.Figure: The generated figure
        """
        logger.info(f"Generating player chemistry plot for {player1} and {player2}...")
        
        # Filter games where both players are on the same home team
        # This requires some preprocessing of the data to find such games
        
        # For demonstration, we'll use sample data
        # In a real implementation, you would find actual data for these players
        
        # Sample data: winning percentage when playing together vs. separately
        together_win_rate = 0.65
        player1_alone_win_rate = 0.55
        player2_alone_win_rate = 0.50
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        categories = [f'{player1} alone', f'{player2} alone', 'Together']
        win_rates = [player1_alone_win_rate, player2_alone_win_rate, together_win_rate]
        
        bars = plt.bar(categories, win_rates, color=['lightblue', 'lightblue', 'royalblue'])
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.2f}', ha='center', va='bottom')
        
        plt.title(f'Win Rates: {player1} and {player2}', fontsize=16)
        plt.xlabel('Player Combination', fontsize=12)
        plt.ylabel('Win Rate', fontsize=12)
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the figure
        filename = f'player_chemistry_{player1.replace(" ", "_")}_{player2.replace(" ", "_")}.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        
        logger.info(f"Saved player chemistry plot as {filename}")
        
        return plt.gcf()
    
    def plot_prediction_examples(self, test_examples, predictions, actual=None, n=5):
        """
        Plot examples of predictions.
        
        Args:
            test_examples (pd.DataFrame): Test data examples
            predictions (list): Predicted players
            actual (list, optional): Actual players (ground truth)
            n (int): Number of examples to show
        
        Returns:
            plt.Figure: The generated figure
        """
        logger.info(f"Generating prediction examples plot for {n} examples...")
        
        n = min(n, len(test_examples))
        
        # Create a figure with n rows of 2 columns each
        fig, axes = plt.subplots(n, 1, figsize=(12, 5 * n))
        
        if n == 1:
            axes = [axes]
        
        for i in range(n):
            ax = axes[i]
            
            example = test_examples.iloc[i]
            prediction = predictions[i] if i < len(predictions) else "N/A"
            ground_truth = actual[i] if actual and i < len(actual) else None
            
            # Get home and away team
            home_team = example['home_team']
            away_team = example['away_team']
            
            # Get players
            home_players = []
            for j in range(5):
                col = f'home_{j}'
                if col in example and example[col] != '?' and not pd.isna(example[col]):
                    home_players.append(example[col])
                elif col in example and example[col] == '?':
                    home_players.append(f"MISSING (Predicted: {prediction})")
            
            away_players = []
            for j in range(5):
                col = f'away_{j}'
                if col in example and example[col] != '?' and not pd.isna(example[col]):
                    away_players.append(example[col])
            
            # Create a table-like visualization
            ax.axis('off')
            ax.set_title(f"Example {i+1}: {home_team} vs {away_team}", fontsize=14)
            
            # Home team
            home_y = 0.8
            ax.text(0.05, home_y, f"{home_team} (Home):", fontweight='bold', fontsize=12)
            for j, player in enumerate(home_players):
                home_y -= 0.1
                if "MISSING" in player:
                    ax.text(0.1, home_y, player, color='red', fontsize=11)
                else:
                    ax.text(0.1, home_y, player, fontsize=11)
            
            # Away team
            away_y = 0.4
            ax.text(0.05, away_y, f"{away_team} (Away):", fontweight='bold', fontsize=12)
            for j, player in enumerate(away_players):
                away_y -= 0.1
                ax.text(0.1, away_y, player, fontsize=11)
            
            # Prediction and ground truth
            if ground_truth:
                ax.text(0.7, 0.8, f"Predicted: {prediction}", fontsize=12)
                ax.text(0.7, 0.7, f"Actual: {ground_truth}", fontsize=12)
                ax.text(0.7, 0.6, f"Correct: {'Yes' if prediction == ground_truth else 'No'}", 
                       fontsize=12, color='green' if prediction == ground_truth else 'red')
            else:
                ax.text(0.7, 0.8, f"Predicted: {prediction}", fontsize=12)
        
        plt.tight_layout()
        
        # Save the figure
        filename = 'prediction_examples.png'
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        
        logger.info(f"Saved prediction examples plot as {filename}")
        
        return fig

if __name__ == "__main__":
    # Example usage with dummy data
    import numpy as np
    
    # Create visualizer
    visualizer = NBAVisualizer(output_dir='figures')
    
    # Generate dummy data
    np.random.seed(42)
    
    # Team win rates
    teams = ['LAL', 'GSW', 'BOS', 'MIA', 'CHI', 'NYK', 'DAL', 'PHI', 'HOU', 'TOR']
    seasons = ['2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']
    
    data = {
        'home_team': np.random.choice(teams, 1000),
        'away_team': np.random.choice(teams, 1000),
        'season': np.random.choice(seasons, 1000),
        'outcome': np.random.randint(0, 2, 1000)
    }
    
    df = pd.DataFrame(data)
    
    # Plot team win rates
    visualizer.plot_team_win_rates(df)
    
    # Player stats
    player_stats = {}
    players = [
        'LeBron James', 'Stephen Curry', 'Kevin Durant', 'Giannis Antetokounmpo',
        'James Harden', 'Kawhi Leonard', 'Anthony Davis', 'Nikola Jokic',
        'Joel Embiid', 'Luka Doncic', 'Jayson Tatum', 'Damian Lillard',
        'Klay Thompson', 'Draymond Green', 'Kyrie Irving', 'Russell Westbrook',
        'Chris Paul', 'Jimmy Butler', 'Devin Booker', 'Zion Williamson',
        'Trae Young', 'Ja Morant', 'Donovan Mitchell', 'Bam Adebayo'
    ]
    
    for player in players:
        player_stats[player] = {
            'total_games': np.random.randint(50, 300),
            'wins': np.random.randint(30, 200),
            'win_rate': np.random.uniform(0.4, 0.8),
            'avg_points': np.random.uniform(10, 30),
            'avg_rebounds': np.random.uniform(3, 12),
            'avg_assists': np.random.uniform(2, 10)
        }
    
    # Plot player stats
    visualizer.plot_player_stats(player_stats, metric='win_rate')
    
    # Test data
    test_data = pd.DataFrame({
        'season': ['2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015'],
        'count': np.random.randint(80, 150, 9)
    })
    test_data.set_index('season', inplace=True)
    
    # Plot test matches per year
    visualizer.plot_test_matches_per_year(pd.DataFrame(data)) 