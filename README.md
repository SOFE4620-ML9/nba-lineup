# NBA Lineup Prediction

## Project Objective
The goal of this project is to predict the optimal fifth player for a home team in an NBA game, given partial lineup data and other game-related features. The model maximizes the home team's overall performance based on historical data.

## Dataset
The dataset contains lineups of all NBA games from 2007 to 2015, including:
- Game-related features
- Player statistics
- Team compositions
- Game outcomes

## Project Structure
```
├── dataset               # Raw data files
│   ├── training          # Training data files (2007-2015)
│   ├── evaluation        # Test data and labels (renamed from 'eval')
│   └── Matchups_metadata.txt  # Data dictionary
├── src                   # Source code
│   ├── data              # Data processing modules
│   ├── models            # Machine learning models
│   └── visualization     # Data visualization scripts
├── notebooks             # Jupyter notebooks for exploration and analysis
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## Setup and Installation

### Instant Run with Nix (No Clone Needed!)
The easiest way to use this project is to run it directly from GitHub using Nix Flakes without even cloning the repository:

> **Note:** If you don't have Nix installed, you can install it using the Determinate Nix Installer: 
> ```bash
> curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install
> ```

```bash
# Enable the binary cache for faster installation (one-time setup)
nix run github:cachix/cachix -- use sofe4620ml9

# Run the model on a sample dataset (2015 only)
nix run github:sofe4620-ml9/nba-lineup

# Run the model on the full dataset
nix run github:sofe4620-ml9/nba-lineup#run-full

# Run the test suite
nix run github:sofe4620-ml9/nba-lineup#test
```

These commands will:
1. Download the repository
2. Set up all dependencies
3. Run the model with the appropriate configuration
4. Output results to a temporary directory

All dependencies are automatically handled by Nix, and binary caching makes subsequent runs much faster.

### One-Click Setup with Nix (Local Development)
If you want to develop or modify the code, clone the repository and run:

```bash
# Set up the complete development environment
nix develop
```

This single command will:
1. Create a development shell with all required dependencies
2. Set up a Python virtual environment (if not already present)
3. Install all required Python packages
4. Create necessary output directories
5. Provide convenient functions to run the model

After running `nix develop`, you can use the following commands:
- `run_model` - Run the model on a sample dataset (2015 only)
- `run_model full` - Run the model on the complete dataset
- `run_test` - Test predictions using a trained model

### Manual Setup (Alternative)
If you prefer not to use Nix or are on a system without Nix support:

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p output/figures logs
```

## Using Binary Caching for Faster Builds

We use Cachix to speed up Nix builds. To configure your system to use our binary cache:

```bash
# Install Cachix
nix-env -iA cachix -f https://cachix.org/api/v1/install

# Configure the SOFE4620ML9 cache
cachix use sofe4620ml9
```

This will dramatically reduce build times as most dependencies will be downloaded pre-built instead of being compiled locally.

## Data Processing
The data processing pipeline includes:
1. Loading the training data from multiple CSV files
2. Cleaning and preprocessing the data
3. Feature engineering
4. Train-test splitting

## Model Training
We train a machine learning model to predict the optimal fifth player for the home team based on:
- The existing four players on the home team
- The five players on the away team
- Other relevant game features

## Evaluation
The model is evaluated on a held-out test set, with metrics including:
- Accuracy of player prediction
- Impact on team performance

## Usage
Run the prediction pipeline manually:

```bash
python src/main.py
```

For interactive exploration, open the Jupyter notebooks:

```bash
jupyter notebook
```

## Results
Model performance and analysis will be documented in the presentation and final report.

## Project Timeline
- Code Submission: March 17, 11:59 PM
- Presentation Submission: March 19, 11:59 AM
