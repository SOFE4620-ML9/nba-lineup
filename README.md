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
│   ├── eval              # Test data and labels
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

### Using Nix Development Shell (Recommended)
> **Note:** If you don't have Nix installed, you can install it using the Determinate Nix Installer: `curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | \
  sh -s -- install`

```bash
# Activate the development environment
nix develop

# Activate Python virtual environment
source venv/bin/activate
```

### Using Python Virtual Environment
Alternative setup without Nix:

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

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
Run the prediction pipeline:

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
