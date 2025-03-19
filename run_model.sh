#!/usr/bin/env bash

# Set up debugging
set -x  # Print each command before executing
set -e  # Exit immediately if a command exits with a non-zero status

# Add pre-run validation
python -c "from src.data.validation import validate_features; validate_features()"

# Create logs directory
mkdir -p logs

# Create output directories
mkdir -p output/figures

# Train the model on a small sample first using Nix environment
echo "Training model on a small dataset (2015 only)..."
MPLBACKEND=Agg nix-shell -p "python312.withPackages(ps: with ps; [ pandas numpy scikit-learn matplotlib seaborn jupyter scipy openpyxl ])" --run "python src/main.py --data-dir dataset --output-dir output --model-type random_forest --save-model --visualize --years 2015" 2>&1 | tee logs/training_run_$(date +%Y%m%d_%H%M%S).log

# Run prediction using the trained model on just the test data
echo "Testing prediction on test data..."
MPLBACKEND=Agg nix-shell -p "python312.withPackages(ps: with ps; [ pandas numpy scikit-learn matplotlib seaborn jupyter scipy openpyxl ])" --run "python src/main.py --data-dir dataset --output-dir output --model-type random_forest --load-model output/random_forest_model.pkl --predict-only --visualize" 2>&1 | tee logs/prediction_run_$(date +%Y%m%d_%H%M%S).log

# Post-run checks
if [ -f "output/predictions.csv" ]; then
    echo "Validation: Predictions generated successfully"
    python src/report/validation.py --input output/predictions.csv
else
    echo "Error: Predictions file not found!" >&2
    exit 1
fi

# Print the unique values from the predictions.csv file
echo "Unique players in predictions.csv:"
nix-shell -p "python312.withPackages(ps: with ps; [ pandas ])" --run "python -c \"import pandas as pd; df = pd.read_csv('output/predictions.csv'); print(df['Fifth_Player'].value_counts())\""

# If all looks good, train on all years and generate final predictions
echo "Would you like to train the model on the full dataset? (y/n)"
read -p "> " response

if [[ "$response" == "y" || "$response" == "Y" ]]; then
    # Train the model on the full dataset and generate predictions
    echo "Training model on the full dataset..."
    MPLBACKEND=Agg nix-shell -p "python312.withPackages(ps: with ps; [ pandas numpy scikit-learn matplotlib seaborn jupyter scipy openpyxl ])" --run "python src/main.py --data-dir dataset --output-dir output --model-type random_forest --save-model --visualize" 2>&1 | tee logs/full_training_run_$(date +%Y%m%d_%H%M%S).log
    
    # Also train a gradient boosting model for comparison (optional)
    echo "Would you like to also train a gradient boosting model for comparison? (y/n)"
    read -p "> " gb_response
    
    if [[ "$gb_response" == "y" || "$gb_response" == "Y" ]]; then
        echo "Training gradient boosting model for comparison..."
        MPLBACKEND=Agg nix-shell -p "python312.withPackages(ps: with ps; [ pandas numpy scikit-learn matplotlib seaborn jupyter scipy openpyxl ])" --run "python src/main.py --data-dir dataset --output-dir output --model-type gradient_boosting --save-model --visualize" 2>&1 | tee logs/gradient_boosting_run_$(date +%Y%m%d_%H%M%S).log
    fi
    
    # Print completion message
    echo "Model training and prediction complete!"
    echo "Results saved to the 'output' directory."
else
    echo "Skipping full dataset training. You can run this script again later to train on the full dataset."
fi 