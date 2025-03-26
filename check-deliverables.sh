#!/usr/bin/env bash
set -e

check_file() {
    if [ ! -f "$1" ]; then
        echo "Missing required file: $1"
        exit 1
    fi
}

check_file output/predictions.csv
check_file output/random_forest_model.pkl
check_file output/figures/feature_importance.png

echo "All core deliverables present!"
python -c "import pandas as pd; pd.read_csv('output/predictions.csv')[['Game_ID', 'Home_Team', 'Fifth_Player']].to_csv('output/submission.csv', index=False)"