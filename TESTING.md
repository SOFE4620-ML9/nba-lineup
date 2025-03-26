# Testing Guide for NBA Lineup Prediction

This guide outlines how to test the Nix flake and Cachix integration for the NBA Lineup Prediction project to ensure everything works as expected.

## Prerequisites

- Nix installed with flakes enabled
- Internet connection
- Approximate testing time: 15-20 minutes

## Test Plan

### 1. Test Direct Running from GitHub

**Objective**: Verify that the project can be run directly from GitHub using `nix run`

```bash
# Configure Cachix for faster downloads
nix run github:cachix/cachix -- use sofe4620ml9

# Run the default command (sample dataset)
nix run github:sofe4620-ml9/nba-lineup

# Check the output directory for results
ls -la output/

# Run the full dataset command
nix run github:sofe4620-ml9/nba-lineup#run-full

# Run the test command
nix run github:sofe4620-ml9/nba-lineup#test
```

**Expected results**:
- Each command should download the repository and dependencies if not already cached
- The model should run and produce output in the temporary directory
- No errors should occur during execution

### 2. Test Local Development with `nix develop`

**Objective**: Verify that the development environment can be set up locally

```bash
# Clone the repository if you haven't already
git clone https://github.com/sofe4620-ml9/nba-lineup
cd nba-lineup

# Enter the development shell
nix develop

# Test the convenience functions
run_model
run_test
run_model full
```

**Expected results**:
- The development shell should load successfully
- The Python virtual environment should be created or activated
- The convenience functions should work and produce results in the output directory

### 3. Test Cachix Integration

**Objective**: Verify that binary caching works correctly and speeds up subsequent builds

```bash
# Clear the local Nix store to force rebuilding/downloading
sudo nix-collect-garbage -d

# First run (should download from cache or build)
time nix run github:sofe4620-ml9/nba-lineup

# Second run (should be faster as dependencies are cached)
time nix run github:sofe4620-ml9/nba-lineup
```

**Expected results**:
- The second run should be significantly faster than the first run
- You should see messages indicating substitutions from the cache

### 4. Test GitHub Actions Workflow (For Maintainers)

**Objective**: Verify that the GitHub Actions workflow correctly builds and caches the project

1. Make a small change to the project (e.g., update a comment in a file)
2. Commit and push to GitHub
3. Go to the GitHub Actions tab and monitor the workflow
4. Check that the build completes successfully and pushes to Cachix

**Expected results**:
- The workflow should complete without errors
- The build outputs should be pushed to the Cachix cache

## Troubleshooting Common Issues

### Binary Cache Not Working

If packages are still being built from source instead of downloaded from the cache:

```bash
# Check Nix configuration
nix show-config | grep substituters

# Force using substitutes
nix run github:sofe4620-ml9/nba-lineup --option substitute true

# Try using the cache explicitly
cachix use sofe4620ml9
```

### Flake Not Found

If you get "flake not found" errors:

```bash
# Make sure flakes are enabled
mkdir -p ~/.config/nix
echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf

# Try specifying the branch explicitly
nix run github:sofe4620-ml9/nba-lineup/main
```

### GitHub Actions Failures

If the GitHub Actions workflow fails:

1. Check that the `SOFE4620ML9_CACHIX` secret is correctly set in the repository settings
2. Verify the Cachix authentication token has write permissions
3. Check the logs for specific error messages

## Result Validation

After running the model, you should validate that:

1. The `output` directory contains the expected files:
   - Model file (e.g., `random_forest_model.pkl`)
   - Prediction results (e.g., `predictions.csv`)
   - Visualization plots (in the `figures` subdirectory)

2. The predictions have the expected format with the following columns:
   - Game_ID
   - Home_Team
   - Fifth_Player 