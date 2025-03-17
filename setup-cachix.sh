#!/usr/bin/env bash
#
# NBA Lineup Prediction - Cachix Setup Script
#
# This script helps set up Cachix for the NBA Lineup Prediction project.
# It will:
# 1. Install Cachix if not already installed
# 2. Check if the cache exists, and use it (not create it)
# 3. Update the cachix.nix file with the proper public key
# 4. Push the current build to the cache
#
# Usage: ./setup-cachix.sh [your-auth-token] [cache-name] [public-key]

set -e

echo -e "\e[1;32m🚀 NBA Lineup Prediction - Cachix Setup\e[0m"
echo "========================================"

# Check for auth token
if [ -z "$1" ]; then
  echo -e "\e[1;31m❌ Error: Please provide your Cachix auth token as an argument.\e[0m"
  echo "Usage: ./setup-cachix.sh [your-auth-token] [cache-name] [public-key]"
  echo "You can get your auth token at https://app.cachix.org/personal-auth-tokens"
  exit 1
fi

AUTH_TOKEN="$1"
CACHE_NAME="${2:-sofe4620ml9}"
PROVIDED_PUBLIC_KEY="$3"

# Install Cachix if not found
if ! command -v cachix &> /dev/null; then
  echo -e "\e[1;34m📦 Installing Cachix...\e[0m"
  nix-env -iA cachix -f https://cachix.org/api/v1/install
fi

# Log in to Cachix - fixed to properly pass the auth token
echo -e "\e[1;33m🔑 Logging in to Cachix...\e[0m"
echo "$AUTH_TOKEN" | cachix authtoken --stdin

# Check if the cache exists, use it (don't try to create it)
echo -e "\e[1;34m🔄 Configuring cache: $CACHE_NAME\e[0m"
if cachix use "$CACHE_NAME"; then
  echo -e "\e[1;32m✅ Cache configured successfully.\e[0m"
else
  echo -e "\e[1;31m❌ Error: Failed to configure cache. Please check your Cachix setup.\e[0m"
  exit 1
fi

# Try to get the public key from cachix info command
CACHIX_INFO_PUBLIC_KEY=$(cachix info "$CACHE_NAME" 2>/dev/null | grep "Public Key:" | cut -d: -f2- | xargs)

# Use provided public key if cachix info fails
if [ -n "$CACHIX_INFO_PUBLIC_KEY" ]; then
  PUBLIC_KEY="$CACHIX_INFO_PUBLIC_KEY"
  echo -e "\e[1;32m✅ Retrieved public key from Cachix: $PUBLIC_KEY\e[0m"
elif [ -n "$PROVIDED_PUBLIC_KEY" ]; then
  PUBLIC_KEY="$PROVIDED_PUBLIC_KEY"
  echo -e "\e[1;32m✅ Using provided public key: $PUBLIC_KEY\e[0m"
else
  # Fallback to a known public key for this specific cache if it's the default cache
  if [ "$CACHE_NAME" = "sofe4620ml9" ]; then
    PUBLIC_KEY="sofe4620ml9.cachix.org-1:tfElwtWLIFUGDl2crSvsyqeORRkDgmAyKI7LcinaTJc="
    echo -e "\e[1;32m✅ Using fallback public key for $CACHE_NAME: $PUBLIC_KEY\e[0m"
  else
    echo -e "\e[1;33m⚠️ Warning: Could not determine public key. Please provide it as the third argument.\e[0m"
    echo "Usage: ./setup-cachix.sh [your-auth-token] [cache-name] [public-key]"
    PUBLIC_KEY=""
  fi
fi

# Update the cachix.nix file with the public key if we have one
if [ -n "$PUBLIC_KEY" ]; then
  echo -e "\e[1;34m🔄 Updating cachix.nix with the public key...\e[0m"
  if [ -f "cachix.nix" ]; then
    # Extract the cache name from the public key for the sed pattern
    CACHE_PREFIX=$(echo "$PUBLIC_KEY" | cut -d. -f1)
    sed -i "s|$CACHE_PREFIX\.cachix\.org-1:.*|$PUBLIC_KEY|g" cachix.nix
    echo -e "\e[1;32m✅ cachix.nix updated.\e[0m"
  else
    echo -e "\e[1;33m⚠️ Warning: cachix.nix file not found. Creating it...\e[0m"
    cat > cachix.nix << EOF
{ config, ... }:

{
  nix = {
    settings = {
      substituters = [
        "https://$CACHE_NAME.cachix.org"
        "https://cache.nixos.org/"
      ];
      trusted-public-keys = [
        "$PUBLIC_KEY"
        "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
      ];
    };
  };
}
EOF
    echo -e "\e[1;32m✅ cachix.nix created.\e[0m"
  fi
else
  echo -e "\e[1;33m⚠️ Warning: No public key available. Skipping cachix.nix update.\e[0m"
fi

# Build and push to the cache - using flake outputs instead of default.nix
echo -e "\e[1;34m🏗️ Building and pushing to the cache...\e[0m"

# Make sure jq is available for JSON parsing
if ! command -v jq &> /dev/null; then
  echo -e "\e[1;34m📦 Installing jq for JSON parsing...\e[0m"
  nix-env -iA nixpkgs.jq
fi

# Build the flake outputs
echo -e "\e[1;34mBuilding flake outputs...\e[0m"
nix build .#default --no-link 2>/dev/null || echo -e "\e[1;33m⚠️ Warning: Could not build default output\e[0m"
nix build .#run-full --no-link 2>/dev/null || echo -e "\e[1;33m⚠️ Warning: Could not build run-full output\e[0m"
nix build .#test --no-link 2>/dev/null || echo -e "\e[1;33m⚠️ Warning: Could not build test output\e[0m"

# Get all derivation outputs from the flake
echo -e "\e[1;34mPushing flake outputs to Cachix...\e[0m"
FLAKE_OUTPUTS=$(nix flake show --json 2>/dev/null | jq -r '.packages | to_entries[] | .key + "/" + (.value | to_entries[] | .key)' 2>/dev/null)

if [ -n "$FLAKE_OUTPUTS" ]; then
  for output in $FLAKE_OUTPUTS; do
    echo -e "\e[1;34mPushing $output...\e[0m"
    nix build .#$output --no-link 2>/dev/null && \
    nix path-info --json .#$output 2>/dev/null | jq -r '.[].path' | cachix push "$CACHE_NAME" || \
    echo -e "\e[1;33m⚠️ Warning: Could not push $output\e[0m"
  done
else
  echo -e "\e[1;33m⚠️ Warning: No flake outputs found to push\e[0m"
  echo "Trying to build and push specific system outputs..."
  
  # Try to build and push for the current system
  CURRENT_SYSTEM=$(nix eval --impure --expr 'builtins.currentSystem' 2>/dev/null || echo "x86_64-linux")
  echo -e "\e[1;34mCurrent system: $CURRENT_SYSTEM\e[0m"
  
  for pkg in "default" "run-full" "test"; do
    echo -e "\e[1;34mBuilding and pushing $CURRENT_SYSTEM/$pkg...\e[0m"
    nix build .#packages.$CURRENT_SYSTEM.$pkg --no-link 2>/dev/null && \
    nix path-info --json .#packages.$CURRENT_SYSTEM.$pkg 2>/dev/null | jq -r '.[].path' | cachix push "$CACHE_NAME" || \
    echo -e "\e[1;33m⚠️ Warning: Could not push $CURRENT_SYSTEM/$pkg\e[0m"
  done
fi

# Push all current build results in the Nix store
echo -e "\e[1;34mPushing all relevant paths in Nix store...\e[0m"
nix-store -qR --include-outputs $(nix-instantiate --expr '(import <nixpkgs> {}).hello' 2>/dev/null) 2>/dev/null | \
  cachix push "$CACHE_NAME" || echo -e "\e[1;33m⚠️ Warning: Could not push store paths\e[0m"

echo ""
echo -e "\e[1;32m✅ Cachix setup complete!\e[0m"
echo -e "\e[1;32m✅ Cache name: $CACHE_NAME\e[0m"
if [ -n "$PUBLIC_KEY" ]; then
  echo -e "\e[1;32m✅ Public key: $PUBLIC_KEY\e[0m"
fi
echo ""
echo "Next steps:"
echo "1. Set up GitHub Actions by adding your auth token as a secret named ${CACHE_NAME^^}_CACHIX"
echo "2. Push your changes to GitHub to trigger the CI pipeline"
echo "3. Share the cache name with your users so they can benefit from faster builds"
echo ""
echo "Users can enable the cache with: cachix use $CACHE_NAME"
echo ""