# Setting Up Cachix for the NBA Lineup Prediction Project

This guide explains how to set up and use [Cachix](https://cachix.org/) to speed up build times for the NBA Lineup Prediction project.

## What is Cachix?

Cachix is a binary cache service for Nix. It allows you to:
- Cache built packages (binaries) so they don't have to be built from source
- Share these cached packages with other users
- Dramatically reduce build times when using Nix

## User Setup (For Project Users)

### 1. Set Up Cachix

If you're using this project, follow these steps for faster builds:

```bash
# Install Cachix (one-time setup)
nix-env -iA cachix -f https://cachix.org/api/v1/install

# Enable the SOFE4620ML9 cache (one-time setup)
cachix use sofe4620ml9
```

### 2. Alternative: Import the cachix.nix Configuration

If you're using NixOS or have a `configuration.nix` file, you can import the provided `cachix.nix` file directly:

```nix
# In your configuration.nix
imports = [ 
  ./cachix.nix 
  # ... other imports
];
```

Then rebuild your configuration:

```bash
sudo nixos-rebuild switch  # For NixOS
# or
home-manager switch        # For Home Manager
```

### 3. Using with nix run

The binary cache is used automatically when running:

```bash
# Run the model with cached binaries
nix run github:sofe4620-ml9/nba-lineup
```

## Project-Specific Setup (For Project Maintainers)

### 1. Create a Cachix Cache

```bash
# Install Cachix
nix-env -iA cachix -f https://cachix.org/api/v1/install

# Login to Cachix
cachix authtoken <your-auth-token>

# Create the cache
cachix create sofe4620ml9
```

### 2. Set Up GitHub Actions for Automatic Caching

The repository already includes a `.github/workflows/build-and-cache.yml` file that automatically:
- Builds the project for multiple platforms (Linux and macOS)
- Pushes the build results to Cachix
- Ensures new changes are cached automatically

To make this work:
1. Go to your repository settings on GitHub
2. Navigate to "Secrets and variables" > "Actions"
3. Add a new secret with name `SOFE4620ML9_CACHIX` and your Cachix auth token as value

### 3. Update the Public Key

After creating your Cachix cache, you'll receive a public key. Update this key in:
- `cachix.nix` file (replace `sofe4620ml9.cachix.org-1:tfElwtWLIFUGDl2crSvsyqeORRkDgmAyKI7LcinaTJc=` with your actual public key if different)

## Benefits of Using Cachix

1. **Much faster installation**: Avoid compiling Python, data science libraries, and other dependencies from source
2. **Consistent builds**: Everyone gets the exact same binaries
3. **Cross-platform support**: Works on Linux and macOS

## Troubleshooting

If packages are still being built from source:

1. Verify your Cachix configuration:
   ```bash
   nix show-config | grep substituters
   ```

2. Try forcing a substitution check:
   ```bash
   nix build github:sofe4620-ml9/nba-lineup --option substitute true
   ```

3. Clear your local build results to force using the cache:
   ```bash
   nix-store --gc
   ``` 