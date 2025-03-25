# Cachix configuration for NBA Lineup Prediction project
#
# This file can be imported in your Nix configuration to automatically
# set up the binary cache for this project
#
# Usage in configuration.nix:
# imports = [ 
#   ./cachix.nix 
#   # ... other imports
# ];
#
# Or use it directly with:
# nix-build -I nixpkgs-overlays="[import ./cachix.nix]"

{ pkgs, ... }:

{
  nix = {
    settings = {
      substituters = [
        "https://sofe4620ml9.cachix.org"
        "https://cache.nixos.org/"
      ];
      trusted-public-keys = [
        "sofe4620ml9.cachix.org-1:tfElwtWLIFUGDl2crSvsyqeORRkDgmAyKI7LcinaTJc=
        "cache.nixos.org-1:6NCHdD59X431o0gWypbMrAURkbJ16ZPMQFGspcDShjY="
      ];
    };
    
    # Enable flakes and new nix command
    extraOptions = ''
      experimental-features = nix-command flakes
    '';
  };
} 