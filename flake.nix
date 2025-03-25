{
  description = "NBA Lineup Analysis";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.11";
    nixpkgs-unstable.url = "github:nixos/nixpkgs/nixos-unstable";
    # NixOS-WSL for helper functions
    nixos-wsl.url = "github:nix-community/NixOS-WSL";
  };

  outputs = { self, nixpkgs, nixpkgs-unstable, ... }@inputs: 
  let
    allSystems = ["x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin"];
    forAllSystems = nixpkgs.lib.genAttrs allSystems;

    # Add this new overlay to make unstable packages available
    overlayUnstable = _: prev: {
      unstable = import nixpkgs-unstable {
        inherit (prev) system;
        config.allowUnfree = true;
      };
    };

    overlays = {
      unstable = overlayUnstable;
    };
  in {
    inherit overlays;

    # Development shells for different platforms
    devShells = forAllSystems (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            experimental-features = ["nix-command" "flakes"];
          };
        };
      in {
        default = pkgs.mkShell {
          name = "nba-lineup-dev-shell";
          nativeBuildInputs = with pkgs; [
            # Git
            git

            # Python environment
            python312
            python312Packages.pip
            pipenv

            # Development tools
            nodejs_20
            nodePackages.npm

            # Shell utilities
            bashInteractive
            bash-completion
          ];
          shellHook = ''
            echo "NBA Lineup development environment activated!"
            # Set up local environment if needed
            if [ ! -d "./venv" ]; then
              echo "Setting up Python virtual environment..."
              python -m venv venv
            fi
            
            # Use virtual environment
            source venv/bin/activate
            
            # Ensure git is properly configured for VSCode
            git config --local core.editor "code --wait"
            
            # Create symbolic link to git in /bin if it doesn't exist
            if [ ! -e /bin/git ] && [ -e "$(which git)" ]; then
              if [ -w /bin ]; then
                ln -sf "$(which git)" /bin/git
                echo "Created symlink for git in /bin"
              else
                echo "Warning: Cannot create symlink in /bin (need sudo)"
              fi
            fi
          '';
        };
      }
    );
  };
} 