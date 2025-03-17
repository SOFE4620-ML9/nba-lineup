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
    
    # Helper function to create Python environment with dependencies
    makePythonEnv = pkgs: pkgs.python312.withPackages (ps: with ps; [
      # Data science packages
      pandas
      numpy
      scikit-learn
      matplotlib
      seaborn
      jupyter
      scipy
      openpyxl
      
      # Development tools
      pip
      virtualenv
      pytest
    ]);
    
    # Script to prepare the environment
    makeSetupScript = pkgs: ''
      # Create output directories needed for the project
      mkdir -p output/figures
      mkdir -p logs
      
      # Make sure we have a Python virtual environment
      if [ ! -d "./venv" ]; then
        echo "Setting up Python virtual environment..."
        ${pkgs.python312}/bin/python -m venv venv
        source venv/bin/activate
        
        # Install requirements if requirements.txt exists
        if [ -f "requirements.txt" ]; then
          echo "Installing dependencies from requirements.txt..."
          pip install -r requirements.txt
        fi
      else
        # Just activate the existing environment
        source venv/bin/activate
      fi
    '';
    
    # Create a runner script for the model
    makeModelRunner = {pkgs, runFull ? false}: pkgs.writeShellScriptBin "run-nba-model" ''
      set -e
      cd $(${pkgs.coreutils}/bin/mktemp -d)
      
      # Clone repo if we're not already in it
      if [ ! -f "flake.nix" ]; then
        echo "📥 Cloning repository..."
        ${pkgs.git}/bin/git clone https://github.com/sofe4620-ml9/nba-lineup .
      fi
      
      # Prepare environment
      ${makeSetupScript pkgs}
      
      # Run the model
      echo "🏀 Running NBA lineup prediction model..."
      ${if runFull then ''
        python src/main.py --data-dir dataset --output-dir output --model-type random_forest --save-model --visualize
      '' else ''
        python src/main.py --data-dir dataset --output-dir output --model-type random_forest --save-model --visualize --years 2015
      ''}
      
      echo "✅ Model run complete! Results are in the output directory."
    '';
    
    # Create a test runner script
    makeTestRunner = pkgs: pkgs.writeShellScriptBin "run-nba-test" ''
      set -e
      cd $(${pkgs.coreutils}/bin/mktemp -d)
      
      # Clone repo if we're not already in it
      if [ ! -f "flake.nix" ]; then
        echo "📥 Cloning repository..."
        ${pkgs.git}/bin/git clone https://github.com/sofe4620-ml9/nba-lineup .
      fi
      
      # Prepare environment
      ${makeSetupScript pkgs}
      
      # Run the test
      echo "🧪 Running NBA lineup prediction test..."
      python src/main.py --data-dir dataset --output-dir output --model-type random_forest --load-model output/random_forest_model.pkl --predict-only --visualize
      
      echo "✅ Test run complete! Results are in the output directory."
    '';
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
        
        pythonEnv = makePythonEnv pkgs;
      in {
        default = pkgs.mkShell {
          name = "nba-lineup-dev-shell";
          nativeBuildInputs = with pkgs; [
            # Git
            git

            # Python environment
            pythonEnv

            # Development tools
            nodejs_20
            nodePackages.npm

            # Shell utilities
            bashInteractive
            bash-completion
          ];
          
          # Add metadata for the shell
          meta = {
            description = "NBA Lineup Prediction Model";
            homepage = "https://github.com/sofe4620-ml9/nbalineup";
            license = "MIT";
            maintainers = [
              {
                name = "RyzeNGrind";
                email = "git@ryzengrind.xyz";
                github = "RyzeNGrind";
              }
              {
                name = "Jathavan Anton Geetharaj";
                email = "jathavan.antongeetharaj@ontariotechu.net";
                github = "Jathavan01";
              }
            ];
          };
          
          # Auto-install hooks
          shellHook = ''
            # Banner message
            echo "=========================================================="
            echo "🏀 NBA Lineup Prediction - Development Environment"
            echo "=========================================================="
            
            # Create output directories needed for the project
            mkdir -p output/figures
            mkdir -p logs
            
            # Make sure we have a Python virtual environment
            if [ ! -d "./venv" ]; then
              echo "Setting up Python virtual environment..."
              ${pkgs.python312}/bin/python -m venv venv
              source venv/bin/activate
              
              # Install requirements if requirements.txt exists
              if [ -f "requirements.txt" ]; then
                echo "Installing dependencies from requirements.txt..."
                pip install -r requirements.txt
              fi
            else
              # Just activate the existing environment
              source venv/bin/activate
            fi
            
            # Ensure git is properly configured for VSCode
            git config --local core.editor "code --wait" 2>/dev/null || true
            
            # Create run script executor function for convenience
            run_model() {
              local mode="$1"
              echo "Running the model..."
              
              if [ "$mode" == "full" ]; then
                # Run on full dataset
                python src/main.py --data-dir dataset --output-dir output --model-type random_forest --save-model --visualize
              else
                # Run on sample data (2015 only)
                python src/main.py --data-dir dataset --output-dir output --model-type random_forest --save-model --visualize --years 2015
              fi
            }
            
            # Define function to run tests
            run_test() {
              python src/main.py --data-dir dataset --output-dir output --model-type random_forest --load-model output/random_forest_model.pkl --predict-only --visualize
            }
            
            # Export the functions so they're available in the shell
            export -f run_model
            export -f run_test
            
            # Instructions for the user
            echo ""
            echo "🚀 Environment ready! You can now:"
            echo "  • run_model        - Run the model on a small dataset (2015 only)"
            echo "  • run_model full   - Run the model on the full dataset"
            echo "  • run_test         - Test predictions using a trained model"
            echo ""
            echo "📊 All outputs will be saved to the 'output' directory"
            echo "📝 Logs will be saved to the 'logs' directory"
            echo ""
          '';
        };
      }
    );
    
    # Add runnable apps that can be used with 'nix run'
    apps = forAllSystems (system: let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in {
      # Run on sample dataset (2015 only)
      default = {
        type = "app";
        program = "${makeModelRunner {pkgs = pkgs; runFull = false;}}/bin/run-nba-model";
      };
      
      # Run on full dataset
      run-full = {
        type = "app";
        program = "${makeModelRunner {pkgs = pkgs; runFull = true;}}/bin/run-nba-model";
      };
      
      # Run tests
      test = {
        type = "app";
        program = "${makeTestRunner pkgs}/bin/run-nba-test";
      };
    });
    
    # Add packages for caching
    packages = forAllSystems (system: let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      
      # Create derivations for each package
      defaultPkg = makeModelRunner {pkgs = pkgs; runFull = false;};
      fullPkg = makeModelRunner {pkgs = pkgs; runFull = true;};
      testPkg = makeTestRunner pkgs;
    in {
      default = defaultPkg;
      run-full = fullPkg;
      test = testPkg;
      
      # Add explicit defaultPackage for compatibility with older Nix versions
      defaultPackage = defaultPkg;
    });
    
    # Add explicit defaultPackage attribute for compatibility
    defaultPackage = forAllSystems (system: 
      self.packages.${system}.default
    );
  };
}