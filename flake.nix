{
  description = "NBA Lineup Nix Flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        pythonEnv = pkgs.python312.withPackages (ps: with ps; [
          pandas
          numpy
          scikit-learn
          matplotlib
          seaborn
          scipy
          openpyxl
          pyyaml
        ]);
        
        nba-lineup-script = pkgs.writeShellScriptBin "nba-lineup" ''
          export PYTHONPATH="${self.outPath}/src:${pythonEnv}/${pythonEnv.sitePackages}"
          ${pythonEnv}/bin/python -m src.main "$@"
        '';
      in {
        devShells.default = pkgs.mkShell {
          packages = [ 
            pythonEnv 
            nba-lineup-script
          ];
          
          # Add these environment variables
          PYTHONPATH = "${self.outPath}/src:${pythonEnv}/${pythonEnv.sitePackages}";
          PWD = "${self.outPath}";
          
          shellHook = ''
            run_model() {
              nba-lineup --data-dir dataset --output-dir output --model-type random_forest --test-data "''${1:-2015}"
            }
            run_full() {
              nba-lineup --data-dir dataset --output-dir output --model-type random_forest --full-dataset --years 2007-2015
            }
            
            export -f run_model run_full  # Critical for making functions available
            
            echo "Available commands:"
            echo "run_model [YEAR] - Run with sample dataset (default: 2015)"
            echo "run_full         - Run with full dataset (2007-2015)"
          '';
        };

        apps = {
          run-sample = {
            type = "app";
            program = "${nba-lineup-script}/bin/nba-lineup";
          };
          
          run-full = {
            type = "app";
            program = "${nba-lineup-script}/bin/nba-lineup";
          };


          default = self.apps.${system}.run-sample;
        };
        
        packages = {
          default = nba-lineup-script;

          # Fixing the run-full package definition
          run-full = pkgs.stdenv.mkDerivation {
            name = "run-full";
            src = ./.;
            buildInputs = [ pkgs.deterministic-kvm ];  # Add necessary build inputs here
            buildPhase = ''
              echo "Building project..."
              # Add build steps here
            '';
            installPhase = ''
              mkdir -p $out/bin
              cp -r * $out/bin/
            '';
          };
        };
      });
}