{
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
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python312
            python312Packages.pandas
            python312Packages.numpy
            python312Packages.scikit-learn
            python312Packages.matplotlib
            python312Packages.seaborn
            python312Packages.jupyter
            python312Packages.scipy
            python312Packages.openpyxl
            python312Packages.pyyaml
          ];
          shellHook = ''
            export PYTHONPATH="${self}/src:$PYTHONPATH"
            echo "NBA Lineup Prediction dev shell activated"
          '';
        };

        packages.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python312
            python312Packages.pandas
            python312Packages.numpy
            python312Packages.scikit-learn
            python312Packages.matplotlib
            python312Packages.seaborn
            python312Packages.pyyaml
          ];
        };
        apps = {
          x86_64-linux = {
            default = pkgs.writeShellApplication {
              name = "nba-lineup";
              runtimeInputs = [ pythonEnv ];
              text = ''
                export PYTHONPATH="${self}/src:${pythonEnv}/${pythonEnv.sitePackages}:$PYTHONPATH"
                ${pythonEnv}/bin/python -m src.main \
                  --data-path "${self}/dataset" \
                  --output-dir "${self}/output" \
                  --model-type random_forest \
                  "$@"
              '';
            };

            run-full = pkgs.writeShellApplication {
              name = "run-full";
              runtimeInputs = [ pythonEnv ];
              text = ''
                export PYTHONPATH="${self}/src:${pythonEnv}/${pythonEnv.sitePackages}:$PYTHONPATH"
                ${pythonEnv}/bin/python -m src.main \
                  --full \
                  --data-path "${self}/dataset" \
                  --output-dir "${self}/output" \
                  --model-type random_forest
              '';
            };
          };
        };
      });
}        