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
        apps.default = {
          type = "app";
          program = pkgs.writeShellApplication {
            name = "run-model";
            runtimeInputs = [ pythonEnv ];
            text = ''
              export PYTHONPATH="$PWD/src:$PYTHONPATH"
              ${pythonEnv}/bin/python "$PWD/src/main.py" \
                --data-dir dataset \
                --output-dir output \
                --model-type random_forest \
                "$@"
            '';
          };
        };
        apps.run-full = {
          type = "app";
          program = let
            pkgs = nixpkgs.legacyPackages.${system};
            script = pkgs.writeShellScriptBin "run-full" ''
              export PYTHONPATH="$PWD/src:${pythonEnv}/${pythonEnv.sitePackages}"
              ${pythonEnv}/bin/python "$PWD/src/main.py" \
                --test-data "$PWD/dataset/evaluation/NBA_test.csv" \
                --output "$PWD/output/predictions.csv" \
                --debug
            '';
          in "${script}/bin/run-full";
        };

        apps.test = {
          type = "app";
          program = let
            pkgs = nixpkgs.legacyPackages.${system};
            pythonEnv = pkgs.python312.withPackages(ps: with ps; [
              pandas numpy scikit-learn matplotlib seaborn pyyaml
            ]);
          in "${pkgs.writeShellScriptBin "test" ''
            export PYTHONPATH="$PWD/src:${pythonEnv}/${pythonEnv.sitePackages}:$PYTHONPATH"
            ${pythonEnv}/bin/python "$PWD/src/main.py" \
              --data-dir "$PWD/dataset" \
              --output-dir "$PWD/output" \
              --model-type random_forest \
              --load-model "$PWD/output/random_forest_model" \
              --visualize \
              --years 2015 \
              "$@"
          ''}/bin/test";
        };

        apps.report = {
          type = "app";
          program = "${self}/src/report/generator.py";
        };