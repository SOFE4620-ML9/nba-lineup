{
  inputs = {
    Snixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
    #nixpkgs.url = "github:NixOS/nixpkgs/205fd4226592cc83fd4c0885a3e4c9c400efabb5"; # Pinned to working version
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        projectDir = ./.;
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
          ];
          shellHook = ''
            export PYTHONPATH="${projectDir}/src:$PYTHONPATH"
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
          program = let
            pkgs = nixpkgs.legacyPackages.${system};
            script = pkgs.writeShellScriptBin "run-model" ''
              #!/usr/bin/env bash
              export PYTHONPATH="${projectDir}/src:$PYTHONPATH"
              ${pkgs.python312}/bin/python ${projectDir}/src/main.py \
                --data-dir dataset \
                --output-dir output \
                --model-type random_forest \
                "$@"
            '';
          in "${script}/bin/run-model";
        };
        apps.run-full = {
          type = "app";
          program = let 
            pkgs = nixpkgs.legacyPackages.${system};
            pythonEnv = pkgs.python312.withPackages(ps: with ps; [
              pandas numpy scikit-learn matplotlib seaborn jupyter scipy openpyxl
            ]);
            script = pkgs.writeShellScriptBin "run-full" ''
              export PYTHONPATH="${projectDir}/src:$PYTHONPATH"
              ${pythonEnv}/bin/python ${projectDir}/src/main.py \
                --data-dir dataset \
                --output-dir output \
                --model-type random_forest \
                "$@"
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
            export PYTHONPATH="${projectDir}/src:${pythonEnv}/${pythonEnv.sitePackages}:$PYTHONPATH"
            ${pythonEnv}/bin/python ${projectDir}/src/main.py \
              --data-dir dataset \
              --output-dir output \
              --model-type random_forest \
              --load-model output/random_forest_model \
              --visualize \
              --years 2015 \
              "$@"
          ''}/bin/test";
        };

        apps.report = {
          type = "app";
          program = "${projectDir}/src/report/generator.py";
        };
      });
}