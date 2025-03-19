{
  description = "NBA Lineup Prediction Project";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      pythonEnv = pkgs.python311.withPackages (ps: [
        ps.pandas
        ps.numpy
        ps.scikit-learn
        ps.matplotlib
        ps.joblib
        ps.tqdm
        ps.pyyaml
        ps.seaborn
        ps.pytest  # Add pytest here
      ]);
      
      # Fix dataset path to match project structure
      commonArgs = "--data-path ./dataset";  # Changed from ${self}/dataset
      
    in {
      apps.${system} = {
        default = {
          type = "app";
          program = "${pkgs.writeShellScriptBin "nba-lineup" ''
            #!${pkgs.runtimeShell}
            export PYTHONPATH=${self}/src:${self}:${pythonEnv}/${pythonEnv.sitePackages}:$PYTHONPATH
            cd ${self}
            ${pythonEnv}/bin/python -m src.main ${commonArgs} "$@"
          ''}/bin/nba-lineup";
        };

        run-full = {
          type = "app";
          program = "${pkgs.writeShellScriptBin "nba-lineup-full" ''
            #!${pkgs.runtimeShell}
            export PYTHONPATH=${self}/src:${self}:${pythonEnv}/${pythonEnv.sitePackages}:$PYTHONPATH
            cd ${self}
            ${pythonEnv}/bin/python -m src.main ${commonArgs} --years 2007-2015 "$@"
          ''}/bin/nba-lineup-full";
        };

        test = {
          type = "app";
          program = "${pkgs.writeShellScriptBin "nba-lineup-test" ''
            #!${pkgs.runtimeShell}
            export PYTHONPATH=${self}/src:${self}:${pythonEnv}/${pythonEnv.sitePackages}:$PYTHONPATH
            cd ${self}
            ${pythonEnv}/bin/python -m pytest ${self}/tests
          ''}/bin/nba-lineup-test";
        };
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = [ 
          pythonEnv
          pkgs.pytest  # Add pytest to dev shell
        ];
        shellHook = ''
          export PYTHONPATH=$PWD/src:$PYTHONPATH
          echo "NBA Lineup Prediction development shell"
          echo "Python packages available: pandas, numpy, scikit-learn, matplotlib, pytest"
        '';
      };
    };
}