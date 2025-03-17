{
  description = "NBA Lineup Prediction Project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
  };

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
      ]);
    in {
      packages.${system}.default = pythonEnv;

      apps.${system}.default = {
        type = "app";
        program = toString (pkgs.writeShellScriptBin "nba-lineup" ''
          #!${pkgs.runtimeShell}
          ${pythonEnv}/bin/python ${self}/src/main.py \
            --data-path ${self}/data/raw/nba_games.csv "$@"
        '');
      };

      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [ pythonEnv ];
        shellHook = ''
          echo "NBA Lineup Prediction development shell"
          echo "Python packages available: pandas, numpy, scikit-learn, matplotlib"
        '';
      };
    };
}