{
  description = "NBA Lineup Prediction Project";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      pythonEnv = pkgs.python3.withPackages (ps: with ps; [
        pandas numpy scikit-learn matplotlib seaborn pytest pyyaml
      ]);

      commonArgs = "--data-path ${self}/dataset --years 2015 --output-dir ./output";

    in {
      packages.${system} = {
        default = pythonEnv;
        
        nba-lineup = pkgs.stdenv.mkDerivation {
          name = "nba-lineup";
          src = self;
          buildInputs = [ pythonEnv ];
          installPhase = ''
            mkdir -p $out/bin
            cp -r src $out/
            ln -s ${pythonEnv}/bin/python $out/bin/nba-python
          '';
        };

        nba-lineup-tests = pkgs.stdenv.mkDerivation {
          name = "nba-lineup-tests";
          src = self;
          installPhase = ''
            mkdir -p $out/tests
            if [ -d tests ]; then
              cp -r tests/* $out/tests/
            fi
          '';
        };
      };

      apps.${system} = {
        default = {
          type = "app";
          program = "${pkgs.writeShellScriptBin "nba-lineup" ''
            ${self.packages.${system}.nba-lineup}/bin/nba-python -m src.main \
              ${commonArgs} --no-visualize "$@"
          ''}/bin/nba-lineup";
        };

        run-full = {
          type = "app";
          program = "${pkgs.writeShellScriptBin "nba-lineup-full" ''
            ${self.packages.${system}.nba-lineup}/bin/nba-python -m src.main \
              --data-path ${self}/dataset --years 2007-2015 \
              --output-dir ./full-output "$@"
          ''}/bin/nba-lineup-full";
        };

        test = {
          type = "app";
          program = "${pkgs.writeShellScriptBin "nba-test" ''
            mkdir -p tests
            if [ -d ${self.packages.${system}.nba-lineup-tests}/tests ]; then
              cp -r ${self.packages.${system}.nba-lineup-tests}/tests/* ./tests/
            fi
            ${self.packages.${system}.nba-lineup}/bin/nba-python -m pytest \
              tests/ -vv "$@"
          ''}/bin/nba-test";
        };
      };
    };
}