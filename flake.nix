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

      testScript = pkgs.writeShellScriptBin "nba-test" ''
        # Create temporary test directory
        TEST_DIR=$(mktemp -d)
        cp -r ${self.packages.${system}.nba-lineup-tests}/tests/. "$TEST_DIR"
        chmod -R u+w "$TEST_DIR"
        
        # Run pytest in isolated environment
        ${self.packages.${system}.nba-lineup}/bin/nba-python -m pytest \
          "$TEST_DIR" -vv "$@"
        
        # Cleanup
        rm -rf "$TEST_DIR"
      '';

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
          src = ./.;
          installPhase = ''
            mkdir -p $out/tests
            touch $out/tests/__init__.py
            if [ -d tests ]; then
              cp -r tests/* $out/tests/
            fi
          '';
        };

        report-generator = pkgs.stdenv.mkDerivation {
          name = "nba-report-generator";
          buildInputs = [ self.packages.${system}.pythonEnv ];
          src = ./.;
          installPhase = ''
            mkdir -p $out/bin
            echo "${pkgs.python3}/bin/python src/report/generator.py \$@" > $out/bin/generate-report
            chmod +x $out/bin/generate-report
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
          program = "${testScript}/bin/nba-test";
        };

        report = utils.lib.mkApp {
          drv = self.packages.${system}.report-generator;
          exePath = "/bin/generate-report";
        };
      };
    };
}