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
        # Create proper app definitions
        nba-lineup-script = pkgs.writeShellScriptBin "nba-lineup" ''
          # Use absolute path to project source from Nix store
          export PYTHONPATH="${self.outPath}/src:${pythonEnv}/${pythonEnv.sitePackages}"
          ${pythonEnv}/bin/python -m src.main "$@"
        '';
      in {
        # Correct dev shell configuration
        devShells.default = pkgs.mkShell {
          packages = [ pythonEnv ];
        };

        # Correct app structure with proper typing
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
        
        packages.default = nba-lineup-script;
      });
}