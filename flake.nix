{
  inputs = {
    nixpkgs = {
      url = "github:nixos/nixpkgs/nixos-unstable";
    };
    flake-utils = {
      url = "github:numtide/flake-utils";
    };
    pypi-deps-db = {
      url = "github:DavHau/pypi-deps-db/690fd7b0505ca6c962f5c4b8336d76215de7bb43";
      flake = false;
    };
    mach-nix = {
      url = github:DavHau/mach-nix;
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
      inputs.pypi-deps-db.follows = "pypi-deps-db";
    };
  };

  outputs = { self, nixpkgs, flake-utils, mach-nix, pypi-deps-db }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs { 
          inherit system;
          config.allowUnfree = true;
        };
        mach-nix-utils = import mach-nix {
          inherit pkgs;
        };

        my_python = mach-nix-utils.mkPython {
          requirements = (builtins.readFile ./requirements.txt) + ''
            ipython
            mypy
          '';
        };
        gpu_libs = with pkgs; [
          cudatoolkit_11
          cudnn_cudatoolkit_11
        ];
      in
      {
        devShell = pkgs.mkShell {
          name = "manolo_deepbrain_torch";
          buildInputs = with pkgs; [
            my_python
          ] ++ gpu_libs;

          nativeBuildInputs = with pkgs; [
            wrapGAppsHook
          ];
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath ( gpu_libs ++ [ "/run/opengl-driver/" ]) ;
        };
      });
}
