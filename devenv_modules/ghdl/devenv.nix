# As long as devenv does not support
# composition of devenv.yaml files you have to
# ensure that your devenv.yaml includes the following
# definition:
#
# vivado:
#   url: github:lschuermann/nur-packages
#   inputs:
#     nixpkgs:
#       follows: nixpkg
#
#
{
  pkgs,
  inputs,
  lib,
  config,
  ...
}: let
  unstablePkgs = import inputs.nixpkgs-unstable {system = pkgs.stdenv.system;};
  rosettaPkgs =
    if unstablePkgs.stdenv.isDarwin && unstablePkgs.stdenv.isAarch64
    then unstablePkgs.pkgsx86_64Darwin
    else unstablePkgs;
in {
  options = {
    languages.vhdl = {
      enable = lib.mkOption {
        type = lib.types.bool;
        default = false;
        description = "set to true to include ghdl";
      };
      vivado.enable = lib.mkOption {
        type = lib.types.bool;
        default = false;
        description = "Install vivado. Important: you have to download the vivado installer yourself and place it in the nix store.";
      };
    };
  };

  config.packages = let
    vivadoPkgs = import inputs.vivado {pkgs = pkgs;};
  in [
    (lib.mkIf config.languages.vhdl.enable rosettaPkgs.ghdl)
    (lib.mkIf config.languages.vhdl.vivado.enable vivadoPkgs.vivado-2020_1)
  ];
}
