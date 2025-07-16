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

  # choose 24.11 release for ghdl as later releases are broken due to gnat-{13, 14}
  # not building for x86_64-darwin
  # see https://github.com/NixOS/nixpkgs/issues/385174
  pkgsForGhdl = import inputs.nixpkgs-24-11 {system = pkgs.stdenv.system;};
  rosettaPkgs =
    if pkgsForGhdl.stdenv.isDarwin && pkgsForGhdl.stdenv.isAarch64
    then pkgsForGhdl.pkgsx86_64Darwin
    else pkgsForGhdl;
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
