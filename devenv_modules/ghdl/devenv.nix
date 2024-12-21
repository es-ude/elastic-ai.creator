{pkgs, inputs, ... }:

let
  unstablePkgs = import inputs.nixpkgs-unstable { system = pkgs.stdenv.system; };
  rosettaPkgs =
    if unstablePkgs.stdenv.isDarwin && unstablePkgs.stdenv.isAarch64 then 
      unstablePkgs.pkgsx86_64Darwin
    else 
      unstablePkgs;
in
{
  packages = [
    rosettaPkgs.ghdl
  ];
}
