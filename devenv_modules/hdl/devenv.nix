{
  pkgs,
  inputs,
  lib,
  config,
  ...
}: let
  mac_ghdl = import ./ghdl.nix {inherit pkgs lib;};
in {
  options = {
    languages.verilog = {
      enable = lib.mkOption {
        type = lib.types.bool;
        default = false;
        description = "set to true to add iverilog, gtkwave and svls";
      };
    };
    languages.vhdl = {
      enable = lib.mkOption {
        type = lib.types.bool;
        default = false;
        description = "set to true to include ghdl, gtkwave and vhdl-lsp";
      };
      ghdl.enable = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = "whether to install ghdl, will be downloaded from github for apple silicon";
      };
      vivado.enable = lib.mkOption {
        type = lib.types.bool;
        default = false;
        description = "Install vivado. Important: you have to download the vivado installer yourself and place it in the nix store.";
      };
    };
  };

  config.tasks = mac_ghdl.tasks;
  config.scripts = mac_ghdl.scripts;
  config.packages = let
    vivadoPkgs = import inputs.vivado {pkgs = pkgs;};
    vhdl = config.languages.vhdl.enable;
    verilog = config.languages.verilog.enable;
    ghdl = config.languages.vhdl.ghdl.enable;
    mkIf = lib.mkIf;
    isNotAppleSilicon = !(pkgs.stdenv.isDarwin && pkgs.stdenv.isAarch64);
  in [
    (mkIf config.languages.vhdl.vivado.enable vivadoPkgs.vivado-2020_1)
    (mkIf vhdl pkgs.vhdl-ls)
    (mkIf (vhdl && ghdl && isNotAppleSilicon) pkgs.ghdl)
    (mkIf (vhdl || verilog) pkgs.gtkwave)
    (mkIf (vhdl || verilog) pkgs.zlib) # needed for ghdl/iverilog vpi
    (mkIf verilog pkgs.iverilog)
    (mkIf verilog pkgs.svls)
  ];
}
