{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

let
  unstablePkgs = import inputs.nixpkgs-unstable { system = pkgs.stdenv.system; };
  pyp = pkgs.python311Packages;
in
{

  packages = [
    pkgs.git
    pkgs.doxygen
    pkgs.diffedit3
    pkgs.asciidoctor
    pkgs.gtkwave  # visualize wave forms from hw simulations
    pkgs.python312
    pkgs.python310
    pkgs.hwatch  # continually rerun a command and print its output
    pkgs.jujutsu  # next generation vcs for mental sanity
    pkgs.diffedit3  # 3-pane merge editor
    pkgs.act  # test github workflows locally
    unstablePkgs.mypy  # python type checker
    unstablePkgs.ruff  # linter/formatter for python
    unstablePkgs.vale  # syntax aware linter for prose
    unstablePkgs.act  # run github workflows locally
  ]; 
  languages.c.enable = true;
  languages.python = {
    enable = true;
    package = pkgs.python311;
    poetry.enable = true;
    poetry.activate.enable = false;
    uv.enable = false;
    venv.enable = false;
    
    uv.package = unstablePkgs.uv;
    uv.sync.enable = false;
    uv.sync.allExtras = false;

  };

  ## Commented out while we're configuring pre-commit manually
  # pre-commit.hooks = {
  #   shellcheck.enable = true;
  #   ripsecrets.enable = true; # don't commit secrets
  #   ruff.enable = true; # lint and automatically fix simple problems/reformat
  #   taplo.enable = true; # reformat toml
  #   nixfmt-rfc-style.enable = true; # reformat nix
  #   ruff-format.enable = true;
  #   mypy = {
  #     enable = false;
  #   }; # check type annotations
  #   end-of-file-fixer.enable = true;
  #   commitizen.enable = true; # help adhering to commit style guidelines
  #   check-toml.enable = true; # check toml syntax
  #   check-case-conflicts.enable = true;
  #   check-added-large-files.enable = true;
  # };

}
