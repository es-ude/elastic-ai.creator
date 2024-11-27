{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

let
  unstablePkgs = import inputs.nixpkgs-unstable { system = pkgs.stdenv.system; };
  rosettaPkgs =
    if unstablePkgs.stdenv.isDarwin && unstablePkgs.stdenv.isAarch64 then
      unstablePkgs.pkgsx86_64Darwin
    else
      unstablePkgs;
  pyp = unstablePkgs.python312Packages;
in
{

  packages = [
    pkgs.git
    rosettaPkgs.ghdl
    pkgs.gtkwave
    pyp.python-lsp-server
    pyp.pylsp-mypy
    pyp.pylsp-rope
    unstablePkgs.mypy
    unstablePkgs.ruff

  ];

  languages.python = {
    enable = true;
    package = unstablePkgs.python312;
    uv.enable = true;
    uv.package = unstablePkgs.uv;
    uv.sync.enable = true;
    uv.sync.allExtras = true;
  };

  tasks = {
    "bash:link_uv_lock" = {
        exec = "ln -s uvPython3.12.lock uv.lock";
        before = ["devenv:enterShell"];
        status = "[ -L uv.lock ]";
    };
  };

  pre-commit.hooks = {
    shellcheck.enable = true;
    ripsecrets.enable = true; # don't commit secrets
    ruff.enable = true; # lint and automatically fix simple problems/reformat
    taplo.enable = true; # reformat toml
    nixfmt-rfc-style.enable = true; # reformat nix
    ruff-format.enable = true;
    mypy = {
      enable = false;
    }; # check type annotations
    end-of-file-fixer.enable = true;
    commitizen.enable = true; # help adhering to commit style guidelines
    check-toml.enable = true; # check toml syntax
    check-case-conflicts.enable = true;
    check-added-large-files.enable = true;
  };

  enterTest =
    let
      runCommandWithPython =
        cmd:
        (
          version:
          "UV_PROJECT_ENVIRONMENT=.venv_${version} uv run --no-env-file --isolated --noprogress --python ${version} ${cmd}"
        );
      runPytest = runCommandWithPython "pytest";
    in
    ''
      rm uv.lock
      if [ -f
      ln -s uvPython3.11.lock uv.lock
      ${runPytest "python3.11"}
      rm uv.lock
      ln -s uvPython3.12.lock uv.lock
      uv run --python python3.12 pytest
    '';
}
