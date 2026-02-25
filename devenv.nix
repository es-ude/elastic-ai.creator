{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: let
  unstablePkgs = import inputs.nixpkgs-unstable {system = pkgs.stdenv.system;};
in {
  # override these in your devenv.local.nix as needed
  languages.vhdl = {
    enable = lib.mkDefault true;
  };
  languages.verilog.enable = true;

  packages = [
    pkgs.git-cliff
    pkgs.pikchr
    pkgs.jujutsu
    pkgs.graphviz
    pkgs.cocogitto
    pkgs.alejandra # nix formatter
    pkgs.zlib # needed as dependency cocotb/ghdl under circumstances
    pkgs.iverilog
    pkgs.plantuml
    pkgs.plantuml-server
    pkgs.jetty_11
  ];

  languages.c.enable = true;
  languages.nix.enable = true;
  cachix.pull = ["nixpkgs-python"];
  languages.python = {
    enable = true;
    package = pkgs.python312;
    uv.enable = true;
    uv.package = unstablePkgs.uv;
    uv.sync.enable = true;
    uv.sync.allExtras = true;
  };
  processes = {
    serve_docs.exec = "serve_docs";
    plantuml_server.exec = "plantuml_server";
  };

  env = {
    JETTY_HOME = "${pkgs.jetty_11}";
    PLANTUML_SERVER_HOME = "${pkgs.plantuml-server}";
  };

  scripts = {
    plantuml_server.exec = ''
      java \
        -jar ${pkgs.jetty_11}/start.jar \
          --module=deploy,http,jsp \
          jetty.home=${pkgs.jetty_11} \
          jetty.base=${pkgs.plantuml-server} \
          jetty.http.host="127.0.0.1" \
          jetty.http.port="8081"
    '';
    serve_docs = {
      exec = "${unstablePkgs.uv}/bin/uv run sphinx-autobuild -j auto docs build/docs/";
    };
  };

  tasks = let
    uv_run = "${unstablePkgs.uv}/bin/uv run";
  in {
    "check:slow-tests" = {
      exec = "${uv_run} pytest -m '(simulation or slow) and not hardware'";
      before = ["check:tests"];
    };

    "check:fast-tests" = {
      exec = ''
        ${uv_run} coverage run
        ${uv_run} coverage xml
      '';
      before = ["check:tests"];
    };

    "check:types" = {
      exec = "${uv_run} mypy -p elasticai.creator";
      before = ["check:code-lint"];
    };

    "check:python-lint" = {
      exec = "${uv_run} ruff check";
      before = ["check:code-lint"];
    };

    "check:commit-lint" = {
      exec = ''
        if $CI; then
          ${pkgs.cocogitto}/bin/cog check ..$GITHUB_SOURCE_REF
        else
          ${pkgs.cocogitto}/bin/cog check --from-latest-tag --ignore-merge-commits
        fi
      '';
    };

    "check:nix-lint" = {
      exec = "${pkgs.alejandra}/bin/alejandra --exclude ./.devenv --exclude ./.devenv.flake.nix -c .";
      before = ["check:code-lint"];
    };

    "check:formatting" = {
      exec = "${uv_run} ruff format --check";
      before = ["check:code-lint"];
    };

    "check:architecture" = {
      exec = "${uv_run} tach check";
      before = ["check:code-lint"];
    };

    "package:build" = {
      exec = "${unstablePkgs.uv}/bin/uv build";
    };

    "docs:single-page" = {
      exec = ''
        export LC_ALL=C  # necessary to run in github action
        ${uv_run} sphinx-build -b singlehtml docs build/docs
      '';
    };

    "docs:build" = {
      exec = ''
        export LC_ALL=C  # necessary to run in github action
        ${uv_run} sphinx-build -j auto -b html docs build/docs
        touch build/docs/.nojekyll  # prevent github from trying to build the docs
      '';
    };

    "docs:clean" = {
      exec = ''
        rm -rf build/docs/*
      '';
    };

    "check:code-lint" = {
    };

    "check:tests" = {
    };
  };
}
