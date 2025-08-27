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
    unstablePkgs.jujutsu
    pkgs.graphviz
    pkgs.cocogitto
    unstablePkgs.mypy # python type checker
    unstablePkgs.vale # syntax aware linter for prose
    unstablePkgs.act # run github workflows locally
    pkgs.alejandra # nix formatter
    pkgs.zlib # needed as dependency cocotb/ghdl under circumstances
    pkgs.iverilog
  ];

  languages.c.enable = true;
  languages.nix.enable = true;
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
  };

  scripts = {
    serve_docs = {
      exec = "${unstablePkgs.uv}/bin/uv run sphinx-autobuild -j auto docs build/docs/";
    };

    new_docs_for_creator_plugin = {
      exec = ''
        mkdir -p elasticai/creator_plugins/$1/docs/modules/ROOT/pages
        echo "= $1
        " > elasticai/creator_plugins/$1/docs/modules/ROOT/pages/index.adoc
        touch elasticai/creator_plugins/$1/docs/modules/ROOT/nav.adoc
        echo "name: <name>
        version: true
        title: <title>
        nav:
         - modules/ROOT/nav.adoc" > elasticai/creator_plugins/$1/docs/antora.yml
      '';
      package = pkgs.bash;
      description = "create new plugin including docs folder";
    };

    new_meta_for_creator_plugin = {
      exec = ''
        echo "
        [[plugins]]
        name = '$1'
        version = '0.1'
        api_version = 'xx.xx'
        target_runtime = 'runtime'
        target_platform = 'platform'

        " > elasticai/creator_plugins/$1/meta.toml
      '';
      package = pkgs.bash;
      description = "create a new minimal meta.toml file for a plugin";
    };

    new_creator_plugin = {
      exec = ''
        if [ -d elasticai/creator_plugins/$1 ]; then
           mkdir -p elasticai/creator_plugins/$1
           touch elasticai/creator_plugins/$1/__init__.py
           new_meta_for_creator_plugin $1
           new_docs_for_creator_plugin $1
        else
          echo "plugin already exists"
        fi
      '';
      package = pkgs.bash;
      description = "create a new creator plugin incl. meta.toml and docs";
    };

    run_vhdl_tbs = {
      exec = ''
        LAST_EXIT=0
        START_DIR=$1
        FILE_PATTERN=$2
        for tb in $(find $START_DIR -type f -iname $FILE_PATTERN); do
          ${unstablePkgs.uv}/bin/uv run $tb
          tmp_state=$?
          NUM_TESTS=$(($NUM_TESTS + 1))
          if [[ $tmp_state -ne 0 ]] ; then
            NUM_FAILS=$(($NUM_FAILS + 1))
            FAILED_TESTS+=("$tb")
          fi
        done
        if [[ NUM_FAILS -gt 0 ]]; then
          echo ""
          echo "--------------Summary: $(basename $0)----------------------"
          echo "$NUM_FAILS out of $NUM_TESTS failed:"
          for tb in $FAILED_TESTS; do
            echo "  $tb"
          done
          exit 1
        fi
      '';
      package = pkgs.bash;
      description = "search for all testbenches in given directory and  run them using given command";
    };
  };

  tasks = let
    uv_run = "${unstablePkgs.uv}/bin/uv run";
  in {
    "check:vhdl-plugins" = {
      exec = ''
        run_vhdl_tbs . "run_tbs.py"
      '';
      before = ["check:tests"];
    };

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
      exec = "${pkgs.alejandra}/bin/alejandra --exclude ./.devenv.flake.nix -c .";
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

