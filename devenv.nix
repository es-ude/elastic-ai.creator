{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: let
  unstablePkgs = import inputs.nixpkgs-unstable {system = pkgs.stdenv.system;};
  asciidoctorKroki = pkgs.buildNpmPackage {
    pname = "asciidoctor-kroki";
    version = "0.17.0";
    src = pkgs.fetchFromGitHub {
      owner = "Mogztter";
      repo = "asciidoctor-kroki";
      rev = "v0.17.0";
      sha256 = "sha256-N1zDTNjIA4jHa+h3mHLQJTJApmbPueAZpv0Jbm2H31o=";
    };
    npmDepsHash = "sha256-DU7zBkACn26Ia4nx5cby0S2weTNE3kMNg080yR1knjw=";
    dontNpmBuild = true;
    PUPPETEER_SKIP_CHROMIUM_DOWNLOAD = "1";
  };
  antoraWithKroki = pkgs.writeShellScriptBin "antora" ''
    export NODE_PATH=${asciidoctorKroki}/lib/node_modules:${pkgs.antora}/lib/node_modules
    exec ${pkgs.antora}/bin/antora "$@"
  '';
in {
  # override these in your devenv.local.nix as needed
  languages.vhdl = {
    enable = lib.mkDefault true;
    vivado.enable = lib.mkDefault false;
  };

  packages = [
    pkgs.kramdown-asciidoc
    pkgs.git
    pkgs.pikchr
    unstablePkgs.jujutsu
    pkgs.gtkwave # visualize wave forms from hw simulations
    pkgs.graphviz
    antoraWithKroki
    unstablePkgs.mypy # python type checker
    unstablePkgs.vale # syntax aware linter for prose
    unstablePkgs.act # run github workflows locally
    pkgs.alejandra # nix formatter
  ];

  languages.c.enable = true;
  languages.nix.enable = true;
  languages.python = {
    enable = true;
    package = pkgs.python311;
    uv.enable = true;
    uv.package = unstablePkgs.uv;
    uv.sync.enable = true;
    uv.sync.allExtras = true;
  };

  scripts = {
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

  tasks = {
    "check:vhdl-plugins" = {
      exec = ''
        run_vhdl_tbs . "run_tbs.py"
      '';
      before = ["check:all"];
    };

    "check:slow-tests" = {
      exec = "${unstablePkgs.uv}/bin/uv run pytest -m 'simulation'";
      before = ["check:all"];
    };

    "check:fast-tests" = {
      exec = ''
        ${unstablePkgs.uv}/bin/uv run coverage run
        ${unstablePkgs.uv}/bin/uv run coverage xml
      '';
      before = ["check:all"];
    };

    "check:types" = {
      exec = "${unstablePkgs.uv}/bin/uv run mypy";
      # before = ["devenv:enterTest"];
    };

    "check:python-lint" = {
      exec = "${unstablePkgs.uv}/bin/uv run ruff check";
      before = ["check:all"];
    };

    "check:commit-lint" = {
      exec = "${unstablePkgs.uv}/bin/uv run cog check";
      before = ["check:all"];
    };

    "check:nix-lint" = {
      exec = "${pkgs.alejandra}/bin/alejandra --exclude ./.devenv.flake.nix -c .";
      before = ["check:all"];
    };

    "check:formatting" = {
      exec = "${unstablePkgs.uv}/bin/uv run ruff format --check";
      before = ["check:all"];
    };

    "package:build" = {
      exec = "${unstablePkgs.uv}/bin/uv build";
      before = ["all:build" "check:all"];
    };

    "package:clean" = {
      exec = "if [ -d dist ]; then rm -r dist; fi";
      before = ["all:clean"];
      after = ["check:all"];
    };

    "docs:build" = let
      out_dir = "docs/modules/api/pages";
      nav_file = "docs/modules/api/partials/nav.adoc";
      pkg_name = "elasticai.creator";
      pysciidoc = "UV_PROJECT_ENVIRONMENT=venv-py311 ${unstablePkgs.uv}/bin/uv run -p 3.11 pysciidoc";
    in {
      exec = ''
        if [ ! -d docs/modules/api ]; then mkdir -p docs/modules/api; fi
        ${pkgs.kramdown-asciidoc}/bin/kramdoc README.md -o docs/modules/ROOT/pages/index.adoc
        ${pkgs.kramdown-asciidoc}/bin/kramdoc CONTRIBUTION.md -o docs/modules/ROOT/pages/contribution.adoc
        ${pysciidoc} --api-output-dir ${out_dir} --nav-file ${nav_file} ${pkg_name}
        ${antoraWithKroki}/bin/antora docs/antora-playbook.yml
      '';
      before = ["all:build" "check:all"];
    };

    "docs:clean" = {
      exec = ''
        if [ -d docs/modules/api/pages ]; then rm -r docs/modules/api/pages; fi
        if [ -e docs/modules/ROOT/pages/index.adoc ]; then rm docs/modules/ROOT/pages/index.adoc; fi
        if [ -e docs/modules/ROOT/pages/contribution.adoc ]; then rm docs/modules/ROOT/pages/contribution.adoc; fi
        if [ -d venv-py311 ]; then rm -r venv-py311; fi
        if [ -d docs/modules/plugins/pages ]; then rm -r docs/modules/plugins/pages; fi
        if [ -d docs/build ]; then rm -r docs/build; fi
      '';
      before = ["all:clean"];
      after = ["check:all"];
    };

    "all:build" = {};
    "all:clean" = {};
    "check:all" = {
      exec = "";
      before = ["devenv:enterTest"];
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

