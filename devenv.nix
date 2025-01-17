{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

let
  unstablePkgs = import inputs.nixpkgs-unstable { system = pkgs.stdenv.system; };
  pyp = pkgs.python310Packages;
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
   in
 
{

  packages = [
    pkgs.kramdown-asciidoc
    pkgs.gtkwave  # visualize wave forms from hw simulations
    antoraWithKroki
    pkgs.antora  # documentation generator
    pkgs.xunit-viewer
    unstablePkgs.mypy  # python type checker
    unstablePkgs.ruff  # linter/formatter for python
    unstablePkgs.vale  # syntax aware linter for prose
    unstablePkgs.act  # run github workflows locally
  ];
  languages.c.enable = true;
  languages.python = {
    enable = true;
    package = pkgs.python311;
    uv.enable = true;
    uv.package = unstablePkgs.uv;
    uv.sync.enable = false;
    uv.sync.allExtras = false;

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
  };

  tasks = {
    "test:vhdl_plugins" = {
      exec = ''echo "running testbenches for discovered plugins"
        UV_PROJECT_ENVIRONEMNT=venv-py311 ${unstablePkgs.uv}/bin/uv run -p 3.11 eai-run-ghdl-tbs-for-plugins
        '';
      before = [ "devenv:enterTest" ];
    };

    "build:package" = {
      exec = "uv build";
      before = ["build:all"];
    };

    "clean:package" = {
      exec = "if [ -d dist ]; then rm -r dist; fi";
      before = ["clean:all"];
    };

     "build:docs" = let
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
      before = ["build:all"];
    };

    "clean:docs" = {
      exec = ''
        if [ -d docs/modules/api/pages ]; then rm -r docs/modules/api/pages; fi
        if [ -e docs/modules/ROOT/pages/index.adoc ]; then rm docs/modules/ROOT/pages/index.adoc; fi
        if [ -e docs/modules/ROOT/pages/contribution.adoc ]; then rm docs/modules/ROOT/pages/contribution.adoc; fi
        if [ -d venv-py311 ]; then rm -r venv-py311; fi
        if [ -d docs/modules/plugins/pages ]; then rm -r docs/modules/plugins/pages; fi
        if [ -d docs/build ]; then rm -r docs/build; fi
        '';
      before = ["clean:all"];
    };


    "build:all" = {};
    "clean:all" = {};
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
