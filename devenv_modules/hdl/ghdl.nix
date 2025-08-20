{
  pkgs,
  lib,
  ...
}: let
  attrOnlyForM1 = attr: lib.optionalAttrs (pkgs.stdenv.isDarwin && pkgs.stdenv.isAarch64) attr;
  ghdl_link = {
    version,
    archive,
    nightly,
  }: "https://github.com/ghdl/ghdl/releases/download/${
    if nightly
    then "nightly/"
    else "v${version}"
  }/${archive}";
  create_ghdl_config = args @ {
    version,
    flavor,
    architecture,
    os,
    checksum,
    nightly ? false,
  }: rec {
    version = args.version;
    flavor = args.flavor; # jit as others fail due to linkage on m1
    architecture = args.architecture;
    operating_sys = args.os;
    name = "ghdl-${flavor}-${version}-${operating_sys}-${architecture}";
    archive = "${name}.tar.gz";
    link = ghdl_link {
      version = version;
      archive = archive;
      nightly = nightly;
    };
    checksum = args.checksum;
  };
  ghdl_configs = [
    (create_ghdl_config {
      version = "5.1.1";
      flavor = "llvm-jit";
      architecture = "aarch64";
      os = "macos15";
      checksum = "f31ebdd210d0f685f7743aafb7026eb31b3e0753c8fe1e2e6205de4bcb74a5c9";
    })

    (create_ghdl_config {
      version = "6.0.0-dev";
      nightly = true;
      flavor = "llvm";
      architecture = "aarch64";
      os = "macos15";
      checksum = "04b230271eae1d6cc6d8fccfaa2d4aae3bb75294553513c8a9e209e19cd421f9";
    })

    (create_ghdl_config {
      version = "5.1.1";
      flavor = "llvm";
      architecture = "aarch64";
      os = "macos14";
      checksum = "e858a3ee3cee22c976354ee7a66ab6377beec1a9383462f0ba583a8df73f46c1";
    })

    (create_ghdl_config {
      version = "5.1.1";
      flavor = "mcode";
      architecture = "x86_64"; # go for rosetta
      os = "macos13";
      checksum = "b386189fd6bcbaa8b7f215afcdadc619f9a2aecf273acbff86621564e2e70c76";
    })

    (create_ghdl_config {
      version = "5.1.1";
      flavor = "llvm";
      architecture = "x86_64"; # go for rosetta
      os = "macos13";
      checksum = "4c35a9d6028d11cbfc7b2a98e0cdb28a56e268e5f586d7246904fec40bc8193a";
    })

    (create_ghdl_config {
      version = "5.1.1";
      flavor = "llvm";
      architecture = "aarch64";
      os = "macos15";
      checksum = "0445652a460d01ab94d3a95fa88f398ae9550973daf5873f66c2480d2c73e209";
    })
  ];
  active_ghdl_config = builtins.elemAt ghdl_configs 5;
in {
  scripts =
    {
    }
    // attrOnlyForM1 (let
      bin_path = "$DEVENV_ROOT/external/${active_ghdl_config.name}/bin";
    in {
      ghdl.exec = lib.mkDefault ''${bin_path}/ghdl "$@"'';
      ghwdump.exec = ''${bin_path}/ghwdump "$@"'';
    });

  tasks =
    {}
    // attrOnlyForM1 {
      "ghdl:setup" = {
        exec = let
          cfg = active_ghdl_config;
        in ''
          if [ ! -d $DEVENV_ROOT/external ]; then
            mkdir $DEVENV_ROOT/external
          fi
          if [ ! -d external/${cfg.name} ]; then
            echo "downloading ghdl"
            cd $DEVENV_ROOT/external/
            ${pkgs.wget}/bin/wget ${cfg.link}
            sha256 -c ${cfg.checksum} ${cfg.archive}
            if [[ ! $? == 0 ]]; then
              echo "could not verify checksum"
              exit 1
            fi
            tar -xzf ${cfg.archive}
          fi
        '';
        before = ["devenv:enterShell"];
      };
    };
}
