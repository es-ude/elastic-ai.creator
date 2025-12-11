{
  pkgs,
  inputs,
  lib,
  config,
  ...
}: {
  options = {
    torch = {
      backend = lib.mkOption {
        type = lib.types.str;
        default = "default";
        description = "Specify the PyTorch backend to use (default, cpu, cu128, cu116, cu115, cu113, cu111, cu102, rocm). 'auto' uses the default pypi provided backend.";
      };
    };
  };

  # Use config.torch.backend to set env default, allow overrides via devenv.local.nix
  config = {
    env = let
      index =
        if config.torch.backend == "default"
        then ""
        else " default=https://pypi.org/simple pytorch=https://download.pytorch.org/whl/${config.torch.backend}";
    in {
      UV_INDEX = "${index}";
    };
  };
}
