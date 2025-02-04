{pkgs, lib, ...} :

let
  buildInputs = with pkgs; [
    cudaPackages.cuda_cudart
    cudaPackages.cudatoolkit
    cudaPackages.cudnn
    stdenv.cc.cc
    libuv
    zlib
  ];
in 
{
  env.UV_PYTHON_PREFERENCE = lib.mkForce "only-system";
  packages = with pkgs; [
    cudaPackages.cuda_nvcc
  ];

  languages.vhdl.enable = true;
  languages.vhdl.vivado.enable = true;

  env = {
    LD_LIBRARY_PATH = "${
      with pkgs;
      lib.makeLibraryPath buildInputs
    }:/run/opengl-driver/lib:/run/opengl-driver-32/lib";
    XLA_FLAGS = "--xla_gpu_cuda_data_dir=${pkgs.cudaPackages.cudatoolkit}"; # For tensorflow with GPU support
    CUDA_PATH = pkgs.cudaPackages.cudatoolkit;
  };

}
