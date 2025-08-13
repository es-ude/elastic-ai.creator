# HDL Development Environment Module

This module provides comprehensive HDL (Hardware Description Language) development tools for both VHDL and Verilog development within the devenv environment.

## Features

### VHDL Support
- **GHDL**: Open-source VHDL simulator (with Apple Silicon support via Rosetta)
- **VHDL LSP**: Language server for VHDL development
- **Vivado**: Xilinx Vivado design suite (optional)

### Verilog Support
- **Iverilog**: Open-source Verilog simulator
- **SVLS**: SystemVerilog Language Server
- **GTKWave**: Waveform viewer (shared between VHDL and Verilog)

### Additional Tools
- **GTKWave**: Universal waveform viewer
- **zlib**: Required for GHDL/Iverilog VPI support

## Configuration

### Basic Setup

Add the following to your `devenv.yaml`:

```yaml
imports:
  - ./devenv_modules/hdl/
```

### Available Options

```nix
{
  languages.vhdl = {
    enable = false;          # Enable VHDL support
    vivado.enable = false;   # Enable Xilinx Vivado (requires manual setup)
  };

  languages.verilog = {
    enable = false;          # Enable Verilog/SystemVerilog support
  };
}
```

### Example Configuration

In your `devenv.nix` or `devenv.local.nix`:

```nix
{
  # Enable VHDL development with GHDL
  languages.vhdl.enable = true;

  # Enable Verilog development
  languages.verilog.enable = true;

  # Optionally enable Vivado (requires additional setup)
  # languages.vhdl.vivado.enable = true;
}
```

## Vivado Setup (Optional)

### Prerequisites

To use Vivado, you need to add the Vivado build rules to your `devenv.yaml`:

```yaml
inputs:
  vivado:
    url: github:lschuermann/nur-packages
    inputs:
      nixpkgs:
        follows: nixpkgs
```

### Installing Vivado

1. Download the Vivado installer from [Xilinx Downloads](https://www.xilinx.com/member/forms/download/xef.html?filename=Xilinx_Unified_2020.1_0602_1208.tar.gz)
2. Add the installer to your Nix store (see [Nix Cheatsheet](https://nixos.wiki/wiki/Cheatsheet#Adding_files_to_the_store) for large files)
3. Enable Vivado in your configuration: `languages.vhdl.vivado.enable = true;`

## Platform Support

### Apple Silicon (M1/M2/M3)
- GHDL is automatically downloaded and configured for Apple Silicon
- Uses pre-built binaries from the GHDL project
- Rosetta compatibility for x86_64 fallbacks when needed

### Intel/AMD (x86_64)
- Uses system GHDL package from nixpkgs
- Full native support for all tools

## Scripts and Tasks

The module provides several convenience scripts and tasks:

### Scripts
- `ghdl`: GHDL simulator command
- `ghwdump`: GHDL waveform dump utility (Apple Silicon only)

### Tasks
- `ghdl:setup`: Downloads and sets up GHDL for Apple Silicon (runs automatically)

## Customization

### Overriding GHDL Binary

If you need to use a custom GHDL installation, you can override the script in your `devenv.local.nix`:

```nix
{
  scripts = {
    ghdl.exec = ''
      /path/to/your/custom/ghdl "$@"
    '';
  };
}
```

The module uses `lib.mkDefault` for the GHDL script, so your local configuration will automatically take precedence.

## Troubleshooting

### Common Issues

1. **Apple Silicon GHDL not found**
   - The `ghdl:setup` task should run automatically on first shell entry
   - Check that the download completed successfully in `$DEVENV_ROOT/external/`
