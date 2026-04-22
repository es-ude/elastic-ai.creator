#!/bin/bash

## analyze files in an order that makes sure dependencies are analyzed before their depending units
## this could be done, e.g. by using post-order traversal in case we have a dependency tree

mkdir -p ghdl
ghdl -a --workdir=ghdl -fsynopsys build/lstm_network/linear/*_rom.vhd
ghdl -a --workdir=ghdl -fsynopsys build/lstm_network/*_rom_*.vhd
ghdl -a --workdir=ghdl -fsynopsys build/lstm_network/*_ram_*.vhd
ghdl -a --workdir=ghdl -fsynopsys build/*/linear/fp_linear_0.vhd

ghdl -a --workdir=ghdl -fsynopsys build/lstm_network/*cell_hard*.vhd
ghdl -a --workdir=ghdl -fsynopsys build/lstm_network/lstm_cell.vhd
ghdl -a --workdir=ghdl -fsynopsys build/lstm_network/lstm_network.vhd
ghdl -a --workdir=ghdl -fsynopsys build/lstm_network_tb.vhd

ghdl -r --workdir=ghdl/ -fsynopsys lstm_network_tb
