import os


def create_makefile(destination_dir, stop_time="4000ns"):
    makefile_content = f"""# automatically generates a Makefile for GHDL

# vhdl files
FILES = $(shell find -type f -name "*.vhd")
VHDLEX = .vhd

# testbench
TESTBENCHFILE = ${{TESTBENCH}}_tb
TESTBENCHPATH = $(shell find -type f -name "${{TESTBENCH}}_tb.vhd")

# GHDL configuration
GHDL_CMD = ghdl
GHDL_FLAGS  = --ieee=synopsys --warn-no-vital-generic

# simulation configuration
SIMDIR = .simulation
STOP_TIME = {stop_time}
# Simulation break condition
# GHDL_SIM_OPT = --assert-level=error
GHDL_SIM_OPT = --stop-time=$(STOP_TIME) --ieee-asserts=disable-at-0

WAVEFORM_VIEWER = gtkwave

.PHONY: clean

all: clean make run #view

make:
ifeq ($(strip $(TESTBENCH)),)
\t\t@echo "TESTBENCH not set. Use TESTBENCH=<value> to set it."
\t\t@exit 1
endif

\t@mkdir $(SIMDIR)
\t@$(GHDL_CMD) -i $(GHDL_FLAGS) --workdir=$(SIMDIR) --work=work $(TESTBENCHPATH) $(FILES) > $(SIMDIR)/make_output.txt
\t@$(GHDL_CMD) -m $(GHDL_FLAGS) --workdir=$(SIMDIR) --work=work $(TESTBENCHFILE) > $(SIMDIR)/make_output.txt

run:
\t@$(GHDL_CMD) -r $(GHDL_FLAGS) --workdir=$(SIMDIR) $(TESTBENCHFILE) --vcd=$(SIMDIR)/$(TESTBENCHFILE).vcd $(GHDL_SIM_OPT) > $(SIMDIR)/make_output.txt

view:
\t@$(WAVEFORM_VIEWER) $(SIMDIR)/$(TESTBENCHFILE).vcd > /dev/null 2>&1 &

clean:
\t@rm -rf $(SIMDIR) *.cf
"""
    destination_path = os.path.join(destination_dir, "makefile")
    with open(destination_path, "w") as file:
        file.write(makefile_content)

    print(
        f"Makefile has been created and saved to {destination_dir} with"
        f" STOP_TIME={stop_time}"
    )
