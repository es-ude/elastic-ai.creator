# Example of how to run the simulation
# make TESTBENCH=sigmoid

# vhdl files
FILES = source/*
VHDLEX = .vhd

# testbench
TESTBENCHPATH = testbench/${TESTBENCHFILE}$(VHDLEX)
TESTBENCHFILE = ${TESTBENCH}_tb

#GHDL CONFIG
GHDL_CMD = ghdl
GHDL_FLAGS  = --ieee=synopsys --warn-no-vital-generic

SIMDIR = simulation
STOP_TIME = 20us
# Simulation break condition
#GHDL_SIM_OPT = --assert-level=error
GHDL_SIM_OPT = --stop-time=$(STOP_TIME) --ieee-asserts=disable-at-0

WAVEFORM_VIEWER = gtkwave

.PHONY: clean

all: clean make run view

make:
ifeq ($(strip $(TESTBENCH)),)
	@echo "TESTBENCH not set. Use TESTBENCH=<value> to set it."
	@exit 1
endif

	@mkdir $(SIMDIR)
	@$(GHDL_CMD) -i $(GHDL_FLAGS) --workdir=$(SIMDIR) --work=work $(TESTBENCHPATH) $(FILES)
	@$(GHDL_CMD) -m $(GHDL_FLAGS) --workdir=$(SIMDIR) --work=work $(TESTBENCHFILE)

run:
	@$(GHDL_CMD) -r $(GHDL_FLAGS) --workdir=$(SIMDIR) $(TESTBENCHFILE) --vcd=$(SIMDIR)/$(TESTBENCHFILE).vcd $(GHDL_SIM_OPT)

view:
	@$(WAVEFORM_VIEWER) $(SIMDIR)/$(TESTBENCHFILE).vcd > /dev/null 2>&1 &

clean:
	@rm -rf $(SIMDIR) *.cf