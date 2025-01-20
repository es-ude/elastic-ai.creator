set_property -dict { PACKAGE_PIN H11   IOSTANDARD LVCMOS33 } [get_ports { clk_100m }]; # IO_L13P_T2_MRCC_14
create_clock -add -name clk_100 -period 10 -waveform {0 5} [get_ports { clk_100m }];

set_property -dict { PACKAGE_PIN G11   IOSTANDARD LVCMOS33 } [get_ports { clk_32m }]; # 32MHz
create_clock -add -name clk_32 -period 31.25 -waveform {0 15.625} [get_ports { clk_32m }];


set_property -dict { PACKAGE_PIN L12 IOSTANDARD LVCMOS33 } [get_ports { fpga_busy }];


#  SPI
set_property -dict { PACKAGE_PIN N11 IOSTANDARD LVCMOS33 } [get_ports { spi_clk }];
set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets spi_clk]


#set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets spi_clk_IBUF] 

#[Place 30-574] Poor placement for routing between an IO pin and BUFG. If this sub optimal condition is acceptable for this design, you may use the CLOCK_DEDICATED_ROUTE constraint in the .xdc file to demote this message to a WARNING. However, the use of this override is highly discouraged. These examples can be used directly in the .xdc file to override this clock rule.
#	< set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets spi_clk_IBUF] >

#	spi_clk_IBUF_inst (IBUF.O) is locked to IOB_X0Y3
#	 and spi_clk_IBUF_BUFG_inst (BUFG.I) is provisionally placed by clockplacer on BUFGCTRL_X0Y0


set_property -dict { PACKAGE_PIN P12 IOSTANDARD LVCMOS33 } [get_ports { spi_ss_n }];

set_property -dict { PACKAGE_PIN P11 IOSTANDARD LVCMOS33 } [get_ports { spi_mosi }];
set_property -dict { PACKAGE_PIN M12 IOSTANDARD LVCMOS33 } [get_ports { spi_miso }];

set_property -dict { PACKAGE_PIN H12 IOSTANDARD LVCMOS33 } [get_ports { leds[0] }];
set_property -dict { PACKAGE_PIN J12 IOSTANDARD LVCMOS33 } [get_ports { leds[1] }];
set_property -dict { PACKAGE_PIN k12 IOSTANDARD LVCMOS33 } [get_ports { leds[2] }];
set_property -dict { PACKAGE_PIN J11 IOSTANDARD LVCMOS33 } [get_ports { leds[3] }];

## Configuration options, can be used for all designs
set_property BITSTREAM.CONFIG.CONFIGRATE 50 [current_design]
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property CFGBVS VCCO [current_design]
set_property BITSTREAM.GENERAL.COMPRESS TRUE [current_design]
