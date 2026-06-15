//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     17.02.2023 23:08:27
// Copied on: 	    §{date_copy_created}
// Module Name:     Template for building a single-port BlockRAM on device
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v0
// Processing:      Data applied on posedge CLK_RAM
// Dependencies:    None
//
// State: 	        Works!
// Improvements:    None
// Parameters:      BITWIDTH    --> Bitwidth of input data
//                  ROMWIDTH    --> Number of cells to store data
//                  DATAFILE    --> Absolute path to file with initial values (if "" zero)
// Information:     https://www.dsprelated.com/showarticle/1337.php
//////////////////////////////////////////////////////////////////////////////////


module LUTROM #(
    parameter BITWIDTH = 8,
    parameter ROMWIDTH = 8
)(
    input wire CLK_SYS,
    input wire EN,
    input wire [$clog2(ROMWIDTH)-'d1:'d0] ADR,
    output wire [BITWIDTH-'d1:'d0] DOUT
);
    localparam [BITWIDTH * ROMWIDTH - 'd1:0] LUTROM_DATA = {8'd7, 8'd6, 8'd5, 8'd4, 8'd3, 8'd2, 8'd1, 8'd0};


    assign DOUT = (EN) ? LUTROM_DATA[ADR*BITWIDTH+:BITWIDTH] : 'd0;

endmodule
