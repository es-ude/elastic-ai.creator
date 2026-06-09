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
//                  RAMWIDTH    --> Number of cells to store data
//                  DATAFILE    --> Absolute path to file with initial values (if "" zero)
// Information:     https://www.dsprelated.com/showarticle/1337.php
//////////////////////////////////////////////////////////////////////////////////


module BRAM_SINGLE#(
    parameter BITWIDTH = 12,
    parameter RAMWIDTH = 32,
    parameter DATAFILE = "bram_preload.mem"
)(
    input wire CLK_RAM,
    input wire EN,
    input wire WE,
    input wire [$clog2(RAMWIDTH)-'d1:'d0] ADR,
    input wire [BITWIDTH-'d1:'d0] DIN,
    output reg [BITWIDTH-'d1:'d0] DOUT
);

    (* ram_style = "block" *)
    reg [BITWIDTH-'d1:'d0] bram_block ['d0:RAMWIDTH-'d1];

    initial begin
        if(DATAFILE != "") begin
            $readmemh(DATAFILE, bram_block, 'd0, RAMWIDTH-'d1);
        end
    end
           
    integer i0; 
    always @(posedge CLK_RAM) begin
        if (EN) begin
            if (WE) begin
                bram_block[ADR] = DIN;
            end else begin
                DOUT = bram_block[ADR];
            end
        end
    end
endmodule
