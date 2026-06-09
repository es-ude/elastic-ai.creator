//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     17.02.2023 23:08:27
// Copied on: 	    §{date_copy_created}
// Module Name:     Template for building a dual-port B(lock)RAM on device
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v0
// Processing:      Data applied on posedge CLK_RAM, handling to BRAM on negedge CLK_RAM
// Dependencies:    DATA_FILE is optional
//
// State: 	        Works!
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input data
//                  RAMWIDTH --> Number of cells to store data
//                  DATA_FILE --> String with path to file for initial writing (if "" zero)
// Information:     https://www.dsprelated.com/showarticle/1337.php
//////////////////////////////////////////////////////////////////////////////////


module BRAM_DUAL#(
    parameter BITWIDTH = 12,
    parameter RAMWIDTH = 32,
    parameter DATAFILE = "bram_preload.mem"
)(
    input wire CLK_RAM,
    input wire EN,
    input wire WE_A, WE_B,
    input wire [$clog2(RAMWIDTH)-'d1:'d0] ADR_A, ADR_B,
    input wire [BITWIDTH-'d1:'d0] DIN_A, DIN_B,
    output reg [BITWIDTH-'d1:'d0] DOUT_A, DOUT_B
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
        if(EN) begin
            if(WE_A) begin
                bram_block[ADR_A] <= DIN_A;
            end else begin
                DOUT_A <= bram_block[ADR_A];
            end
            if(WE_B) begin
                bram_block[ADR_B] <= DIN_B;
            end else begin
                DOUT_B <= bram_block[ADR_B];
            end
        end
    end
endmodule
