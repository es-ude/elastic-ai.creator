//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     13.11.2020 09:22:46
// Copied on: 	    §{date_copy_created}
// Module Name:     DSP-based Multiplier for Signed Input
// Target Devices:  ASIC, FPGA
// Tool Versions:   1v0
// Description:     DSP-based multiplication with parametrized bitwidth
// Processing:      Direct processing
// Dependencies:    Needs pipeline buffer in upper module
//
// State: 	        Works!
// Dependency:      FPGA has DSP slices
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input data
//
//////////////////////////////////////////////////////////////////////////////////


(* use_dsp = "yes" *)
module MULT_SIGNED#(
    parameter BITWIDTH = 6'd6
)(
    input wire signed [BITWIDTH-'d1:0]      A,
    input wire signed [BITWIDTH-'d1:0]      B,
    output reg signed [2*BITWIDTH-'d1:0]    Q
);
    always@(*)
        Q = A * B;
    
endmodule