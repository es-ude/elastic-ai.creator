//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     21.01.2026 20:12:45
// Copied on: 	    §{date_copy_created}
// Module Name:     Activation Function: Absolute
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v0
// Processing:      LUT-based processing
// Dependencies:    None
//
// State: 	        Works!
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input data
//////////////////////////////////////////////////////////////////////////////////


module ACT_ABSOLUTE#(
    parameter integer BITWIDTH = 12
)(
    input wire signed [BITWIDTH-1:0] A,
    output wire signed [BITWIDTH-1:0] Q
);

    assign Q = (A[BITWIDTH-1]) ? (~A) + 'sd1 : A;

endmodule
