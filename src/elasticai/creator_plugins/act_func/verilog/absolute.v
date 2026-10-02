//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     21.01.2026 20:12:45
// Copied on: 	    §{date_copy_created}
// Module Name:     Activation Function: Absolute
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v1
// Processing:      LUT-based processing
// Dependencies:    None
//
// State: 	        Works!
// Improvements:    None
// Parameters:      BITWIDTH_IN     --> Bitwidth of input data
//                  BITWIDTH_OUT    --> Bitwidth of the output data
//////////////////////////////////////////////////////////////////////////////////


module ACT_ABSOLUTE#(
    parameter integer BITWIDTH_IN = 12,
    parameter integer BITWIDTH_OUT = 12
)(
    input wire signed [BITWIDTH_IN-1:0] A,
    output wire signed [BITWIDTH_OUT-1:0] Q
);

    wire signed [BITWIDTH_IN-1:0] abs_val;
    assign abs_val = (A[BITWIDTH_IN-1]) ? -A : A;
    assign Q = abs_val[(BITWIDTH_IN-1)-:BITWIDTH_OUT];

endmodule
