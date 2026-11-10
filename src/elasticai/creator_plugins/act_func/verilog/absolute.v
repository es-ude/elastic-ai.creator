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
//                  NUM_INPUTS      --> Number of in/outputs
//////////////////////////////////////////////////////////////////////////////////


module ACT_ABSOLUTE#(
    parameter integer BITWIDTH_IN = 12,
    parameter integer BITWIDTH_OUT = 12,
    parameter integer NUM_INPUTS = 1
)(
    input wire signed [NUM_INPUTS*BITWIDTH_IN-1:0] A,
    output wire signed [NUM_INPUTS*BITWIDTH_OUT-1:0] Q
);

    genvar k;
    for(k = 0; k < NUM_INPUTS; k = k + 1) begin
        wire signed [BITWIDTH_IN-1:0] abs_val;
        assign abs_val = (A[BITWIDTH_IN-1]) ? -A : A;
        assign Q[k*BITWIDTH_OUT+:BITWIDTH_OUT] = abs_val[(BITWIDTH_IN-1)-:BITWIDTH_OUT];
    end
endmodule
