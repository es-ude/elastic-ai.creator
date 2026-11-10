//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     22.01.2026, 08:20:44
// Copied on: 	    §{date_copy_created}
// Module Name:     Bitshifted Programmable ReLU-Activation Function
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v1
// Processing:      Bitshifted scaling for negative values with rounding
// Dependencies:    None
//
// State: 	        Works!
// Improvements:    None
// Parameters:      BITWIDTH_IN     --> Bitwidth of input data
//                  BITWIDTH_OUT    --> Bitwidth of output data
//                  SCALING --> Number of bits for bit-shifting negative values
//                  NUM_INPUTS      --> Number of in/outputs
//////////////////////////////////////////////////////////////////////////////////


module ACT_PRELU2#(
    parameter integer BITWIDTH_IN = 4,
    parameter integer BITWIDTH_OUT = 4,
    parameter integer NUM_INPUTS = 1
)(
    input wire signed [NUM_INPUTS*BITWIDTH_IN-1:0] A,
    output wire signed [NUM_INPUTS*BITWIDTH_OUT-1:0] Q
);
    localparam integer SCALING = 1;

    genvar k;
    for(k = 0; k < NUM_INPUTS; k = k + 1) begin
        wire signed [BITWIDTH_IN-1:0] val;
        assign val = A[k*BITWIDTH_IN+:BITWIDTH_IN];
        assign Q[k*BITWIDTH_OUT+:BITWIDTH_OUT] = (val[BITWIDTH_IN-1]) ? {{(SCALING){1'b1}}, val[(BITWIDTH_IN-1)-:(BITWIDTH_IN-SCALING)]}: val;
    end
endmodule
