//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     10.11.2026 16:52:11
// Copied on: 	    §{date_copy_created}
// Module Name:     HardSigmoid-Activation Function for DNN
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v0
// Processing:      LUT-based processing
// Dependencies:    None
//
// State: 	        Works!
// Improvements:    None
// Parameters:      BITWIDTH_IN     --> Bitwidth of input data
//                  BITWIDTH_OUT    --> Bitwidth of output data
//                  NUM_INPUTS      --> Number of in/outputs
//////////////////////////////////////////////////////////////////////////////////


module ACT_HARDSIGMOID#(
    parameter integer BITWIDTH_IN = 8,
    parameter integer BITWIDTH_OUT = 8,
    parameter integer NUM_INPUTS = 1
)(
    input wire signed [NUM_INPUTS*BITWIDTH_IN-1:0] A,
    output wire signed [NUM_INPUTS*BITWIDTH_OUT-1:0] Q
);

    localparam signed [BITWIDTH_IN-1:0] XMAX_VAL = 8'sd32;
    localparam signed [BITWIDTH_IN-1:0] XMIN_VAL = -8'sd32;
    localparam signed [BITWIDTH_OUT-1:0] YMAX_VAL = 8'sd16;
    localparam signed [BITWIDTH_OUT-1:0] YMIN_VAL = 8'sd0;
    localparam signed [BITWIDTH_OUT-1:0] BIAS = 8'sd8;
    localparam integer SHIFT = 2;

    genvar k;
    for(k = 0; k < NUM_INPUTS; k = k + 1) begin
        wire signed [BITWIDTH_IN-1:0] val;
        assign val = A[k*BITWIDTH_IN+:BITWIDTH_IN];
        assign Q[k*BITWIDTH_OUT+:BITWIDTH_OUT] = (val > XMAX_VAL) ? YMAX_VAL : ((val >= XMIN_VAL) ? (BIAS + (val >>> SHIFT)) : YMIN_VAL);
    end
endmodule
