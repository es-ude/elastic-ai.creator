//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     11.08.2023 07:59:57
// Copied on: 	    §{date_copy_created}
// Module Name:     ReLU-Activation Function for DNN
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v1
// Processing:      LUT-based processing
// Dependencies:    None
//
// State: 	        Works!
// Improvements:    None
// Parameters:      BITWIDTH_IN     --> Bitwidth of input data
//                  BITWIDTH_OUT    --> Bitwidth of output data
//                  NUM_INPUTS      --> Number of in/outputs
//////////////////////////////////////////////////////////////////////////////////


module ACT_RELU#(
    parameter integer BITWIDTH_IN = 12,
    parameter integer BITWIDTH_OUT = 12,
    parameter integer NUM_INPUTS = 1
)(
    input wire signed [NUM_INPUTS*BITWIDTH_IN-1:0] A,
    output wire signed [NUM_INPUTS*BITWIDTH_OUT-1:0] Q
);

    genvar k;
    for(k = 0; k < NUM_INPUTS; k = k + 1) begin
        assign Q = (A[(k+1)*BITWIDTH_IN-1]) ? 'sd0 : A[((k+1)*BITWIDTH_IN-1)-:BITWIDTH_OUT];
    end
endmodule
