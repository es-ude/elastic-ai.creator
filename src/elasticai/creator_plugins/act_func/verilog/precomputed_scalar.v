//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     21.01.2026, 11:21
// Copied on: 	    §{date_copy_created}
// Module Name:     Precomputed Activation Function (SCALAR) used in DNN
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v1
// Processing:      LUT-based processing
// Dependencies:    None
//
// State: 	        Not tested!
// Improvements:    None
// Parameters:      BITWIDTH_IN     --> Bitwidth of input data
//                  BITWIDTH_OUT    --> Bitwidth of output data
//                  NUM_INPUTS      --> Number of in/outputs
//////////////////////////////////////////////////////////////////////////////////


module ACT_PRECOMPUTED_SCALAR #(
    parameter integer BITWIDTH_IN  = 4,
    parameter integer BITWIDTH_OUT = 4,
    parameter integer NUM_INPUTS = 1
)(
    input  wire signed [NUM_INPUTS*BITWIDTH_IN-1:0]  A,
    output reg  signed [NUM_INPUTS*BITWIDTH_OUT-1:0] Q
);

    localparam integer NUM_VALUES = 3;
    localparam signed [(NUM_VALUES-1)*BITWIDTH_IN-1:0] VALUES_X = { 4'sd4, -4'sd6 };
    localparam signed [NUM_VALUES*BITWIDTH_OUT-1:0] VALUES_Y = { 4'sd1, 4'sd0, -4'sd1 };

    integer i, k;
    reg found;
    reg signed [BITWIDTH_IN-1:0] val_i;
    reg signed [BITWIDTH_IN-1:0]  threshold_i;
    reg signed [BITWIDTH_OUT-1:0] value_i;

    always @(*) begin
        for (k = 0; k < NUM_INPUTS; k = k + 1) begin
            found = 1'b0;
            val_i = A[k*BITWIDTH_OUT+:BITWIDTH_OUT];
            Q[k*BITWIDTH_OUT+:BITWIDTH_OUT] = VALUES_Y[(NUM_VALUES*BITWIDTH_OUT-1) -: BITWIDTH_OUT];
            for (i = 0; i < NUM_VALUES-1; i = i + 1) begin
                threshold_i = VALUES_X[i*BITWIDTH_IN +: BITWIDTH_IN];
                value_i     = VALUES_Y[i*BITWIDTH_OUT +: BITWIDTH_OUT];
                if (!found && val_i <= threshold_i) begin
                    Q     = value_i;
                    found = 1'b1;
                end
            end
        end
    end
endmodule