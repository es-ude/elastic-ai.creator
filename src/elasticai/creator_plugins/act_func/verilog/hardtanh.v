//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     22.01.2026 10:41:45
// Copied on: 	    §{date_copy_created}
// Module Name:     HardTanh-Activation Function for DNN
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v1
// Processing:      LUT-based processing
// Dependencies:    None
//
// State: 	        Works!
// Improvements:    None
// Parameters:      BITWIDTH_IN     --> Bitwidth of input data
//                  BITWIDTH_OUT    --> Bitwidth of output data
//////////////////////////////////////////////////////////////////////////////////


module ACT_HARDTANH#(
    parameter integer BITWIDTH_IN = 4,
    parameter integer BITWIDTH_OUT = 4
)(
    input wire signed [BITWIDTH_IN-1:0] A,
    output wire signed [BITWIDTH_OUT-1:0] Q
);

    localparam signed [BITWIDTH_OUT-1:0] MAX_VAL = 4'sd4;
    localparam signed [BITWIDTH_OUT-1:0] MIN_VAL = -4'sd4;

    assign Q = (A > MAX_VAL) ? MAX_VAL : ((A > MIN_VAL) ? A : MIN_VAL);

endmodule
