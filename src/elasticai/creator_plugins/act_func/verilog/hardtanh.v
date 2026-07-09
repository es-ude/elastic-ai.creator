//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     22.01.2026 10:41:45
// Copied on: 	    §{date_copy_created}
// Module Name:     HardTanh-Activation Function for DNN
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v0
// Processing:      LUT-based processing
// Dependencies:    None
//
// State: 	        Works!
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input data
//////////////////////////////////////////////////////////////////////////////////


module ACT_HARDTANH#(
    parameter BITWIDTH = 5'd4
)(
    input wire signed [BITWIDTH-'d1:0] A,
    output wire signed [BITWIDTH-'d1:0] Q
);

    localparam signed [BITWIDTH-'d1:0] MAX_VAL = 4'sd4;
    localparam signed [BITWIDTH-'d1:0] MIN_VAL = -4'sd4;

    assign Q = (A > MAX_VAL) ? MAX_VAL : ((A > MIN_VAL) ? A : MIN_VAL);

endmodule
