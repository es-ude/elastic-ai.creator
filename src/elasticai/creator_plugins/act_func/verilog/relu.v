//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     11.08.2023 07:59:57
// Copied on: 	    §{date_copy_created}
// Module Name:     ReLU-Activation Function for DNN
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v0
// Processing:      LUT-based processing
// Dependencies:    None
//
// State: 	        Works!
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input data
//////////////////////////////////////////////////////////////////////////////////


module ACT_RELU#(
    parameter BITWIDTH = 5'd12
)(
    input wire signed [BITWIDTH-'d1:0] A,
    output wire signed [BITWIDTH-'d1:0] Q
);

    assign Q = (A[BITWIDTH-'d1]) ? 'd0 : A;

endmodule
