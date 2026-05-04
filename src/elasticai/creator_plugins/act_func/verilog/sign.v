//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     11.08.2023 07:59:57
// Copied on: 	    §{date_copy_created}
// Module Name:     Sign-Activation Function for DNN
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v0
// Processing:      LUT-based processing
// Dependencies:    None
//
// State: 	        Works!
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input data
//////////////////////////////////////////////////////////////////////////////////


module ACT_SIGN#(
    parameter BITWIDTH = 5'd4
)(
    input wire signed [BITWIDTH-'d1:0] A,
    output wire signed [BITWIDTH-'d1:0] Q
);

    localparam signed [2*BITWIDTH-'d1:0] YRANGE = {-4'sd4, 4'sd4};
    assign Q = (A[BITWIDTH-'d1]) ? YRANGE[BITWIDTH+:BITWIDTH] : YRANGE['d0+:BITWIDTH];

endmodule
