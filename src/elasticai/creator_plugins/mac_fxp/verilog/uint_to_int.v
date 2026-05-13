//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date: 	16.01.2024 15:57:24
// Copied on: 	    §{date_copy_created}
// Module Name:     Number converter unsigned to signed integer
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v0
// Description:     Number transformation from unsigned to signed
// Dependencies:    None
//
// State:		    Works!
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input data
//////////////////////////////////////////////////////////////////////////////////


module UINT_TO_INT#(
    parameter BITWIDTH = 12
)(
    input wire [BITWIDTH-'d1:0] A,
    output wire signed [BITWIDTH-'d1:0] Q
);

    assign Q = {1'd0 ^ A[BITWIDTH-'d1], A[BITWIDTH-'d2:0]};
endmodule
