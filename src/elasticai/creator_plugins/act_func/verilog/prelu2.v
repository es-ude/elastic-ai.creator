//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     22.01.2026, 08:20:44
// Copied on: 	    §{date_copy_created}
// Module Name:     Bitshifted Programmable ReLU-Activation Function
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v0
// Processing:      Bitshifted scaling for negative values with rounding
// Dependencies:    None
//
// State: 	        Works!
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input data
//                  SCALING --> Number of bits for bit-shifting negative values
//////////////////////////////////////////////////////////////////////////////////


module ACT_PRELU2#(
    parameter BITWIDTH = 5'd4
)(
    input wire signed [BITWIDTH-'d1:0] A,
    output wire signed [BITWIDTH-'d1:0] Q
);
    localparam SCALING = 5'd1;

    assign Q = (A[BITWIDTH-'d1]) ? {{(SCALING){1'b1}}, A[(BITWIDTH-'d1)-:(BITWIDTH-SCALING)]} + |A[SCALING-'d1:0] : A;

endmodule
