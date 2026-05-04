//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     22.01.2026, 08:20:44
// Copied on: 	    §{date_copy_created}
// Module Name:     Programmable ReLU-Activation Function for DNN
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v0
// Processing:      LUT-based processing
// Dependencies:    None
//
// State: 	        Works!
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input data
//                  SCALING --> Number of bits for bit-shifting negative values
//////////////////////////////////////////////////////////////////////////////////


module ACT_PRELU#(
    parameter BITWIDTH = 5'd12,
    parameter SCALING = 5'd8
)(
    input wire signed [BITWIDTH-'d1:0] A,
    output wire signed [BITWIDTH-'d1:0] Q
);


    assign Q = (A[BITWIDTH-'d1]) ? {{(SCALING){1'b1}}, A[(BITWIDTH-'d1)-:(BITWIDTH-SCALING)]} : A;

endmodule
