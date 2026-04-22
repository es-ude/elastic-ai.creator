//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     17.03.2025, 10:44
// Copied on: 	    §{date_copy_created}
// Module Name:     MATH_MAX
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v0
// Description:     Math-Max Detection between two signed integer inputs
// Processing:      Logical
// Dependencies:    None
//
// State: 	        Not tested on hardware!
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input data
//////////////////////////////////////////////////////////////////////////////////


module MATH_MAX#(
    parameter BITWIDTH = 5'd16  
)(
    input wire signed [BITWIDTH-'d1:'d0] A,
    input wire signed [BITWIDTH-'d1:'d0] B,
    output wire signed [BITWIDTH-'d1:'d0] Q
);
    wire [BITWIDTH-'d1:'d0] substraction_val;
    assign substraction_val = A - B;

    assign Q = (substraction_val[BITWIDTH-'d1]) ? B : A;

endmodule
