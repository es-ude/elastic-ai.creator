/////////////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     27.09.2026, 08:58:32
// Copied on: 	    §{date_copy_created}
// Module Name:     Template of a linear / fully connected layer (only inference)
// Target Devices:  FPGA / ASIC
// Tool Versions:   1v0
// Processing:
// Dependencies:    None
//
// State: 	        Not tested!
// Improvements:    None
// Parameters:      BITWIDTH        --> Bitwidth of input data (total bitwidth)
//                  FRAC_WIDTH      --> Bitwidth of the fraction width in input data
/////////////////////////////////////////////////////////////////////////////////////////


module FXP_DOWNCONVERTER #(
    parameter integer BITWIDTH   = 8,
    parameter integer FRAC_WIDTH = 8
)(
    input  wire signed [2*BITWIDTH-1:0] DATA_IN,
    output wire signed [BITWIDTH-1:0]   DATA_OUT
);

    localparam integer UPPER_BITS = BITWIDTH - FRAC_WIDTH + 1;

    wire                  sign     = DATA_IN[2*BITWIDTH-1];
    wire [UPPER_BITS-1:0] upper    = DATA_IN[2*BITWIDTH-1 -: UPPER_BITS];
    wire                  overflow = (|upper) & ~(&upper);

    wire [BITWIDTH-1:0] max_val = {1'b0, {(BITWIDTH-1){1'b1}}};
    wire [BITWIDTH-1:0] min_val = {1'b1, {(BITWIDTH-1){1'b0}}};

    assign DATA_OUT = overflow ? (sign ? min_val : max_val)
                               : DATA_IN[FRAC_WIDTH +: BITWIDTH];

endmodule
