//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     13.11.2020 09:11:26
// Copied on: 	    §{date_copy_created}
// Module Name:     Bitwise Adder module (Half) with Carry-Out
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v0
// Processing:      Logical Design
//
// State: 	        Works!
// Dependencies:    None
// Improvements:    None
// Parameters:      None
//
//////////////////////////////////////////////////////////////////////////////////


module ADDER_LUT_HALF(
    input wire A,
    input wire B,
    output wire Cout,
    output wire Q
);
    assign Cout = A & B;
    assign Q = A ^ B;

endmodule
