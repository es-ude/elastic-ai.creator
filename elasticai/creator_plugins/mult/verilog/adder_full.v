//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     13.11.2020 09:11:26
// Copied on: 	    §{date_copy_created}
// Module Name:     Bitwise Adder module (Full) with Carry-In/Out
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v0
// Processing:      Logical Design
//
// State: 	        Works!
// Dependencies:    ADDER_HALF
// Improvements:    None
// Parameters:      None
//
//////////////////////////////////////////////////////////////////////////////////


module ADDER_LUT_FULL(
    input wire A,
    input wire B,
    input wire Cin,
    output wire Cout,
    output wire Q
);
    wire C0, C1, Q0;
    assign Cout = C0 | C1;

    ADDER_LUT_HALF ADD0(A, B, C0, Q0);
    ADDER_LUT_HALF ADD1(Cin, Q0, C1, Q);

endmodule
