//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     22.01.2026, 08:20:44
// Copied on: 	    §{date_copy_created}
// Module Name:     Programmable ReLU-Activation Function
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v1
// Processing:      Multiplication for negative values with rounding
// Dependencies:    None
//
// State: 	        Works!
// Improvements:    None
// Parameters:      BITWIDTH_IN  --> Bitwidth of input data
//                  BITWIDTH_OUT --> Bitwidth of output data
//                  SCALING --> Number of bits for bit-shifting negative values
//////////////////////////////////////////////////////////////////////////////////


module ACT_PRELU#(
    parameter integer BITWIDTH_IN = 4,
    parameter integer BITWIDTH_OUT = 4
)(
    input wire signed [BITWIDTH_IN-1:0] A,
    output reg signed [BITWIDTH_OUT-1:0] Q
);
    localparam integer FRACWIDTH = 2;
    localparam integer signed SCALING = 2;

    reg signed [2*BITWIDTH_IN-1:0] step;
    always@(*) begin
        if(A[BITWIDTH_IN-1]) begin
            step = (A * SCALING);
            Q = step[FRACWIDTH+:BITWIDTH_OUT];
        end else begin
            step = 'sd0;
            Q = A;
        end
    end
endmodule
