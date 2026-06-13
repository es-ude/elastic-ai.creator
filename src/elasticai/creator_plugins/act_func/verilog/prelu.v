//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     22.01.2026, 08:20:44
// Copied on: 	    §{date_copy_created}
// Module Name:     Programmable ReLU-Activation Function
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v0
// Processing:      Multiplication for negative values with rounding
// Dependencies:    None
//
// State: 	        Works!
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input data
//                  SCALING --> FxP-scaling value for negative inputs
//////////////////////////////////////////////////////////////////////////////////


module ACT_PRELU#(
    parameter BITWIDTH = 5'd4
)(
    input wire signed [BITWIDTH-'d1:0] A,
    output reg signed [BITWIDTH-'d1:0] Q
);
    localparam FRACWIDTH = 5'd2;
    localparam signed SCALING = 5'd2;

    reg signed [2*BITWIDTH-'d1:0] step;
    always@(*) begin
        if(A[BITWIDTH-'d1]) begin
            step = (A * SCALING);
            Q = step[FRACWIDTH+:BITWIDTH] + |step[0+:FRACWIDTH];
        end else begin
            step = 'd0;
            Q = A;
        end
    end
endmodule
