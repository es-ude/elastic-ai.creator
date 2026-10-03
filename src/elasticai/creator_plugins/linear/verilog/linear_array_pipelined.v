//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     24.09.2026, 16:12:34
// Copied on: 	    §{date_copy_created}
// Module Name:     Template of a linear / fully connected layer (only inference, pipelined)
// Target Devices:  FPGA
// Tool Versions:   1v0
// Processing:
// Dependencies:    MAC operator, multipliers (DSP, LUT), linear_array
//
// State: 	        Not tested!
// Improvements:    None
// Parameters:      BITWIDTH        --> Bitwidth of input data
//                  SIZE_INPUT      --> Number of input values
//                  SIZE_OUTPUT     --> Number of output values (=number of neurons)
//                  NUM_MULT        --> Number of multiplier units used in the MAC
//                  BITS_SCALE_BIAS --> Bits for left-shifting the input bias value (to apply FxP)
//                  BITS_SCALE_DOUT --> Bits for right-shifting the output value (to apply FxP)
//////////////////////////////////////////////////////////////////////////////////


module LINEAR_ARRAY_PIPELINED#(
    parameter integer BITWIDTH = 8,
    parameter integer SIZE_INPUT = 4,
    parameter integer SIZE_OUTPUT = 3,
    parameter integer NUM_MULT = 1,
    parameter integer BITS_SCALE_BIAS = 0,
    parameter integer BITS_SCALE_DOUT = 0
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire DO_CALC,
    output wire MOD_RDY,
    input wire signed [SIZE_INPUT* BITWIDTH-1:0] DATA_IN,
    output reg signed [SIZE_OUTPUT* BITWIDTH-1:0] DATA_OUT,
    output wire DATA_VALID
);

    wire [SIZE_OUTPUT* BITWIDTH-1:0] linear_dout;
    wire linear_done;
    reg linear_done_dly;
    assign DATA_VALID = linear_done && linear_done_dly;

    LINEAR_ARRAY#(
        .BITWIDTH(BITWIDTH),
        .SIZE_INPUT(SIZE_INPUT),
        .SIZE_OUTPUT(SIZE_OUTPUT),
        .NUM_MULT(NUM_MULT),
        .BITS_SCALE_BIAS(BITS_SCALE_BIAS),
        .BITS_SCALE_DOUT(BITS_SCALE_DOUT)
    ) LINEAR (
        .CLK_SYS(CLK_SYS),
        .RSTN(RSTN),
        .EN(EN),
        .DO_CALC(DO_CALC),
        .MOD_RDY(MOD_RDY),
        .DATA_IN(DATA_IN),
        .DATA_OUT(linear_dout),
        .DATA_VALID(linear_done)
    );

    always@(posedge CLK_SYS) begin
        if(~RSTN) begin
            linear_done_dly <= 1'd0;
            DATA_OUT <= 'd0;
        end else begin
            linear_done_dly <= linear_done;
            DATA_OUT <= (linear_done && !linear_done_dly) ? linear_dout : DATA_OUT;
        end
    end

endmodule
