//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     18.09.2026, 15:47:03
// Copied on: 	    §{date_copy_created}
// Module Name:     Template of a linear / fully connected layer (only inference)
// Target Devices:  FPGA
// Tool Versions:   1v0
// Processing:
// Dependencies:    MAC operator, multipliers (DSP, LUT)
//
// State: 	        Not tested!
// Improvements:    Actual rounding error at output, coming from downscaling? Add
// Parameters:      BITWIDTH        --> Bitwidth of input data
//                  SIZE_INPUT      --> Number of input values
//                  SIZE_OUTPUT     --> Number of output values (=number of neurons)
//                  NUM_MULT        --> Number of multiplier units used in the MAC
//                  BITS_SCALE_BIAS --> Bits for left-shifting the input bias value (to apply FxP)
//                  BITS_SCALE_DOUT --> Bits for right-shifting the output value (to apply FxP)
//////////////////////////////////////////////////////////////////////////////////


module LINEAR_ARRAY#(
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
    // --- Local parameters to set signals
    localparam integer BITS_WEIGHTS_NEURON = SIZE_INPUT * BITWIDTH;
    localparam integer STOP_NEURON_CNT = SIZE_INPUT / NUM_MULT - 1;

    // --- Parameter ordering = BIAS: {n-1, ..., 0}, Weights: {w_(n-1), ... w_0}
    localparam signed [SIZE_OUTPUT* BITWIDTH-1:0] BIAS = { 8'sd3, 8'sd2, 8'sd1 };
    localparam signed [SIZE_OUTPUT* BITS_WEIGHTS_NEURON-1:0] WEIGHTS = { -8'sd12, 8'sd11, -8'sd10, 8'sd9, -8'sd8, 8'sd7, -8'sd6, 8'sd5, -8'sd4, 8'sd3, -8'sd2, 8'sd1 };

    // --- Internal signals
    wire [BITWIDTH-1:0] mac_dout;
    wire mac_done, calc_trigger_ops, clear_ops;
    reg calc_delay, sample_mac;
    reg [$clog2(STOP_NEURON_CNT):0] cnt_ite;
    reg [$clog2(SIZE_OUTPUT):0] cnt_idx;
    wire [SIZE_OUTPUT * BITS_WEIGHTS_NEURON-1:0] data0;

    assign clear_ops = ~(|cnt_ite || mac_done);
    assign calc_trigger_ops = EN && DO_CALC && calc_delay;
    assign DATA_VALID = mac_done && cnt_idx != SIZE_OUTPUT;
    assign MOD_RDY = DATA_VALID;

    // --- Padding the input to full array
    genvar idx;
    generate
        for (idx = 0; idx < SIZE_OUTPUT; idx = idx + 1) begin : g_flatten
            assign data0[(idx*BITS_WEIGHTS_NEURON)+:BITS_WEIGHTS_NEURON] = DATA_IN;
        end
    endgenerate

    // --- Using the MAC
    MAC#(
        .BITWIDTH(BITWIDTH),
        .NUM_MULT(NUM_MULT),
        .SIZE_INPUT(SIZE_INPUT * SIZE_OUTPUT),
        .BITS_SCALE_BIAS(BITS_SCALE_BIAS),
        .BITS_SCALE_DOUT(BITS_SCALE_DOUT)
    ) MAC_UNIT (
        .CLK_SYS(CLK_SYS),
        .RSTN(RSTN),
        .EN(EN),
        .DO_CALC(calc_trigger_ops),
        .DO_CLEAR(clear_ops),
        .IN_BIAS(BIAS[(cnt_idx*BITWIDTH)+:BITWIDTH]),
        .IN_WEIGHTS(WEIGHTS),
        .IN_DATA(data0),
        .OUT_DATA(mac_dout),
        .DATA_RDY(mac_done)
    );

    // --- Control Sequence
    always@(posedge CLK_SYS) begin
        if(~RSTN) begin
            calc_delay <= 1'd0;
            sample_mac <= 1'd0;
            cnt_ite <= 'd0;
            cnt_idx <= 'd0;
            DATA_OUT <= 'd0;
        end else begin
            calc_delay <= DO_CALC;
            DATA_OUT <= (sample_mac) ? {mac_dout, DATA_OUT[BITWIDTH+:(SIZE_OUTPUT-1)*BITWIDTH]} : DATA_OUT;
            sample_mac <= ~mac_done && (cnt_ite == STOP_NEURON_CNT);
            if(calc_trigger_ops || ~mac_done) begin
                cnt_ite <= (calc_delay || cnt_ite == STOP_NEURON_CNT) ? 'd0 : cnt_ite + 'd1;
                if(NUM_MULT == SIZE_INPUT) begin
                    cnt_idx <= cnt_idx + ((~calc_delay) ? 'd1 : 'd0);
                end else begin
                    cnt_idx <= cnt_idx + ((cnt_ite == STOP_NEURON_CNT) ? 'd1 : 'd0);
                end
            end else begin
                cnt_ite <= 'd0;
                cnt_idx <= 'd0;
            end
        end
    end
endmodule
