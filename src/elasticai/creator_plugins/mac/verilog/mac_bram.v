//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
// 
// Create Date:     13.07.2026 09:22:58
// Copied on: 	    §{date_copy_created}
// Module Name:     Multiply-Accumulate Operator for BRAM Processing
// Target Devices:  FPGA / ASIC (call LUT-based multiplier with custom integration)
// Tool Versions:   1v1
// Description:     Performing a MAC Operation on Device (with Clamping, Pipelined Multiplier and Parallisation)
// Processing:      Data applied on posedge clk
//                  First cycle with DO_CALC --> Reset MAC output, add bias and stream data into pipeline
//                  After first cycle --> pipelined MAC operation
// Dependencies:    Parameter ordering in BRAM (weights, bias)
//
// State: 	        Works!
// Improvements:    None
// Parameters:      BITWIDTH            --> Bitwidth of input data
//                  SIZE_INPUT          --> Length of used data samples
//                  NUM_MULT            --> Number of used multiplier in parallel
//                  BITS_SCALE_BIAS     --> Bits for "left-shifting" the input bias
//                  BITS_SCALE_DOUT     --> Bits for right-shifting the output value (to apply FxP)
//                  INDEX_WEIGHTS_START --> Index of first weight in BRAM
//                  INDEX_BITWIDTH      --> Counter bitwidth of the BRAM index selector
//////////////////////////////////////////////////////////////////////////////////


module MAC#(
    parameter integer BITWIDTH = 8,
    parameter integer NUM_MULT = 2,
    parameter integer SIZE_INPUT = 2,
    parameter integer BITS_SCALE_BIAS = 0,
    parameter integer BITS_SCALE_DOUT = 0,
    parameter integer INDEX_WEIGHTS_START = 0,
    parameter integer INDEX_BITWIDTH = 4
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire DO_CALC,
    input wire DO_CLEAR,
    output wire RNW_BRAM,
    output wire [(NUM_MULT * INDEX_BITWIDTH)-'d1:0] IDX_BRAM,
    input wire signed [(NUM_MULT* BITWIDTH) -'d1:0] IN_BRAM,
    input wire signed [(NUM_MULT* BITWIDTH) -'d1:0] IN_DATA,
    output wire signed [(2* BITWIDTH) -'d1:0] OUT_DATA,
    output wire DATA_RDY
);
    localparam integer INDEX_WEIGHTS_STOPP = INDEX_WEIGHTS_START + SIZE_INPUT + 1;
    localparam integer NUM_K_PIPELINE_STAGE = 2;
    localparam integer NUM_CYC_COMPLETE_WOPAD = SIZE_INPUT / NUM_MULT;
    localparam integer NUM_ZERO_PADDING = (NUM_K_PIPELINE_STAGE * NUM_MULT) + (SIZE_INPUT - NUM_CYC_COMPLETE_WOPAD * NUM_MULT);
    localparam integer NUM_CYC_COMPLETE = 'd1 + SIZE_INPUT / NUM_MULT;
    localparam integer NUM_CYC_CNTSTOP = NUM_CYC_COMPLETE + 1;

    // --- Definition of internal signals and register
    reg do_calc_dly;
    reg do_operation;
    reg [$clog2(NUM_CYC_CNTSTOP):0] cnt_cyc_calc;
    wire do_clear_mac;
    assign do_clear_mac = (do_operation && (cnt_cyc_calc == 'd0)) || DO_CLEAR;

    // --- Interfacing the BRAM with the MAC core
    wire signed [BITWIDTH-'d1:0] bias_bram;
    wire signed [(NUM_MULT*BITWIDTH)-'d1:0] piped_data, piped_weights;

    genvar g;
    generate
        for (g = 0; g < NUM_MULT; g = g + 1) begin : gen_addr_wght
            assign IDX_BRAM[(g+1)*INDEX_BITWIDTH-1 -: INDEX_BITWIDTH] =
                (cnt_cyc_calc == 'd0)
                    ? INDEX_WEIGHTS_STOPP-'d1
                    : INDEX_WEIGHTS_START + (cnt_cyc_calc - 'd1)*NUM_MULT + g;
        end
    endgenerate
    assign bias_bram = (cnt_cyc_calc == 'd0) ? IN_BRAM[BITWIDTH-'d1:0] : 'sd0;
    assign RNW_BRAM = 'd0;
    assign piped_data = (|cnt_cyc_calc) ? IN_DATA : 'sd0;
    assign piped_weights = (|cnt_cyc_calc) ? IN_BRAM : 'sd0;

    // --- Computing the math operation
    assign DATA_RDY = !do_operation;

    MAC_CORE#(
        .BITWIDTH(BITWIDTH),
        .NUM_MULT(NUM_MULT),
        .BITS_OVERSIZE($clog2(SIZE_INPUT)),
        .BITS_SCALE_BIAS(BITS_SCALE_BIAS),
        .BITS_SCALE_DOUT(BITS_SCALE_DOUT)
    ) MAC_UNIT (
        .CLK_SYS(CLK_SYS),
        .RSTN(RSTN),
        .EN(EN),
        .DO_CALC(do_operation),
        .DO_CLEAR(do_clear_mac),
        .IN_BIAS(bias_bram),
        .IN_DATA(piped_data),
        .IN_WEIGHTS(piped_weights),
        .OUT_DATA(OUT_DATA)
    );

    // --- Controlling the operations data workflow
    integer i0;
    always@(posedge CLK_SYS) begin
        if(!RSTN) begin
            do_operation <= 1'd0;
            cnt_cyc_calc <= 'd0;
            do_calc_dly <= 1'd0;
        end else begin
            if(!EN) begin
                do_operation <= 1'd0;
                cnt_cyc_calc <= 'd0;
                do_calc_dly <= 1'd0;
            end else if(do_operation) begin
                // --- State: Do Calculation
                do_operation <= !(cnt_cyc_calc == NUM_CYC_CNTSTOP);
                cnt_cyc_calc <= cnt_cyc_calc + 'd1;
                do_calc_dly <= do_calc_dly;
            end else begin
                // --- State: Hold data
                do_operation <= !do_calc_dly && DO_CALC;
                cnt_cyc_calc <= 'd0;
                do_calc_dly <= DO_CALC;
            end
        end
    end
endmodule
