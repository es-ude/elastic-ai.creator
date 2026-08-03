//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
// 
// Create Date:     13.07.2026 09:22:58
// Copied on: 	    §{date_copy_created}
// Module Name:     Multiply-Accumulate Operator for BRAM Processing with Delta-Weights
// Target Devices:  FPGA / ASIC (call LUT-based multiplier with custom integration)
// Tool Versions:   1v1
// Description:     Performing a MAC Operation on Device (with Clamping, Pipelined Multiplier and Parallisation)
// Processing:      Data applied on posedge clk
//                  First cycle with DO_CALC --> Reset MAC output, add bias and stream data into pipeline
//                  After first cycle --> pipelined MAC operation
// Dependencies:    Parameter ordering in BRAM (weights, initial weight, bias)
//
// State: 	        Works!
// Improvements:    Taking more samples from BRAM (direct compression)
// Parameters:      INPUT_BITWIDTH --> Bitwidth of input data
//                  INPUT_RAMWIDTH --> Bitwidth of the data, loaded from BRAM
//                  INPUT_DELTAWIDTH --> Bitwidth of the compressed weights
//                  INPUT_NUM_DATA --> Length of used data samples
//                  NUM_MULT_PARALLEL --> Number of used multiplier in parallel
//                  INDEX_WEIGHTS_START --> Index of first weight in BRAM
//                  INDEX_BITWIDTH --> Counter bitwidth of the BRAM index selector
//                  DELTA_MODE --> Selection of the decompression mode (0=consecutive, 1=fixed-reference)
//////////////////////////////////////////////////////////////////////////////////


module MAC_DELTA#(
    parameter integer INPUT_BITWIDTH = 8,
    parameter integer INPUT_RAMWIDTH = 8,
    parameter integer INPUT_DELTAWIDTH = 4,
    parameter integer INPUT_NUM_DATA = 2,
    parameter integer NUM_MULT_PARALLEL = 2,
    parameter integer INDEX_WEIGHTS_START = 0,
    parameter integer INDEX_BITWIDTH = 4,
    parameter integer DELTA_MODE = 0
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire DO_CALC,
    output wire RNW_BRAM,
    output wire [(NUM_MULT_PARALLEL * INDEX_BITWIDTH)-'d1:0] IDX_BRAM,
    input wire signed [(NUM_MULT_PARALLEL* INPUT_RAMWIDTH) -'d1:0] IN_BRAM,
    input wire signed [(NUM_MULT_PARALLEL* INPUT_BITWIDTH) -'d1:0] IN_DATA,
    output wire signed [(2* INPUT_BITWIDTH) -'d1:0] OUT_DATA,
    output wire DATA_RDY
);
    localparam INDEX_WEIGHTS_STOPP = INDEX_WEIGHTS_START + INPUT_NUM_DATA + 'd2;
    localparam NUM_K_PIPELINE_STAGE = 4'd2;
    localparam NUM_CYC_COMPLETE_WOPAD = INPUT_NUM_DATA / NUM_MULT_PARALLEL;
    localparam NUM_ZERO_PADDING = (NUM_K_PIPELINE_STAGE * NUM_MULT_PARALLEL) + (INPUT_NUM_DATA - NUM_CYC_COMPLETE_WOPAD * NUM_MULT_PARALLEL);
    localparam NUM_CYC_COMPLETE = 'd1 + INPUT_NUM_DATA / NUM_MULT_PARALLEL;
    localparam NUM_CYC_CNTSTOP = NUM_CYC_COMPLETE + 'd1;

    // --- Definition of internal signals and register
    reg do_calc_dly;
    reg do_operation;
    reg [$clog2(NUM_CYC_CNTSTOP):0] cnt_cyc_calc;

    // --- Interfacing the BRAM with the MAC core
    reg signed [INPUT_BITWIDTH-'d1:0] initial_weight;
    wire signed [INPUT_BITWIDTH-'d1:0] bias_bram;
    wire signed [(NUM_MULT_PARALLEL*INPUT_BITWIDTH)-'d1:0] piped_data;
    wire signed [(NUM_MULT_PARALLEL*INPUT_DELTAWIDTH)-'d1:0] piped_wght;

    genvar g;
    generate
        for (g = 0; g < NUM_MULT_PARALLEL; g = g + 1) begin : gen_addr_wght
            assign IDX_BRAM[(g+1)*INDEX_BITWIDTH-1 -: INDEX_BITWIDTH] =
                (cnt_cyc_calc == 'd0)
                    ? INDEX_WEIGHTS_STOPP-'d1
                    : INDEX_WEIGHTS_START + (cnt_cyc_calc - 'd1)*NUM_MULT_PARALLEL + g;
        end
    endgenerate
    assign bias_bram = (cnt_cyc_calc == 'd0) ? IN_BRAM[INPUT_BITWIDTH-'d1:0] : 'sd0;
    assign RNW_BRAM = 'd0;
    assign piped_data = (|cnt_cyc_calc) ? IN_DATA : 'sd0;
    assign piped_wght = (|cnt_cyc_calc) ? IN_BRAM : 'sd0;

    // --- Computing the math operation
    assign DATA_RDY = !do_operation;

    MAC_DELTA_CORE#(
        .INPUT_BITWIDTH(INPUT_BITWIDTH),
        .INPUT_DELTAWIDTH(INPUT_DELTAWIDTH),
        .NUM_MULT_PARALLEL(NUM_MULT_PARALLEL),
        .NUM_SUM_OVERSIZE($clog2(INPUT_NUM_DATA)),
        .DELTA_MODE(DELTA_MODE)
    ) MAC_UNIT (
        .CLK_SYS(CLK_SYS),
        .RSTN(RSTN),
        .EN(EN),
        .DO_CALC(do_operation),
        .IN_BIAS(bias_bram),
        .INITIAL_WEIGHT(initial_weight),
        .IN_DATA(piped_data),
        .IN_WEIGHTS(piped_wght),
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
