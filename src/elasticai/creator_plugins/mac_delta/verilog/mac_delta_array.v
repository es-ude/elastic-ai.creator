//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
// 
// Create Date:     12.07.2026 18:15:33
// Copied on: 	    §{date_copy_created}
// Module Name:     Multiply-Accumulate Operator for Array Processing with Delta-Weights
// Target Devices:  FPGA / ASIC (call LUT-based multiplier with custom integration)
// Tool Versions:   1v1
// Description:     Performing a MAC Operation on Device (with Clamping, Pipelined Multiplier and Parallisation)
// Processing:      Data applied on posedge clk
//                  First cycle with DO_CALC --> Reset MAC output, add bias and stream data into pipeline
//                  After first cycle --> pipelined MAC operation
// Dependencies:    None
//
// State: 	        Works!
// Improvements:    None
// Parameters:      INPUT_BITWIDTH --> Bitwidth of input data
//                  INPUT_DELTAWIDTH --> Bitwidth of the compressed weights
//                  INPUT_NUM_DATA --> Length of used data
//                  NUM_MULT_PARALLEL --> Number of used multiplier in parallel
//                  DELTA_MODE --> Selection of the decompression mode (0=consecutive, 1=fixed-reference)
//////////////////////////////////////////////////////////////////////////////////


module MAC_DELTA#(
    parameter integer INPUT_BITWIDTH = 8,
    parameter integer INPUT_DELTAWIDTH = 4,
    parameter integer INPUT_NUM_DATA = 2,
    parameter integer NUM_MULT_PARALLEL = 2,
    parameter integer DELTA_MODE = 0
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire DO_CALC,
    input wire signed [INPUT_BITWIDTH -'d1:0] IN_BIAS,
    input wire signed [INPUT_BITWIDTH -'d1:0] INITIAL_WEIGHT,
    input wire signed [INPUT_NUM_DATA* INPUT_DELTAWIDTH -'d1:0] IN_WEIGHTS,
    input wire signed [INPUT_NUM_DATA* INPUT_BITWIDTH -'d1:0] IN_DATA,
    output wire signed [2* INPUT_BITWIDTH -'d1:0] OUT_DATA,
    output wire DATA_RDY
);
    localparam NUM_K_PIPELINE_STAGE = 4'd2;
    localparam NUM_CYC_COMPLETE_WOPAD = INPUT_NUM_DATA / NUM_MULT_PARALLEL;
    localparam NUM_ZERO_PADDING = (NUM_K_PIPELINE_STAGE * NUM_MULT_PARALLEL) + (INPUT_NUM_DATA - NUM_CYC_COMPLETE_WOPAD * NUM_MULT_PARALLEL);
    localparam NUM_CYC_COMPLETE = (INPUT_NUM_DATA + NUM_ZERO_PADDING) / NUM_MULT_PARALLEL;
    localparam NUM_CYC_CNTSTOP = NUM_CYC_COMPLETE - 'd1;

    // --- Definition of internal signals and register
    reg do_calc_dly;
    reg active_process;
    reg [$clog2(NUM_CYC_CNTSTOP):0] cnt_cyc_calc;

    // --- Definition of Padded Input
    wire [(INPUT_NUM_DATA + NUM_ZERO_PADDING)* INPUT_BITWIDTH -'d1:0] padded_input_data;
    wire [(INPUT_NUM_DATA + NUM_ZERO_PADDING)* INPUT_DELTAWIDTH -'d1:0] padded_input_wght;
    if(NUM_ZERO_PADDING > 0) begin
        assign padded_input_data = {{(NUM_ZERO_PADDING* INPUT_BITWIDTH){1'd0}}, IN_DATA};
        assign padded_input_wght = {{(NUM_ZERO_PADDING* INPUT_DELTAWIDTH){1'd0}}, IN_WEIGHTS};
    end else begin
        assign padded_input_data = IN_DATA;
        assign padded_input_wght = IN_WEIGHTS;
    end

    // --- Slicing the input data to array
    wire signed [NUM_MULT_PARALLEL* INPUT_BITWIDTH-'d1:0] piped_data;
    wire signed [NUM_MULT_PARALLEL* INPUT_DELTAWIDTH-'d1:0] piped_wght;

    assign piped_data = padded_input_data[(cnt_cyc_calc * NUM_MULT_PARALLEL*INPUT_BITWIDTH)+: NUM_MULT_PARALLEL*INPUT_BITWIDTH];
    assign piped_wght = padded_input_wght[(cnt_cyc_calc * NUM_MULT_PARALLEL*INPUT_DELTAWIDTH)+: NUM_MULT_PARALLEL*INPUT_DELTAWIDTH];

    // --- Computing the math operation
    wire do_operation;

    assign do_operation = active_process;
    assign DATA_RDY = !active_process;

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
        .IN_BIAS(IN_BIAS),
        .INITIAL_WEIGHT(INITIAL_WEIGHT),
        .IN_DATA(piped_data),
        .IN_WEIGHTS(piped_wght),
        .OUT_DATA(OUT_DATA)
    );

    // --- Controlling the operations data workflow
    integer i0;
    always@(posedge CLK_SYS) begin
        if(!RSTN) begin
            active_process <= 1'd0;
            cnt_cyc_calc <= 'd0;
            do_calc_dly <= 1'd0;
        end else begin
            if(!EN) begin
                active_process <= 1'd0;
                cnt_cyc_calc <= 'd0;
                do_calc_dly <= 1'd0;
            end else if(active_process) begin
                // --- State: Do Calculation
                active_process <= !(cnt_cyc_calc == NUM_CYC_CNTSTOP);
                cnt_cyc_calc <= cnt_cyc_calc + 'd1;
                do_calc_dly <= do_calc_dly;
            end else begin
                // --- State: Hold data
                active_process <= !do_calc_dly && DO_CALC;
                cnt_cyc_calc <= 'd0;
                do_calc_dly <= DO_CALC;
            end
        end
    end
endmodule
