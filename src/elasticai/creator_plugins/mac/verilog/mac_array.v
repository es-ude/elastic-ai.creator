//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
// 
// Create Date:     12.07.2026 18:15:33
// Copied on: 	    §{date_copy_created}
// Module Name:     Multiply-Accumulate Operator for Array Processing
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
// Parameters:      BITWIDTH        --> Bitwidth of input data
//                  NUM_MULT        --> Number of used multiplier in parallel
//                  SIZE_INPUT      --> Length of used data
//                  BITS_SCALE_BIAS --> Bits for "left-shifting" the input bias
//                  BITS_SCALE_DOUT --> Bits for right-shifting the output value (to apply FxP)
//////////////////////////////////////////////////////////////////////////////////


module MAC#(
    parameter integer BITWIDTH = 8,
    parameter integer NUM_MULT = 2,
    parameter integer SIZE_INPUT = 2,
    parameter integer BITS_SCALE_BIAS = 0,
    parameter integer BITS_SCALE_DOUT = 0
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire DO_CLEAR,
    input wire DO_CALC,
    input wire signed [BITWIDTH-1:0] IN_BIAS,
    input wire signed [SIZE_INPUT* BITWIDTH-1:0] IN_WEIGHTS,
    input wire signed [SIZE_INPUT* BITWIDTH-1:0] IN_DATA,
    output wire signed [BITWIDTH-1:0] OUT_DATA,
    output wire DATA_RDY
);
    localparam integer NUM_CYC_COMPLETE_WOPAD = SIZE_INPUT / NUM_MULT;
    localparam integer NUM_ZERO_PADDING = (SIZE_INPUT - NUM_CYC_COMPLETE_WOPAD * NUM_MULT);
    localparam integer NUM_CYC_COMPLETE = (SIZE_INPUT + NUM_ZERO_PADDING) / NUM_MULT;
    localparam integer NUM_CYC_CNTSTOP = NUM_CYC_COMPLETE;

    // --- Definition of internal signals and register
    reg do_calc_dly;
    reg active_process;
    reg [$clog2(NUM_CYC_CNTSTOP):0] cnt_cyc_calc;

    // --- Definition of Padded Input
    wire [(SIZE_INPUT + NUM_ZERO_PADDING)* BITWIDTH -'d1:0] padded_input_data, padded_input_wght;
    if(NUM_ZERO_PADDING > 0) begin
        assign padded_input_data = {{(NUM_ZERO_PADDING* BITWIDTH){1'd0}}, IN_DATA};
        assign padded_input_wght = {{(NUM_ZERO_PADDING* BITWIDTH){1'd0}}, IN_WEIGHTS};
    end else begin
        assign padded_input_data = IN_DATA;
        assign padded_input_wght = IN_WEIGHTS;
    end

    // --- Slicing the input data to array
    wire signed [NUM_MULT* BITWIDTH-'d1:0] pipeline_input_a;
    wire signed [NUM_MULT* BITWIDTH-'d1:0] pipeline_input_b;

    assign pipeline_input_a = padded_input_data[(cnt_cyc_calc * NUM_MULT*BITWIDTH)+: NUM_MULT*BITWIDTH];
    assign pipeline_input_b = padded_input_wght[(cnt_cyc_calc * NUM_MULT*BITWIDTH)+: NUM_MULT*BITWIDTH];

    // --- Computing the math operation
    wire do_operation, do_clear_mac;
    assign do_operation = active_process;
    assign do_clear_mac = (active_process && (cnt_cyc_calc == 'd0)) || DO_CLEAR;
    assign DATA_RDY = !active_process;

    MAC_CORE#(
        .BITWIDTH(BITWIDTH),
        .NUM_MULT(NUM_MULT),
        .BITS_OVERSIZE($clog2(SIZE_INPUT/4)),
        .BITS_SCALE_BIAS(BITS_SCALE_BIAS),
        .BITS_SCALE_DOUT(BITS_SCALE_DOUT)
    ) MAC_UNIT (
        .CLK_SYS(CLK_SYS),
        .RSTN(RSTN),
        .EN(EN),
        .DO_CALC(do_operation),
        .DO_CLEAR(do_clear_mac),
        .IN_BIAS(IN_BIAS),
        .IN_DATA(pipeline_input_a),
        .IN_WEIGHTS(pipeline_input_b),
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
            if(EN && active_process) begin
                // --- State: Do Calculation
                active_process <= !(cnt_cyc_calc == NUM_CYC_CNTSTOP);
                cnt_cyc_calc <= cnt_cyc_calc + 'd1;
                do_calc_dly <= do_calc_dly;
            end else begin
                // --- State: Hold data
                active_process <= !do_calc_dly && DO_CALC;
                cnt_cyc_calc <= 'd0;
                do_calc_dly <= EN && DO_CALC;
            end
        end
    end
endmodule
