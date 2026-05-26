//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
// 
// Create Date:     24.05.2026, 08:19
// Copied on: 	    §{date_copy_created}
// Module Name:     Multiply-Accumulate Operator with Delta-compressed Weights
// Target Devices:  FPGA / ASIC (call LUT-based multiplier with custom integration)
// Tool Versions:   1v0
// Description:     Performing a MAC Operation on Device (with Clamping, Pipelined Multiplier and Parallisation)
// Processing:      Data applied on posedge clk
// Dependencies:    None
//
// State: 	        Works!
// Improvements:    None
// Parameters:      INPUT_BITWIDTH --> Bitwidth of input data
//                  INPUT_DELTAWIDTH --> Bitwidth of delta weights
//                  INPUT_NUM_DATA --> Length of used data
//                  NUM_MULT_PARALLEL --> Number of used multiplier in parallel
//////////////////////////////////////////////////////////////////////////////////


module MAC#(
    parameter INPUT_BITWIDTH = 6'd8,
    parameter INPUT_DELTAWIDTH = 12'd4,
    parameter INPUT_NUM_DATA = 12'd2,
    parameter NUM_MULT_PARALLEL = 4'd2
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
    // --- Local parameter for configuring the pipeline and parallisation of MAC
    localparam NUM_K_PIPELINE_STAGE = 4'd2;
    localparam NUM_CYC_COMPLETE_WOPAD = INPUT_NUM_DATA / NUM_MULT_PARALLEL;
    localparam NUM_ZERO_PADDING = INPUT_NUM_DATA - NUM_CYC_COMPLETE_WOPAD * NUM_MULT_PARALLEL;
    localparam NUM_CYC_COMPLETE = (INPUT_NUM_DATA + NUM_ZERO_PADDING) / NUM_MULT_PARALLEL;
    localparam NUM_CYC_CNTSTOP = NUM_CYC_COMPLETE + NUM_K_PIPELINE_STAGE - 'd1;
    localparam NUM_BITWIDTH_MAC = 2* INPUT_BITWIDTH + $clog2(INPUT_NUM_DATA);

    // --- Definition of Padded Input
    wire [(INPUT_NUM_DATA + NUM_ZERO_PADDING)* INPUT_BITWIDTH -'d1:0] padded_input_data, padded_input_wght;
    if(NUM_ZERO_PADDING > 0) begin
        assign padded_input_data = {{NUM_ZERO_PADDING* INPUT_BITWIDTH{1'd0}}, IN_DATA};
        assign padded_input_wght = {{NUM_ZERO_PADDING* INPUT_BITWIDTH{1'd0}}, IN_WEIGHTS};
    end else begin
        assign padded_input_data = IN_DATA;
        assign padded_input_wght = IN_WEIGHTS;
    end

    // --- Definition of internal signals and register
    reg do_calc_dly;
    reg active_process;
    reg [$clog2(NUM_CYC_CNTSTOP):0] cnt_cyc_calc;
    reg signed [INPUT_BITWIDTH-'d1:0] pipeline_input_a [NUM_MULT_PARALLEL-'d1:0];
    reg signed [INPUT_BITWIDTH-'d1:0] pipeline_input_b [NUM_MULT_PARALLEL-'d1:0];
    reg signed [2* INPUT_BITWIDTH-'d1:0] pipeline_output [NUM_MULT_PARALLEL-'d1:0];
    wire signed [2* INPUT_BITWIDTH-'d1:0] mult_output [NUM_MULT_PARALLEL-'d1:0];
    reg signed [NUM_BITWIDTH_MAC-'d1:0] mac_out;
    logic signed [NUM_BITWIDTH_MAC-'d1:0] sum_pipeline;

    wire do_shift_data, is_overflow, is_underflow;
    assign do_shift_data = ~((cnt_cyc_calc == NUM_CYC_CNTSTOP - 'd1) || (cnt_cyc_calc == NUM_CYC_CNTSTOP));

    assign DATA_RDY = ~active_process;
    assign is_overflow = ~mac_out[NUM_BITWIDTH_MAC-'d1] && |mac_out[NUM_BITWIDTH_MAC-'d2:2*INPUT_BITWIDTH-'d1];
    assign is_underflow = mac_out[NUM_BITWIDTH_MAC-'d1] && ~&mac_out[NUM_BITWIDTH_MAC-'d2:2*INPUT_BITWIDTH-'d1];

    // --- Clamping output data
    assign OUT_DATA =   (is_overflow) ? {1'b0, {(2*INPUT_BITWIDTH-'d1){1'b1}}} :
                        ((is_underflow) ? {1'b1, {(2*INPUT_BITWIDTH-'d1){1'b0}}} :
                        mac_out[2*INPUT_BITWIDTH-'d1:0]);
    // --- Using multiplier
    genvar k0;
    generate
        for(k0 = 'd0; k0 < NUM_MULT_PARALLEL; k0 = k0 + 'd1) begin
            MULT_SIGNED#(INPUT_BITWIDTH) MULT_UNIT(
                .A(pipeline_input_a[k0]),
                .B(pipeline_input_b[k0]),
                .Q(mult_output[k0])
            );
        end
    endgenerate
     // --- Adder Tree
    integer k1;
    always@(*) begin
        if(~(RSTN && EN)) begin
            sum_pipeline = 'd0;
        end else begin
            for (k1 = 'd0; k1 < NUM_MULT_PARALLEL; k1 = k1 + 'd1) begin
                if(k1 == 'd0) begin
                    sum_pipeline = pipeline_output[k1];
                end else begin
                    sum_pipeline = sum_pipeline + pipeline_output[k1];
                end
            end
        end
    end
    // --- Control device for pipeline multiplication
    integer i0;
    always@(posedge CLK_SYS) begin
        if(~(RSTN && EN)) begin
            do_calc_dly <= 1'd0;
            active_process <= 1'd0;
            cnt_cyc_calc <= 'd0;
            for(i0 = 'd0; i0 < NUM_MULT_PARALLEL; i0 = i0 + 'd1) begin
                pipeline_input_a[i0] <= 'd0;
                pipeline_input_b[i0] <= 'd0;
                pipeline_output[i0] <= 'd0;
            end
            mac_out <= 'sd0;
        end else begin
            do_calc_dly <= DO_CALC;
            if((~do_calc_dly && DO_CALC) || active_process) begin
                // --- State: Do Calculation
                active_process <= (cnt_cyc_calc == NUM_CYC_CNTSTOP) ? 1'd0 : 1'd1;
                cnt_cyc_calc <= (cnt_cyc_calc == NUM_CYC_CNTSTOP) ? 'd0 : cnt_cyc_calc + 'd1;
                for(i0 = 'd0; i0 < NUM_MULT_PARALLEL; i0 = i0 + 'd1) begin
                    pipeline_input_a[i0] <= (do_shift_data) ? padded_input_data[(cnt_cyc_calc + i0 * NUM_CYC_COMPLETE)* INPUT_BITWIDTH+: INPUT_BITWIDTH] : 'd0;
                    pipeline_input_b[i0] <= INITIAL_WEIGHT + ((do_shift_data) ? {{(INPUT_BITWIDTH-INPUT_DELTAWIDTH){padded_input_wght[(cnt_cyc_calc + i0 * NUM_CYC_COMPLETE + 1)* INPUT_DELTAWIDTH-'d1]}}, padded_input_wght[(cnt_cyc_calc + i0 * NUM_CYC_COMPLETE)* INPUT_DELTAWIDTH+: INPUT_DELTAWIDTH]} : 'd0);
                    pipeline_output[i0] <= mult_output[i0];
                end
                mac_out <= (cnt_cyc_calc == 'd0) ? {{(NUM_BITWIDTH_MAC-INPUT_BITWIDTH){IN_BIAS[INPUT_BITWIDTH-'d1]}}, IN_BIAS} : mac_out + sum_pipeline;
            end else begin
                // --- State: Hold data
                active_process <= active_process;
                cnt_cyc_calc <= cnt_cyc_calc;
                for(i0 = 'd0; i0 < NUM_MULT_PARALLEL; i0 = i0 + 'd1) begin
                    pipeline_input_a[i0] <= pipeline_input_a[i0];
                    pipeline_input_b[i0] <= pipeline_input_b[i0];
                    pipeline_output[i0] <= pipeline_output[i0];
                end
                mac_out <= mac_out;
            end
        end
    end
endmodule
