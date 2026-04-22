//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
// 
// Create Date:     17.01.2025 08:11:51
// Copied on: 	    §{date_copy_created}
// Module Name:     LUT-based Multiply-Accumulate Operator
// Target Devices:  ASIC (using LUT multiplier)
// Tool Versions:   1v0
// Description:     Performing a MAC Operation on Device (with Pipelined Multiplier and Parallisation)
// Processing:      Data applied on posedge clk
// Dependencies:    None
//
// State: 	        Works! (System Test done: 22.01.2025 on Arty A7-35T with 20% usage)
// Improvements:    None
// Parameters:      INPUT_BITWIDTH --> Bitwidth of input data
//                  INPUT_NUM_DATA --> Length of used data
//                  NUM_MULT_PARALLEL --> Number of used multiplier in parallel
//////////////////////////////////////////////////////////////////////////////////


module MAC_LUT#(
    parameter INPUT_BITWIDTH = 6'd8,
    parameter INPUT_NUM_DATA = 12'd2,
    parameter NUM_MULT_PARALLEL = 4'd2
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire DO_CALC,
    input wire signed [INPUT_BITWIDTH -'d1:0] IN_BIAS,
    input wire signed [INPUT_NUM_DATA* INPUT_BITWIDTH -'d1:0] IN_WEIGHTS,
    input wire signed [INPUT_NUM_DATA* INPUT_BITWIDTH -'d1:0] IN_DATA,
    output wire signed [2* INPUT_BITWIDTH -'d1:0] OUT_DATA,
    output wire DATA_VALID
);
    // --- Local parameter for configuring the pipeline and parallisation of MAC
    localparam NUM_K_PIPELINE_STAGE = 4'd2;
    localparam NUM_CYC_COMPLETE_WOPAD = INPUT_NUM_DATA / NUM_MULT_PARALLEL;
    localparam NUM_ZERO_PADDING = INPUT_NUM_DATA - NUM_CYC_COMPLETE_WOPAD * NUM_MULT_PARALLEL;
    localparam NUM_CYC_COMPLETE = (INPUT_NUM_DATA + NUM_ZERO_PADDING) / NUM_MULT_PARALLEL;
    localparam NUM_CYC_CNTSTOP = NUM_CYC_COMPLETE + NUM_K_PIPELINE_STAGE - 'd1;
    localparam NUM_BITWIDTH_MAC = 2* INPUT_BITWIDTH + $clog2(NUM_MULT_PARALLEL);

    // --- Definition of Padded Input
    wire [(INPUT_NUM_DATA + NUM_ZERO_PADDING)* INPUT_BITWIDTH -'d1:0] padded_input_data;
    wire [(INPUT_NUM_DATA + NUM_ZERO_PADDING)* INPUT_BITWIDTH -'d1:0] padded_input_wght;
    assign padded_input_data = {{NUM_ZERO_PADDING* INPUT_BITWIDTH{1'd0}}, IN_DATA};
    assign padded_input_wght = {{NUM_ZERO_PADDING* INPUT_BITWIDTH{1'd0}}, IN_WEIGHTS};

    // --- Definition of internal signals and register
    reg [1:0] do_calc_dly;
    reg active_process;
    reg [$clog2(NUM_CYC_CNTSTOP):0] cnt_cyc_calc;
    reg signed [INPUT_BITWIDTH-'d1:0] pipeline_input_a [NUM_MULT_PARALLEL-'d1:0];
    reg signed [INPUT_BITWIDTH-'d1:0] pipeline_input_b [NUM_MULT_PARALLEL-'d1:0];
    reg signed [2* INPUT_BITWIDTH-'d1:0] pipeline_output [NUM_MULT_PARALLEL-'d1:0];
    wire signed [2* INPUT_BITWIDTH-'d1:0] mult_output [NUM_MULT_PARALLEL-'d1:0];
    reg signed [NUM_BITWIDTH_MAC-'d1:0] mac_out;
    logic signed [NUM_BITWIDTH_MAC-'d1:0] sum_pipeline;

    wire do_shift_data;
    assign do_shift_data = ~((cnt_cyc_calc == NUM_CYC_CNTSTOP - 'd1) || (cnt_cyc_calc == NUM_CYC_CNTSTOP));
    assign OUT_DATA = mac_out[2*INPUT_BITWIDTH-'d1:0];
    assign DATA_VALID = ~active_process;

    // --- Using custom multiplier
    genvar k0;
    generate
        for(k0 = 'd0; k0 < NUM_MULT_PARALLEL; k0 = k0 + 'd1) begin
            // --> Change module with optimized LUT multiplier if desired
            MULT_LUT_SIGNED#(INPUT_BITWIDTH) MULT_UNIT(
                .A(pipeline_input_a[k0]),
                .B(pipeline_input_b[k0]),
                .Q(mult_output[k0])
            );
        end
    endgenerate

    // --- Adder Tree
    integer i0;
    always_comb begin
        if(~(RSTN && EN)) begin
            sum_pipeline = 'd0;
        end else begin
            for (i0 = 'd0; i0 < NUM_MULT_PARALLEL; i0 = i0 + 'd1) begin
                if(i0 == 'd0) begin
                    sum_pipeline = pipeline_output[i0];
                end else begin
                    sum_pipeline = sum_pipeline + pipeline_output[i0];
                end
            end
        end
    end
    // --- Control device for pipeline multiplication
    integer k1;
    always@(posedge CLK_SYS) begin
        if(~(RSTN && EN)) begin
            do_calc_dly <= 2'd0;
            active_process <= 1'd0;
            cnt_cyc_calc <= 'd0;
            for(i0 = 'd0; i0 < NUM_MULT_PARALLEL; i0 = i0 + 'd1) begin
                pipeline_input_a[i0] <= 'd0;
                pipeline_input_b[i0] <= 'd0;
                pipeline_output[i0] <= 'd0;
            end
            mac_out <= 'd0;
        end else begin
            do_calc_dly <= {do_calc_dly[0], DO_CALC};
            if((~do_calc_dly[1] && do_calc_dly[0]) || active_process) begin
                // --- State: Do Calculation
                active_process <= (cnt_cyc_calc == NUM_CYC_CNTSTOP) ? 1'd0 : 1'd1;
                cnt_cyc_calc <= (cnt_cyc_calc == NUM_CYC_CNTSTOP) ? 'd0 : cnt_cyc_calc + 'd1;
                for(i0 = 'd0; i0 < NUM_MULT_PARALLEL; i0 = i0 + 'd1) begin
                    pipeline_input_a[i0] <= (do_shift_data) ? padded_input_data[(cnt_cyc_calc + i0 * NUM_CYC_COMPLETE)* INPUT_BITWIDTH+: INPUT_BITWIDTH] : 'd0;
                    pipeline_input_b[i0] <= (do_shift_data) ? padded_input_wght[(cnt_cyc_calc + i0 * NUM_CYC_COMPLETE)* INPUT_BITWIDTH+: INPUT_BITWIDTH] : 'd0;
                    pipeline_output[i0] <= mult_output[i0];
                end
                mac_out <= (cnt_cyc_calc == 'd0) ? {{(INPUT_BITWIDTH){IN_BIAS[INPUT_BITWIDTH-'d1]}}, IN_BIAS} : mac_out + sum_pipeline;
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

