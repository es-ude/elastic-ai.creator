//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
// 
// Create Date:     24.05.2026, 08:19
// Copied on: 	    §{date_copy_created}
// Module Name:     Computation Core for Multiply-Accumulate Operator with Delta-Compression
// Target Devices:  FPGA / ASIC (call LUT-based multiplier with custom integration)
// Tool Versions:   1v1
// Description:     Performing a MAC Operation on Device (with Clamping, Pipelined Multiplier and Parallization)
// Processing:      Data applied on posedge clk
//                  First cycle with DO_CALC --> Reset MAC output, add bias and stream data into pipeline
//                  After first cycle --> pipelined MAC operation
// Dependencies:    None
//
// State: 	        Works!
// Improvements:    None
// Parameters:      INPUT_BITWIDTH --> Bitwidth of input data
//                  INPUT_DELTAWIDTH --> Bitwidth of delta weights
//                  NUM_MULT_PARALLEL --> Number of used multiplier in parallel
//                  NUM_SUM_OVERSIZE --> Number of bits to oversize the sum unit
//                  DELTA_MODE --> Selection of the decompression mode (0=consecutive, 1=fixed-reference)
//////////////////////////////////////////////////////////////////////////////////


module MAC_DELTA_CORE#(
    parameter integer INPUT_BITWIDTH = 8,
    parameter integer INPUT_DELTAWIDTH = 4,
    parameter integer NUM_MULT_PARALLEL = 2,
    parameter integer NUM_SUM_OVERSIZE = 2,
    parameter integer DELTA_MODE = 0
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire DO_CALC,
    input wire signed [INPUT_BITWIDTH -'d1:0] IN_BIAS,
    input wire signed [INPUT_BITWIDTH -'d1:0] INITIAL_WEIGHT,
    input wire signed [INPUT_DELTAWIDTH * NUM_MULT_PARALLEL -'d1:0] IN_WEIGHTS,
    input wire signed [INPUT_BITWIDTH * NUM_MULT_PARALLEL -'d1:0] IN_DATA,
    output wire signed [2* INPUT_BITWIDTH -'d1:0] OUT_DATA
);
    // --- Local parameter for configuring the pipeline and parallelization of MAC
    localparam NUM_BITWIDTH_MAC = 2* INPUT_BITWIDTH + NUM_SUM_OVERSIZE;

    // --- Definition of internal signals and register
    reg do_calc_dly;
    if(DELTA_MODE == 0) begin
        reg signed [INPUT_BITWIDTH-'d1:0] weight_decompressed [NUM_MULT_PARALLEL-'d1:0];
        reg signed [INPUT_BITWIDTH-'d1:0] weight_decompressed_comb [NUM_MULT_PARALLEL-'d1:0];
    end
    reg signed [INPUT_BITWIDTH-'d1:0] pipeline_input_a [NUM_MULT_PARALLEL-'d1:0];
    reg signed [INPUT_BITWIDTH-'d1:0] pipeline_input_b [NUM_MULT_PARALLEL-'d1:0];
    reg signed [2* INPUT_BITWIDTH-'d1:0] pipeline_output [NUM_MULT_PARALLEL-'d1:0];
    wire signed [2* INPUT_BITWIDTH-'d1:0] mult_output [NUM_MULT_PARALLEL-'d1:0];
    reg signed [NUM_BITWIDTH_MAC-'d1:0] mac_out;
    reg signed [NUM_BITWIDTH_MAC-'d1:0] sum_pipeline;
    wire is_overflow, is_underflow;
    wire do_load_bias;

    assign do_load_bias = DO_CALC && !do_calc_dly;
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
        if(!RSTN) begin
            sum_pipeline = 'd0;
        end else begin
            if(EN) begin
                for (k1 = 'd0; k1 < NUM_MULT_PARALLEL; k1 = k1 + 'd1) begin
                    if(k1 == 'd0) begin
                        sum_pipeline = pipeline_output[k1];
                    end else begin
                        sum_pipeline = sum_pipeline + pipeline_output[k1];
                    end
                end
            end else begin
                sum_pipeline = 'd0;
            end
        end
    end

    generate
        // --- Weight Decompression (Consecutive), with only DELTA_MODE=0
        if(DELTA_MODE == 0) begin : gen_delta_consecutive
            reg signed [INPUT_BITWIDTH-'d1:0] weight_decompressed;
            reg signed [INPUT_BITWIDTH-'d1:0] weight_decompressed_comb [NUM_MULT_PARALLEL-'d1:0];

            integer i1;
            always@(*) begin
                for (i1 = 'd0; i1 < NUM_MULT_PARALLEL; i1 = i1 + 'd1) begin
                    weight_decompressed_comb[i1] = (((i1 == 'd0) ? ((do_load_bias) ? INITIAL_WEIGHT : weight_decompressed) : weight_decompressed_comb[i1-'d1])) +
                            {{(INPUT_BITWIDTH-INPUT_DELTAWIDTH){IN_WEIGHTS[(i1 + 'd1)* INPUT_DELTAWIDTH-'d1]}}, IN_WEIGHTS[(i1 * INPUT_DELTAWIDTH)+: INPUT_DELTAWIDTH]};
                    //weight_decompressed_comb[i1] = {{(INPUT_BITWIDTH-INPUT_DELTAWIDTH){IN_WEIGHTS[(i1 + 'd1)* INPUT_DELTAWIDTH-'d1]}}, IN_WEIGHTS[(i1 * INPUT_DELTAWIDTH)+: INPUT_DELTAWIDTH]};
                end
            end

            integer i0;
            always@(posedge CLK_SYS) begin
                if(!RSTN) begin
                    weight_decompressed <= 'sd0;
                    for(i0 = 'd0; i0 < NUM_MULT_PARALLEL; i0 = i0 + 'd1) begin
                        pipeline_input_a[i0] <= 'sd0;
                    end
                end else if(EN && DO_CALC) begin
                    weight_decompressed <= weight_decompressed_comb[NUM_MULT_PARALLEL-'d1];
                    for(i0 = 'd0; i0 < NUM_MULT_PARALLEL; i0 = i0 + 'd1) begin
                        pipeline_input_a[i0] <= weight_decompressed_comb[i0];
                    end
                end else begin
                    weight_decompressed <= 'sd0;
                    for(i0 = 'd0; i0 < NUM_MULT_PARALLEL; i0 = i0 + 'd1) begin
                        pipeline_input_a[i0] <= 'sd0;
                    end
                end
            end
        end else begin : gen_delta_referencing
            // --- Weight Decompression (Fixed-Referencing), with only DELTA_MODE=1
            integer i0;
            always@(posedge CLK_SYS) begin
                if(!RSTN) begin
                    for(i0 = 'd0; i0 < NUM_MULT_PARALLEL; i0 = i0 + 'd1) begin
                        pipeline_input_a[i0] <= 'sd0;
                    end
                end else if(EN && DO_CALC) begin
                    for(i0 = 'd0; i0 < NUM_MULT_PARALLEL; i0 = i0 + 'd1) begin
                        pipeline_input_a[i0] <= INITIAL_WEIGHT +
                                {{(INPUT_BITWIDTH-INPUT_DELTAWIDTH){IN_WEIGHTS[(i0 + 'd1)* INPUT_DELTAWIDTH-'d1]}}, IN_WEIGHTS[(i0* INPUT_DELTAWIDTH)+: INPUT_DELTAWIDTH]};
                    end
                end else begin
                    for(i0 = 'd0; i0 < NUM_MULT_PARALLEL; i0 = i0 + 'd1) begin
                        pipeline_input_a[i0] <= 'sd0;
                    end
                end
            end
        end
    endgenerate

    // --- Control device for pipeline_input_b, pipeline_output, mac_out (mode-independent)
    integer i2;
    always@(posedge CLK_SYS) begin
        if(!RSTN) begin
            for(i2 = 'd0; i2 < NUM_MULT_PARALLEL; i2 = i2 + 'd1) begin
                pipeline_input_b[i2] <= 'sd0;
                pipeline_output[i2] <= 'sd0;
            end
            do_calc_dly <= 1'd0;
            mac_out <= 'sd0;
        end else begin
            do_calc_dly <= DO_CALC;
            if(EN && DO_CALC) begin
                for(i2 = 'd0; i2 < NUM_MULT_PARALLEL; i2 = i2 + 'd1) begin
                    pipeline_input_b[i2] <= IN_DATA[i2 * INPUT_BITWIDTH+: INPUT_BITWIDTH];
                    pipeline_output[i2] <= mult_output[i2];
                end
                mac_out <= (do_load_bias) ? {{(NUM_BITWIDTH_MAC-INPUT_BITWIDTH){IN_BIAS[INPUT_BITWIDTH-'d1]}}, IN_BIAS} : mac_out + sum_pipeline;
            end else begin
                for(i2 = 'd0; i2 < NUM_MULT_PARALLEL; i2 = i2 + 'd1) begin
                    pipeline_input_b[i2] <= 'sd0;
                    pipeline_output[i2] <= 'sd0;
                end
                mac_out <= mac_out;
            end
        end
    end
endmodule
