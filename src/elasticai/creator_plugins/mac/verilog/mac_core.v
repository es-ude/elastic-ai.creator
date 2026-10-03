//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
// 
// Create Date:     17.01.2025 08:11:51
// Copied on: 	    §{date_copy_created}
// Module Name:     Computation Core for Multiply-Accumulate Operator
// Target Devices:  FPGA / ASIC (call LUT-based multiplier with custom integration)
// Tool Versions:   1v1
// Description:     Performing a MAC Operation on Device (with Clamping, Pipelined Multiplier and Parallization)
// Processing:      Data applied on posedge clk
//                  First cycle with DO_CALC --> Reset MAC output, add bias and stream data into pipeline
//                  After first cycle --> pipelined MAC operation
// Dependencies:    If you want to have a high-speed version, enable the output pipeline buffer to achieve higher clocks (latency += 1 cycle)
//
// State: 	        Works!
// Improvements:    Add a bias on output to reduce truncating errors
// Parameters:      BITWIDTH        --> Bitwidth of input data
//                  NUM_MULT        --> Number of used multiplier in parallel
//                  BITS_OVERSIZE   --> Number of bits to oversize the sum unit
//                  BITS_SCALE_BIAS --> Number of bits to scale the bias up to MAC (to apply FxP)
//                  BITS_SCALE_DOUT --> Bits for right-shifting the output value (to apply FxP)
//////////////////////////////////////////////////////////////////////////////////


module MAC_CORE#(
    parameter integer BITWIDTH = 8,
    parameter integer NUM_MULT = 2,
    parameter integer BITS_OVERSIZE = 1,
    parameter integer BITS_SCALE_BIAS = 0,
    parameter integer BITS_SCALE_DOUT = 0
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire DO_CALC,
    input wire DO_CLEAR,
    input wire signed [BITWIDTH-1:0] IN_BIAS,
    input wire signed [BITWIDTH * NUM_MULT-1:0] IN_WEIGHTS,
    input wire signed [BITWIDTH * NUM_MULT-1:0] IN_DATA,
    output wire signed [BITWIDTH-1:0] OUT_DATA
);
    // --- Local parameter for configuring the pipeline and parallelization of MAC
    localparam integer NUM_BITWIDTH_MAC = 2* BITWIDTH + BITS_OVERSIZE;
    localparam integer BITS_BIAS_UPPER = NUM_BITWIDTH_MAC - BITWIDTH - BITS_SCALE_BIAS;

    // --- Definition of internal signals and register
    reg signed [BITWIDTH-1:0] pipeline_input_a [NUM_MULT-1:0];
    reg signed [BITWIDTH-1:0] pipeline_input_b [NUM_MULT-1:0];
    //reg signed [2* BITWIDTH-1:0] pipeline_output [NUM_MULT-1:0];
    wire signed [2* BITWIDTH-1:0] mult_output [NUM_MULT-1:0];
    reg signed [NUM_BITWIDTH_MAC-1:0] mac_out;
    reg signed [NUM_BITWIDTH_MAC-1:0] sum_pipeline;
    wire is_overflow, is_underflow;

    assign is_overflow = !mac_out[NUM_BITWIDTH_MAC-1] && |mac_out[NUM_BITWIDTH_MAC-2:BITWIDTH+BITS_SCALE_DOUT-1];
    assign is_underflow = mac_out[NUM_BITWIDTH_MAC-1] && ~&mac_out[NUM_BITWIDTH_MAC-2:BITWIDTH+BITS_SCALE_DOUT-1];

    // --- Clamping output data
    assign OUT_DATA =   (is_overflow) ? {1'b0, {(BITWIDTH-1){1'b1}}} :
                        ((is_underflow) ? {1'b1, {(BITWIDTH-1){1'b0}}} :
                        {mac_out[NUM_BITWIDTH_MAC-1], mac_out[BITS_SCALE_DOUT+:(BITWIDTH-1)]});
    // --- Using multiplier
    genvar k0;
    generate
        for(k0 = 'd0; k0 < NUM_MULT; k0 = k0 + 'd1) begin
            MULT_SIGNED#(BITWIDTH) MULT_UNIT(
                .A(pipeline_input_a[k0]),
                .B(pipeline_input_b[k0]),
                .Q(mult_output[k0])
            );
        end
    endgenerate
    // --- Adder Tree
    integer k1;
    always@(*) begin
        for (k1 = 'd0; k1 < NUM_MULT; k1 = k1 + 'd1) begin
            if(k1 == 'd0) begin
                sum_pipeline = mult_output[k1];
            end else begin
                sum_pipeline = sum_pipeline + mult_output[k1];
            end
        end
    end
    // --- Control device for pipeline multiplication
    integer i0;
    always@(posedge CLK_SYS) begin
        if(!RSTN) begin
            for(i0 = 'd0; i0 < NUM_MULT; i0 = i0 + 'd1) begin
                pipeline_input_a[i0] <= 'sd0;
                pipeline_input_b[i0] <= 'sd0;
                //pipeline_output[i0] <= 'sd0;
            end
            mac_out <= 'sd0;
        end else begin
            if(EN && DO_CALC) begin
                // --- State: Do Calculation
                for(i0 = 'd0; i0 < NUM_MULT; i0 = i0 + 'd1) begin
                    pipeline_input_a[i0] <= IN_WEIGHTS[i0 * BITWIDTH+: BITWIDTH];
                    pipeline_input_b[i0] <= IN_DATA[i0 * BITWIDTH+: BITWIDTH];
                    //pipeline_output[i0] <= mult_output[i0];
                end
                mac_out <= ((DO_CLEAR) ? {{(BITS_BIAS_UPPER){IN_BIAS[BITWIDTH-'d1]}}, IN_BIAS, {(BITS_SCALE_BIAS){1'b0}}} : mac_out) + sum_pipeline;
            end else begin
                // --- State: Hold data
                for(i0 = 'd0; i0 < NUM_MULT; i0 = i0 + 'd1) begin
                    pipeline_input_a[i0] <= 'sd0;
                    pipeline_input_b[i0] <= 'sd0;
                    //pipeline_output[i0] <= mult_output[i0];
                end
                mac_out <= mac_out;
            end
        end
    end
endmodule
