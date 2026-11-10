//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     21.01.2026, 11:21
// Copied on: 	    §{date_copy_created}
// Module Name:     Precomputed Activation Function used in DNN
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v1
// Processing:      LUT-based processing
// Dependencies:    None
//
// State: 	        Not tested!
// Improvements:    None
// Parameters:      BITWIDTH_IN     --> Bitwidth of input data
//                  BITWIDTH_OUT    --> Bitwidth of output data
//                  NUM_INPUTS      --> Number of in/outputs
//////////////////////////////////////////////////////////////////////////////////


module ACT_PRECOMPUTED#(
    parameter integer BITWIDTH_IN = 4,
    parameter integer BITWIDTH_OUT = 4,
    parameter integer NUM_INPUTS = 1
)(
    input wire signed [NUM_INPUTS*BITWIDTH_IN-1:0] A,
    output wire signed [NUM_INPUTS*BITWIDTH_OUT-1:0] Q
);

    localparam integer NUM_VALUES = 4;
    localparam signed [NUM_VALUES * BITWIDTH_OUT-1:0] VALUES_Y = { 4'sd0, 4'sd1, 4'sd2, 4'sd3 };
    localparam integer ADDRWIDTH = $clog2(NUM_VALUES);

    // --- LUT with activation function
    wire signed [BITWIDTH_OUT-1:0] lut_func [NUM_VALUES-1:0];
    genvar k0;
    for(k0 = 0; k0 < NUM_VALUES; k0 = k0 + 1) begin
        assign lut_func[NUM_VALUES - 1 - k0] = VALUES_Y[k0*BITWIDTH_OUT+:BITWIDTH_OUT];
    end

    genvar k;
    for(k = 0; k < NUM_INPUTS; k = k + 1) begin
        wire [ADDRWIDTH-1:0] selector;
        assign selector = {~A[(k+1)*BITWIDTH_IN-1], A[((k+1)*BITWIDTH_IN-2)-:(ADDRWIDTH-1)]};
        assign Q[k*BITWIDTH_OUT+:BITWIDTH_OUT] = lut_func[selector];
    end
endmodule
