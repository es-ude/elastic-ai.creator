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
//////////////////////////////////////////////////////////////////////////////////


module ACT_PRECOMPUTED#(
    parameter integer BITWIDTH_IN = 4,
    parameter integer BITWIDTH_OUT = 4
)(
    input wire signed [BITWIDTH_IN-1:0] A,
    output wire signed [BITWIDTH_OUT-1:0] Q
);

    localparam integer NUM_VALUES = 4;
    localparam signed [NUM_VALUES * BITWIDTH_OUT-1:0] PRECOMPUTED = { 4'sd0, 4'sd1, 4'sd2, 4'sd3 };
    localparam integer ADDRWIDTH = $clog2(NUM_VALUES);

    wire [ADDRWIDTH-1:0] selector;
    wire signed [BITWIDTH_OUT-1:0] lut_func [NUM_VALUES-1:0];

    // Slicing vector into array
    genvar k0;
    for(k0 = 0; k0 < NUM_VALUES; k0 = k0 + 1) begin
        assign lut_func[NUM_VALUES - 1 - k0] = PRECOMPUTED[k0*BITWIDTH_OUT+:BITWIDTH_OUT];
    end

    assign selector = {~A[BITWIDTH_IN-1], A[(BITWIDTH_IN-2)-:(ADDRWIDTH-1)]};
    assign Q = lut_func[selector];
endmodule
