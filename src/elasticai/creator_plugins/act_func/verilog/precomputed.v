//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     21.01.2026, 11:21
// Copied on: 	    §{date_copy_created}
// Module Name:     Precomputed Activation Function used in DNN
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v0
// Processing:      LUT-based processing
// Dependencies:    None
//
// State: 	        Not tested!
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input data
//                  NUM_VALUES --> Number of precomputed ACT values
//////////////////////////////////////////////////////////////////////////////////


module ACT_PRECOMPUTED#(
    parameter BITWIDTH = 5'd4,
    parameter NUM_VALUES = 5'd4
)(
    input wire signed [BITWIDTH-'d1:0] A,
    output wire signed [BITWIDTH-'d1:0] Q
);
    localparam ADDRWIDTH = $clog2(NUM_VALUES);
    wire [ADDRWIDTH-'d1:0] selector;

    // Values of precomputed activation function
    localparam signed [NUM_VALUES * BITWIDTH-'d1:0] PRECOMPUTED = { 4'sd0, 4'sd1, 4'sd2, 4'sd3 };
    wire signed [BITWIDTH-'d1:0] lut_func [NUM_VALUES-'d1:0];
    // Slicing vector into array
    genvar k0;
    for(k0 = 'd0; k0 < NUM_VALUES; k0 = k0 + 'd1) begin
        assign lut_func[NUM_VALUES - 'd1 - k0] = PRECOMPUTED[k0*BITWIDTH+:BITWIDTH];
    end

    assign selector = {~A[BITWIDTH-'d1], A[(BITWIDTH-'d2)-:(ADDRWIDTH-'d1)]};
    assign Q = lut_func[selector];
endmodule
