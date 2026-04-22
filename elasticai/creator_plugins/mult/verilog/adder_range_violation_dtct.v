//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     30.07.2025, 14:30:14
// Copied on: 	    §{date_copy_created}
// Module Name:     Adder Range Violation Detector
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v0
// Processing:      Checking the adder output (if the bitwidth will be increased due to number of adders)
//                  if the results is inside the range before pruning (Data valid)
//
// State: 	        Not tested
// Dependencies:    None
// Improvements:    None
// Parameters:      BITWIDTH    --> Number of bitwidth of the adders results
//                  NUM_ADDERS  --> Number of parallel adders -1 (=0, no adders, 1= one adder)
//                  SIGNED      --> Is the result signed or unsigned
//////////////////////////////////////////////////////////////////////////////////


module ADDER_RANGE_DTCT#(
    parameter BITWIDTH = 6'd8,
    parameter NUM_ADDERS = 4'd1,
    parameter SIGNED = 1'd0
)(
    input wire [BITWIDTH-'d1:0] A,
    output wire UPPER_LIMIT,
    output wire DOWNER_LIMIT,
    output wire DATA_VALID
);
    localparam NUM_BITS = (NUM_ADDERS > 0) ? NUM_ADDERS : 1'd1;

    wire [NUM_BITS-'d1:0] extracted [2:0];
    generate
        if (NUM_ADDERS > 0) begin
            assign extracted[0] = A[BITWIDTH-'d2-:NUM_ADDERS];
            assign extracted[1] = A[BITWIDTH-'d1-:NUM_ADDERS];
            assign extracted[2] = A[BITWIDTH-'d2-:NUM_ADDERS];
        end else begin
            assign extracted[0] = 1'd0;
            assign extracted[1] = 1'd0;
            assign extracted[2] = 1'd1;
        end
    endgenerate

    assign UPPER_LIMIT = (SIGNED) ? (!A[BITWIDTH-'d1] && |extracted[0]) : |extracted[1];
    assign DOWNER_LIMIT = (SIGNED) ? (A[BITWIDTH-'d1] && ~&extracted[2]) : 1'd0;
    assign DATA_VALID = !(UPPER_LIMIT || DOWNER_LIMIT);
                            
endmodule
