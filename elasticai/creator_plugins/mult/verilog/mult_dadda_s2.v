//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     15.07.2025 10:11:47
// Copied on: 	    §{date_copy_created}
// Module Name:     Dadda-optimized Array Multiplier (signed, 2-bit)
// Target Devices:  ASIC, FPGA
// Tool Versions:   1v0
// Description:     LUT-based multiplication
// Processing:      Direct processing
// Dependencies:    None
//
// State: 	        Not tested on hardware!
// Dependency:      ADDER_LUT_FULL, ADDER_LUT_HALF
// Improvements:    None
// Parameters:      None
//
//////////////////////////////////////////////////////////////////////////////////

module MULT_DADDA_SIGNED_2BIT(
    input wire signed [1:0]		A,
    input wire signed [1:0]    	B,
    output wire signed [3:0] 	Q
);
	localparam DATA_WIDTH = 'd2;
    wire [DATA_WIDTH-'d1:0] IN [DATA_WIDTH-'d1:0];
    wire S00, C00;
    wire S10, C10;
    wire S20, C20;

    //############################## Step 1: Partial Products ##############################
    //Info: Val[x0][y0] --> x0 = Row, y0 = Column
    genvar x0 ,y0;
    for(x0='d0; x0 < DATA_WIDTH; x0=x0+'d1) begin
        for(y0='d0; y0 < DATA_WIDTH; y0=y0+'d1) begin
            if((y0 == DATA_WIDTH-'d1)||(x0 == DATA_WIDTH-'d1)) begin
                assign IN[x0][y0] = ~(A[y0] & B[x0]);
            end else begin
                assign IN[x0][y0] = A[y0] & B[x0];
            end   
        end
    end  
    //############################## Step 2: Dadda Tree Reduction ##############################
    ADDER_LUT_HALF ADD_S00(IN[0][1], IN[1][0],  C00, S00);
    ADDER_LUT_FULL ADD_S10(1'd1, ~IN[1][1], C00, C10, S10);
    ADDER_LUT_HALF ADD_S20(1'd1, C10,           C20, S20);
    
    //############################## Step 3: OUTPUT ##############################
    assign Q = {S20, S10, S00, IN[0][0]};

endmodule
