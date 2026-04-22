//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     15.07.2025 10:52:57
// Copied on: 	    §{date_copy_created}
// Module Name:     Dadda-optimized Array Multiplier (signed, 4-bit)
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

module MULT_DADDA_SIGNED_4BIT(
    input wire signed [3:0]		A,
    input wire signed [3:0]    	B,
    output wire signed [7:0] 	Q
);
	localparam DATA_WIDTH = 'd4;
    wire [DATA_WIDTH-'d1:0] IN [DATA_WIDTH-'d1:0];
    wire [1:0] S0, C0;
    wire [4:0] S1, C1;
    wire [5:0] S2, C2;

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
    // Stage 1
    ADDER_LUT_HALF ADD_S00(IN[0][3], IN[1][2],          C0[0], S0[0]);
    ADDER_LUT_FULL ADD_S01(1'd1, IN[1][3], IN[2][2],    C0[1], S0[1]);
    
    // Stage 2
    ADDER_LUT_HALF ADD_S10(IN[0][1], IN[1][0],          C1[0], S1[0]);
    ADDER_LUT_FULL ADD_S11(IN[0][2], IN[1][1], IN[2][0],C1[1], S1[1]);
    ADDER_LUT_FULL ADD_S12(S0[0], IN[2][1], IN[3][0],   C1[2], S1[2]);
    ADDER_LUT_FULL ADD_S13(S0[1], C0[0], IN[3][1],      C1[3], S1[3]);
    ADDER_LUT_FULL ADD_S14(IN[2][3], IN[3][2], C0[1],   C1[4], S1[4]);
    
    //Stage 3
    ADDER_LUT_HALF ADD_S20(S1[1], C1[0],            C2[0], S2[0]);
    ADDER_LUT_FULL ADD_S21(S1[2], C1[1], C2[0],     C2[1], S2[1]);
    ADDER_LUT_FULL ADD_S22(S1[3], C1[2], C2[1],     C2[2], S2[2]);
    ADDER_LUT_FULL ADD_S23(S1[4], C1[3], C2[2],     C2[3], S2[3]);
    ADDER_LUT_FULL ADD_S24(~IN[3][3], C1[4], C2[3], C2[4], S2[4]);
    ADDER_LUT_HALF ADD_S25(1'd1, C2[4],             C2[5], S2[5]);
    
    //############################## Step 3: OUTPUT ##############################
    assign Q = {S2, S1[0], IN[0][0]};

endmodule
