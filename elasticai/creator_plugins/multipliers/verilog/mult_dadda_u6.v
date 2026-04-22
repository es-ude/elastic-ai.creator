//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     13.11.2020 09:22:46
// Copied on: 	    §{date_copy_created}
// Module Name:     Dadda-optimized Array Multiplier (unsigned, 6-bit)
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

module MULT_DADDA_UNSIGNED_6BIT(
    input wire [5:0]    A,
    input wire [5:0]    B,
    output wire [11:0] Q
);
	localparam DATA_WIDTH = 6;
    wire [DATA_WIDTH-'d1:0] IN [DATA_WIDTH-'d1:0];
    wire [5:0] S00;     wire [5:0] C00;
    wire [5:0] S10;     wire [5:0] C10;
    wire [7:0] S20;     wire [7:0] C20;
                        wire [8:0] C30;
        
    //############################## Step 1: Partial Products ##############################
    //Info: Val[x0][y0] --> x0 = Row, y0 = Column
    genvar x0 ,y0;
    for(x0=0; x0<DATA_WIDTH; x0=x0+1) begin
        for(y0=0; y0<DATA_WIDTH; y0=y0+1) begin
            //if(y0 == (DATA_WIDTH-'d1)) begin
                //assign IN[x0][y0] = ~(A[y0] & B[x0]);
            //end else begin
                assign IN[x0][y0] = A[y0] & B[x0];
            //end
        end
    end  
    //############################## Step 2: Dadda Tree Reduction ##############################
    // Stage 1
    ADDER_LUT_HALF ADD_S00(IN[0][4], IN[1][3],            C00[0], S00[0]);
    ADDER_LUT_FULL ADD_S01(IN[0][5], IN[1][4], IN[2][3],  C00[1], S00[1]);
    ADDER_LUT_FULL ADD_S02(IN[1][5], IN[2][4], IN[3][3],  C00[2], S00[2]);
    ADDER_LUT_FULL ADD_S03(IN[2][5], IN[3][4], IN[4][3],  C00[3], S00[3]);
    ADDER_LUT_HALF ADD_S04(IN[3][2], IN[4][1],            C00[4], S00[4]);
    ADDER_LUT_HALF ADD_S05(IN[4][2], IN[5][1],            C00[5], S00[5]);
    
    //Stage 2
    ADDER_LUT_HALF ADD_S10(IN[0][3], IN[1][2],            C10[0], S10[0]);
    ADDER_LUT_FULL ADD_S11(S00[0], IN[2][2], IN[3][1],    C10[1], S10[1]);
    ADDER_LUT_FULL ADD_S12(S00[1], C00[0], S00[4],        C10[2], S10[2]);
    ADDER_LUT_FULL ADD_S13(S00[2], C00[1], S00[5],        C10[3], S10[3]);
    ADDER_LUT_FULL ADD_S14(S00[3], C00[2], C00[5],        C10[4], S10[4]);
    ADDER_LUT_FULL ADD_S15(IN[3][5], C00[3], IN[4][4],    C10[5], S10[5]);
    
    //Stage 3
    ADDER_LUT_HALF ADD_S20(IN[0][2], IN[1][1],            C20[0], S20[0]);
    ADDER_LUT_FULL ADD_S21(S10[0], IN[2][1], IN[3][0],    C20[1], S20[1]);
    ADDER_LUT_FULL ADD_S22(S10[1], C10[0], IN[4][0],      C20[2], S20[2]);
    ADDER_LUT_FULL ADD_S23(S10[2], C10[1], IN[5][0],      C20[3], S20[3]);
    ADDER_LUT_FULL ADD_S24(S10[3], C10[2], C00[4],        C20[4], S20[4]);
    ADDER_LUT_FULL ADD_S25(S10[4], C10[3], IN[5][2],      C20[5], S20[5]);
    ADDER_LUT_FULL ADD_S26(S10[5], C10[4], IN[5][3],      C20[6], S20[6]);
    ADDER_LUT_FULL ADD_S27(IN[4][5], C10[5], IN[5][4],    C20[7], S20[7]);
    
    //############################## Step 3: OUTPUT ##############################
    assign Q[0] = IN[0][0];
    ADDER_LUT_HALF ADD_Q1(IN[1][0], IN[0][1],             C30[0], Q[1]);
    ADDER_LUT_FULL ADD_Q2(S20[0], IN[2][0], C30[0],       C30[1], Q[2]);
    genvar i0;
    for(i0 = 0; i0 <= DATA_WIDTH; i0=i0+'d1) begin
        ADDER_LUT_FULL ADD_QX(S20[i0+'d1], C20[i0], C30[i0+'d1], C30[i0+'d2], Q[i0+'d3]);
    end
    ADDER_LUT_FULL ADD_Q11(IN[DATA_WIDTH-'d1][DATA_WIDTH-'d1], C20[DATA_WIDTH+'d1], C30[DATA_WIDTH+'d2], Q[2*DATA_WIDTH-'d1], Q[2*DATA_WIDTH-'d2]);
    
endmodule
