//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     15.07.2025 11:47:32
// Copied on: 	    §{date_copy_created}
// Module Name:     Dadda-optimized Array Multiplier (signed, 8-bit)
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


module MULT_DADDA_SIGNED_8BIT(
    input wire signed [7:0]		A,
    input wire signed [7:0]    	B,
    output wire signed [15:0] 	Q
);
	localparam DATA_WIDTH = 'd8;
    wire [DATA_WIDTH-'d1:0] IN [DATA_WIDTH-'d1:0];
    wire [5:0] S0, C0;
    wire [13:0] S1, C1;
    wire [9:0] S2, C2;
    wire [11:0] S3, C3;
    wire [14:0] S4, C4;

    //############################## Step 1: Partial Products ##############################
    //Info: Val[x0][y0] --> x0 = Row, y0 = Column
    genvar x0 ,y0;
    for(x0='d0; x0 < DATA_WIDTH; x0=x0+'d1) begin
        for(y0='d0; y0 < DATA_WIDTH; y0=y0+'d1) begin
            if((y0 == DATA_WIDTH-'d1) ^ (x0 == DATA_WIDTH-'d1)) begin
                assign IN[x0][y0] = ~(A[y0] & B[x0]);
            end else begin
                assign IN[x0][y0] = A[y0] & B[x0];
            end   
        end
    end  
    //############################ Step 2: Dadda Tree Reduction ############################
    // Stage 1
    ADDER_LUT_HALF ADD_S00(IN[0][6], IN[1][5],          C0[0], S0[0]);
    ADDER_LUT_FULL ADD_S01(IN[0][7], IN[1][6], IN[2][5],C0[1], S0[1]);
    ADDER_LUT_HALF ADD_S02(IN[3][4], IN[4][3],          C0[2], S0[2]);
    ADDER_LUT_FULL ADD_S03(1'd1, IN[1][7], IN[2][6],    C0[3], S0[3]);
    ADDER_LUT_FULL ADD_S04(IN[3][5], IN[4][4], IN[5][3],C0[4], S0[4]);
    ADDER_LUT_FULL ADD_S05(IN[2][7], IN[3][6], IN[4][5],C0[5], S0[5]);
    
    // Stage 2
    ADDER_LUT_HALF ADD_S10(IN[0][4], IN[1][3],          C1[0], S1[0]);
    ADDER_LUT_FULL ADD_S11(IN[0][5], IN[1][4], IN[2][3],C1[1], S1[1]);
    ADDER_LUT_HALF ADD_S12(IN[3][2], IN[4][1],          C1[2], S1[2]);
    ADDER_LUT_FULL ADD_S13(S0[0], IN[2][4], IN[3][3],   C1[3], S1[3]);
    ADDER_LUT_FULL ADD_S14(IN[4][2], IN[5][1], IN[6][0],C1[4], S1[4]);
    ADDER_LUT_FULL ADD_S15(C0[0], S0[1], S0[2],         C1[5], S1[5]);
    ADDER_LUT_FULL ADD_S16(IN[5][2], IN[6][1], IN[7][0],C1[6], S1[6]);
    ADDER_LUT_FULL ADD_S17(C0[1], C0[2], S0[3],         C1[7], S1[7]);
    ADDER_LUT_FULL ADD_S18(S0[4], IN[6][2], IN[7][1],   C1[8], S1[8]);
    ADDER_LUT_FULL ADD_S19(C0[3], C0[4], S0[5],         C1[9], S1[9]);
    ADDER_LUT_FULL ADD_S1A(IN[5][4], IN[6][3], IN[7][2],C1[10], S1[10]);
    ADDER_LUT_FULL ADD_S1B(C0[5], IN[3][7], IN[4][6],   C1[11], S1[11]);
    ADDER_LUT_FULL ADD_S1C(IN[5][5], IN[6][4], IN[7][3],C1[12], S1[12]);
    ADDER_LUT_FULL ADD_S1D(IN[4][7], IN[5][6], IN[6][5],C1[13], S1[13]);
    
    // Stage 3
    ADDER_LUT_HALF ADD_S20(IN[0][3], IN[1][2],          C2[0], S2[0]);
    ADDER_LUT_FULL ADD_S21(S1[0], IN[2][2], IN[3][1],   C2[1], S2[1]);
    ADDER_LUT_FULL ADD_S22(C1[0], S1[1], S1[2],         C2[2], S2[2]);
    ADDER_LUT_FULL ADD_S23(C1[1], C1[2], S1[3],         C2[3], S2[3]);
    ADDER_LUT_FULL ADD_S24(C1[3], C1[4], S1[5],         C2[4], S2[4]);
    ADDER_LUT_FULL ADD_S25(C1[5], C1[6], S1[7],         C2[5], S2[5]);
    ADDER_LUT_FULL ADD_S26(C1[7], C1[8], S1[9],         C2[6], S2[6]);
    ADDER_LUT_FULL ADD_S27(C1[9], C1[10], S1[11],       C2[7], S2[7]);
    ADDER_LUT_FULL ADD_S28(C1[11], C1[12], S1[13],      C2[8], S2[8]);
    ADDER_LUT_FULL ADD_S29(C1[13], IN[5][7], IN[6][6],  C2[9], S2[9]);
    
    // Stage 4
    ADDER_LUT_HALF ADD_S30(IN[0][2], IN[1][1],          C3[0], S3[0]);
    ADDER_LUT_FULL ADD_S31(S2[0], IN[2][1], IN[3][0],   C3[1], S3[1]);
    ADDER_LUT_FULL ADD_S32(C2[0], S2[1], IN[4][0],      C3[2], S3[2]);
    ADDER_LUT_FULL ADD_S33(C2[1], S2[2], IN[5][0],      C3[3], S3[3]);
    ADDER_LUT_FULL ADD_S34(C2[2], S2[3], S1[4],         C3[4], S3[4]);
    ADDER_LUT_FULL ADD_S35(C2[3], S2[4], S1[6],         C3[5], S3[5]);
    ADDER_LUT_FULL ADD_S36(C2[4], S2[5], S1[8],         C3[6], S3[6]);
    ADDER_LUT_FULL ADD_S37(C2[5], S2[6], S1[10],        C3[7], S3[7]);
    ADDER_LUT_FULL ADD_S38(C2[6], S2[7], S1[12],        C3[8], S3[8]);
    ADDER_LUT_FULL ADD_S39(C2[7], S2[8], IN[7][4],      C3[9], S3[9]);
    ADDER_LUT_FULL ADD_S3A(C2[8], S2[9], IN[7][5],      C3[10], S3[10]);
    ADDER_LUT_FULL ADD_S3B(C2[9], IN[6][7], IN[7][6],   C3[11], S3[11]);
    
    // Stage 5
    ADDER_LUT_HALF ADD_S40(IN[0][1], IN[1][0],          C4[0], S4[0]);
    ADDER_LUT_FULL ADD_S41(S3[0], IN[2][0], C4[0],      C4[1], S4[1]);
    ADDER_LUT_FULL ADD_S42(C3[0], S3[1], C4[1],         C4[2], S4[2]);
    ADDER_LUT_FULL ADD_S43(C3[1], S3[2], C4[2],         C4[3], S4[3]);
    ADDER_LUT_FULL ADD_S44(C3[2], S3[3], C4[3],         C4[4], S4[4]);
    ADDER_LUT_FULL ADD_S45(C3[3], S3[4], C4[4],         C4[5], S4[5]);
    ADDER_LUT_FULL ADD_S46(C3[4], S3[5], C4[5],         C4[6], S4[6]);
    ADDER_LUT_FULL ADD_S47(C3[5], S3[6], C4[6],         C4[7], S4[7]);
    ADDER_LUT_FULL ADD_S48(C3[6], S3[7], C4[7],         C4[8], S4[8]);
    ADDER_LUT_FULL ADD_S49(C3[7], S3[8], C4[8],         C4[9], S4[9]);
    ADDER_LUT_FULL ADD_S4A(C3[8], S3[9], C4[9],         C4[10], S4[10]);
    ADDER_LUT_FULL ADD_S4B(C3[9], S3[10], C4[10],       C4[11], S4[11]);
    ADDER_LUT_FULL ADD_S4C(C3[10], S3[11], C4[11],      C4[12], S4[12]);
    ADDER_LUT_FULL ADD_S4D(IN[7][7], C3[11], C4[12],    C4[13], S4[13]);
    ADDER_LUT_HALF ADD_S4E(1'd1, C4[13],                C4[14], S4[14]);
    
    //############################## Step 3: OUTPUT ##############################
    assign Q = {S4[14:0], IN[0][0]};

endmodule
