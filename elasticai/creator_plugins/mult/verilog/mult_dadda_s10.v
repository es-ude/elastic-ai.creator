//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     18.07.2025 18:38:32
// Copied on: 	    §{date_copy_created}
// Module Name:     Dadda-optimized Array Multiplier (signed, 10-bit)
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


module MULT_DADDA_SIGNED_10BIT(
    input wire signed [9:0]		A,
    input wire signed [9:0]    	B,
    output wire signed [19:0] 	Q
);
	localparam DATA_WIDTH = 'd10;
    wire [DATA_WIDTH-'d1:0] IN [DATA_WIDTH-'d1:0];
    wire [1:0] S0, C0;
    wire [17:0] S1, C1;
    wire [21:0] S2, C2;
    wire [13:0] S3, C3;
    wire [16:0] S4, C4;
    wire [17:0] S5, C5;

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
    ADDER_LUT_HALF ADD_S00(IN[0][9], IN[1][8],          C0[0], S0[0]);
    ADDER_LUT_FULL ADD_S01(1'd1, IN[1][9], IN[2][8],    C0[1], S0[1]);
    
    // Stage 2
    ADDER_LUT_HALF ADD_S10(IN[0][6], IN[1][5],          C1[0], S1[0]);
    ADDER_LUT_FULL ADD_S11(IN[0][7], IN[1][6], IN[2][5],C1[1], S1[1]);
    ADDER_LUT_HALF ADD_S12(IN[3][4], IN[4][3],          C1[2], S1[2]);
    ADDER_LUT_FULL ADD_S13(IN[0][8], IN[1][7], IN[2][6],C1[3], S1[3]);
    ADDER_LUT_FULL ADD_S14(IN[3][5], IN[4][4], IN[5][3],C1[4], S1[4]);
    ADDER_LUT_HALF ADD_S15(IN[6][2], IN[7][1],          C1[5], S1[5]);
    ADDER_LUT_FULL ADD_S16(S0[0], IN[2][7], IN[3][6],   C1[6], S1[6]);
    ADDER_LUT_FULL ADD_S17(IN[4][5],IN[5][4], IN[6][3], C1[7], S1[7]);
    ADDER_LUT_FULL ADD_S18(IN[7][2], IN[8][1], IN[9][0],C1[8], S1[8]);
    ADDER_LUT_FULL ADD_S19(C0[0], S0[1], IN[3][7],      C1[9], S1[9]);
    ADDER_LUT_FULL ADD_S1A(IN[4][6], IN[5][5], IN[6][4],C1[10], S1[10]);
    ADDER_LUT_FULL ADD_S1B(IN[7][3], IN[8][2], IN[9][1],C1[11], S1[11]);
    ADDER_LUT_FULL ADD_S1C(C0[1], IN[2][9], IN[3][8],   C1[12], S1[12]);
    ADDER_LUT_FULL ADD_S1D(IN[4][7], IN[5][6], IN[6][5],C1[13], S1[13]);
    ADDER_LUT_FULL ADD_S1E(IN[7][4], IN[8][3], IN[9][2],C1[14], S1[14]);
    ADDER_LUT_FULL ADD_S1F(IN[3][9], IN[4][8], IN[5][7],C1[15], S1[15]);
    ADDER_LUT_FULL ADD_S1G(IN[6][6], IN[7][5], IN[8][4],C1[16], S1[16]);
    ADDER_LUT_FULL ADD_S1H(IN[4][9], IN[5][8], IN[6][7],C1[17], S1[17]);
    
    // Stage 3
    ADDER_LUT_HALF ADD_S20(IN[0][4], IN[1][3],          C2[0], S2[0]);
    ADDER_LUT_FULL ADD_S21(IN[0][5], IN[1][4], IN[2][3],C2[1], S2[1]);
    ADDER_LUT_HALF ADD_S22(IN[3][2], IN[4][1],          C2[2], S2[2]);
    ADDER_LUT_FULL ADD_S23(S1[0], IN[2][4], IN[3][3],   C2[3], S2[3]);
    ADDER_LUT_FULL ADD_S24(IN[4][2], IN[5][1], IN[6][0],C2[4], S2[4]);
    ADDER_LUT_FULL ADD_S25(C1[0], S1[1], S1[2],         C2[5], S2[5]);
    ADDER_LUT_FULL ADD_S26(IN[5][2], IN[6][1], IN[7][0],C2[6], S2[6]);
    ADDER_LUT_FULL ADD_S27(C1[1], C1[2], S1[3],         C2[7], S2[7]);
    ADDER_LUT_FULL ADD_S28(S1[4], S1[5], IN[8][0],      C2[8], S2[8]);
    ADDER_LUT_FULL ADD_S29(C1[3], C1[4], C1[5],         C2[9], S2[9]);
    ADDER_LUT_FULL ADD_S2A(S1[6], S1[7], S1[8],         C2[10], S2[10]);
    ADDER_LUT_FULL ADD_S2B(C1[6], C1[7], C1[8],         C2[11], S2[11]);
    ADDER_LUT_FULL ADD_S2C(S1[9], S1[10], S1[11],       C2[12], S2[12]);
    ADDER_LUT_FULL ADD_S2D(C1[9], C1[10], C1[11],       C2[13], S2[13]);
    ADDER_LUT_FULL ADD_S2E(S1[12], S1[13], S1[14],      C2[14], S2[14]);
    ADDER_LUT_FULL ADD_S2F(C1[12], C1[13], C1[14],      C2[15], S2[15]);
    ADDER_LUT_FULL ADD_S2G(S1[15], S1[16], IN[9][3],    C2[16], S2[16]);
    ADDER_LUT_FULL ADD_S2H(C1[15], C1[16], S1[17],      C2[17], S2[17]);
    ADDER_LUT_FULL ADD_S2I(IN[7][6], IN[8][5], IN[9][4],C2[18], S2[18]);
    ADDER_LUT_FULL ADD_S2J(C1[17], IN[5][9], IN[6][8],  C2[19], S2[19]);
    ADDER_LUT_FULL ADD_S2K(IN[7][7], IN[8][6], IN[9][5],C2[20], S2[20]);
    ADDER_LUT_FULL ADD_S2L(IN[6][9], IN[7][8], IN[8][7],C2[21], S2[21]);
    
    // Stage 4
    ADDER_LUT_HALF ADD_S30(IN[0][3], IN[1][2],          C3[0], S3[0]);
    ADDER_LUT_FULL ADD_S32(S2[0], IN[2][2], IN[3][1],   C3[1], S3[1]);
    ADDER_LUT_FULL ADD_S33(C2[0], S2[1], S2[2],         C3[2], S3[2]);
    ADDER_LUT_FULL ADD_S34(C2[1], C2[2], S2[3],         C3[3], S3[3]);
    ADDER_LUT_FULL ADD_S35(C2[3], C2[4], S2[5],         C3[4], S3[4]);
    ADDER_LUT_FULL ADD_S36(C2[5], C2[6], S2[7],         C3[5], S3[5]);
    ADDER_LUT_FULL ADD_S37(C2[7], C2[8], S2[9],         C3[6], S3[6]);
    ADDER_LUT_FULL ADD_S38(C2[9], C2[10], S2[11],       C3[7], S3[7]);
    ADDER_LUT_FULL ADD_S39(C2[11], C2[12], S2[13],      C3[8], S3[8]);
    ADDER_LUT_FULL ADD_S3A(C2[13], C2[14], S2[15],      C3[9], S3[9]);
    ADDER_LUT_FULL ADD_S3B(C2[15], C2[16], S2[17],      C3[10], S3[10]);
    ADDER_LUT_FULL ADD_S3C(C2[17], C2[18], S2[19],      C3[11], S3[11]);
    ADDER_LUT_FULL ADD_S3D(C2[19], C2[20], S2[21],      C3[12], S3[12]);
    ADDER_LUT_FULL ADD_S3E(C2[21], IN[7][9], IN[8][8],  C3[13], S3[13]);
    
    // Stage 5
    ADDER_LUT_HALF ADD_S40(IN[0][1], IN[1][0],          C4[0], S4[0]);
    ADDER_LUT_FULL ADD_S41(IN[0][2], IN[1][1], IN[2][0],C4[1], S4[1]);
    ADDER_LUT_FULL ADD_S42(S3[0], IN[2][1], IN[3][0],   C4[2], S4[2]);
    ADDER_LUT_FULL ADD_S43(C3[0], S3[1], IN[4][0],      C4[3], S4[3]);
    ADDER_LUT_FULL ADD_S44(C3[1], S3[2], IN[5][0],      C4[4], S4[4]);
    ADDER_LUT_FULL ADD_S45(C3[2], S3[3], S2[4],         C4[5], S4[5]);
    ADDER_LUT_FULL ADD_S46(C3[3], S3[4], S2[6],         C4[6], S4[6]);
    ADDER_LUT_FULL ADD_S47(C3[4], S3[5], S2[8],         C4[7], S4[7]);
    ADDER_LUT_FULL ADD_S48(C3[5], S3[6], S2[10],        C4[8], S4[8]);
    ADDER_LUT_FULL ADD_S49(C3[6], S3[7], S2[12],        C4[9], S4[9]);
    ADDER_LUT_FULL ADD_S4A(C3[7], S3[8], S2[14],        C4[10], S4[10]);
    ADDER_LUT_FULL ADD_S4B(C3[8], S3[9], S2[16],        C4[11], S4[11]);
    ADDER_LUT_FULL ADD_S4C(C3[9], S3[10], S2[18],       C4[12], S4[12]);
    ADDER_LUT_FULL ADD_S4D(C3[10], S3[11], S2[20],      C4[13], S4[13]);
    ADDER_LUT_FULL ADD_S4E(C3[11], S3[12], IN[9][6],    C4[14], S4[14]);
    ADDER_LUT_FULL ADD_S4F(C3[12], S3[13], IN[9][7],    C4[15], S4[15]);
    ADDER_LUT_FULL ADD_S4G(C3[13], IN[8][9], IN[9][8],  C4[16], S4[16]);
    
    // Stage 6
    ADDER_LUT_HALF ADD_S50(C4[0], S4[1],                C5[0], S5[0]);
    ADDER_LUT_FULL ADD_S51(C4[1], S4[2], C5[0],         C5[1], S5[1]);
    ADDER_LUT_FULL ADD_S52(C4[2], S4[3], C5[1],         C5[2], S5[2]);
    ADDER_LUT_FULL ADD_S53(C4[3], S4[4], C5[2],         C5[3], S5[3]);
    ADDER_LUT_FULL ADD_S54(C4[4], S4[5], C5[3],         C5[4], S5[4]);
    ADDER_LUT_FULL ADD_S55(C4[5], S4[6], C5[4],         C5[5], S5[5]);
    ADDER_LUT_FULL ADD_S56(C4[6], S4[7], C5[5],         C5[6], S5[6]);
    ADDER_LUT_FULL ADD_S57(C4[7], S4[8], C5[6],         C5[7], S5[7]);
    ADDER_LUT_FULL ADD_S58(C4[8], S4[9], C5[7],         C5[8], S5[8]);
    ADDER_LUT_FULL ADD_S59(C4[9], S4[10], C5[8],        C5[9], S5[9]);
    ADDER_LUT_FULL ADD_S5A(C4[10], S4[11], C5[9],       C5[10], S5[10]);
    ADDER_LUT_FULL ADD_S5B(C4[11], S4[12], C5[10],      C5[11], S5[11]);
    ADDER_LUT_FULL ADD_S5C(C4[12], S4[13], C5[11],      C5[12], S5[12]);
    ADDER_LUT_FULL ADD_S5D(C4[13], S4[14], C5[12],      C5[13], S5[13]);
    ADDER_LUT_FULL ADD_S5E(C4[14], S4[15], C5[13],      C5[14], S5[14]);
    ADDER_LUT_FULL ADD_S5F(C4[15], S4[16], C5[14],      C5[15], S5[15]);
    ADDER_LUT_FULL ADD_S5G(C4[16], IN[9][9], C5[15],    C5[16], S5[16]);
    ADDER_LUT_HALF ADD_S5H(1'd1, C5[16],                C5[17], S5[17]);
    
    //############################## Step 3: OUTPUT ##############################
    assign Q = {S5, S4[0], IN[0][0]};

endmodule
