//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     13.11.2020 09:22:46
// Copied on: 	    §{date_copy_created}
// Module Name:     Dadda-optimized Array Multiplier (signed, 12-bit)
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

module MULT_DADDA_SIGNED_12BIT(
    input wire signed [11:0]    A,
    input wire signed [11:0]    B,
    output wire signed [23:0] Q
);
	localparam DATA_WIDTH = 12;
	wire [DATA_WIDTH-'d1:0] IN [DATA_WIDTH-'d1:0];
    wire [DATA_WIDTH-'d1:0] S00;    wire [DATA_WIDTH-'d1:0] C00;
    wire [29:0] S10;                wire [29:0] C10;
    wire [29:0] S20;                wire [29:0] C20;
    wire [17:0] S30;                wire [17:0] C30;
    wire [19:0] S40;                wire [19:0] C40;
                                    wire [2*DATA_WIDTH-'d3:0] C50;
        
    //############################## Step 1: Partial Products ##############################
    //Info: Val[x0][y0] --> x0 = Row, y0 = Column
    genvar x0 ,y0;
    for(x0=0; x0<DATA_WIDTH; x0=x0+1) begin
        for(y0=0; y0<DATA_WIDTH; y0=y0+1) begin
            if((y0 == DATA_WIDTH-'d1)||(x0 == DATA_WIDTH-'d1)) begin
                assign IN[x0][y0] = ~(A[y0] & B[x0]);
            end else begin
                assign IN[x0][y0] = A[y0] & B[x0];
            end   
        end
    end  
    //############################## Step 2: Dadda Tree Reduction ##############################
    // Stage 1
    ADDER_LUT_HALF ADD_S00(IN[0][9],  IN[1][8],               C00[0], S00[0]);
    ADDER_LUT_FULL ADD_S01(IN[0][10], IN[1][9],   IN[2][8],   C00[1], S00[1]);
    ADDER_LUT_FULL ADD_S02(IN[0][11], IN[1][10],  IN[2][9],   C00[2], S00[2]);
    ADDER_LUT_FULL ADD_S03(1'b1,      IN[1][11],  IN[2][10],  C00[3], S00[3]);
    ADDER_LUT_FULL ADD_S04(IN[2][11], IN[3][10],  IN[4][9],   C00[4], S00[4]);
    ADDER_LUT_FULL ADD_S05(IN[3][11], IN[4][10],  IN[5][9],   C00[5], S00[5]);
    ADDER_LUT_HALF ADD_S06(IN[3][7],  IN[4][6],               C00[6], S00[6]);
    ADDER_LUT_FULL ADD_S07(IN[3][8],  IN[4][7],   IN[5][6],   C00[7], S00[7]);
    ADDER_LUT_FULL ADD_S08(IN[3][9],  IN[4][8],   IN[5][7],   C00[8], S00[8]);
    ADDER_LUT_FULL ADD_S09(IN[5][8],  IN[6][7],   IN[7][6],   C00[9], S00[9]);
    ADDER_LUT_HALF ADD_S0A(IN[6][5],  IN[7][4],               C00[10], S00[10]);
    ADDER_LUT_FULL ADD_S0B(IN[6][6],  IN[7][5],   IN[8][4],   C00[11], S00[11]);
    
    //Stage 2
    ADDER_LUT_HALF ADD_S10(IN[0][6],  IN[1][5],               C10[0], S10[0]);
    ADDER_LUT_FULL ADD_S11(IN[0][7],  IN[1][6],   IN[2][5],   C10[1], S10[1]);
    ADDER_LUT_FULL ADD_S12(IN[0][8],  IN[1][7],   IN[2][6],   C10[2], S10[2]);
    ADDER_LUT_FULL ADD_S13(S00[0],    IN[2][7],   IN[3][6],   C10[3], S10[3]);
    ADDER_LUT_FULL ADD_S14(S00[1],    C00[0],     S00[6],     C10[4], S10[4]);
    ADDER_LUT_FULL ADD_S15(S00[2],    C00[1],     S00[7],     C10[5], S10[5]);
    ADDER_LUT_FULL ADD_S16(S00[3],    C00[2],     S00[8],     C10[6], S10[6]);
    ADDER_LUT_FULL ADD_S17(S00[4],    C00[3],     S00[9],     C10[7], S10[7]);
    ADDER_LUT_FULL ADD_S18(S00[5],    C00[4],     IN[6][8],   C10[8], S10[8]);
    ADDER_LUT_FULL ADD_S19(IN[4][11], C00[5],     IN[5][10],  C10[9], S10[9]);
    ADDER_LUT_FULL ADD_S1A(IN[5][11], IN[6][10],  IN[7][9],   C10[10], S10[10]);
    ADDER_LUT_FULL ADD_S1B(IN[6][11], IN[7][10],  IN[8][9],   C10[11], S10[11]);
    ADDER_LUT_HALF ADD_S1C(IN[3][4],  IN[4][3],               C10[12], S10[12]);
    ADDER_LUT_FULL ADD_S1D(IN[3][5],  IN[4][4],   IN[5][3],   C10[13], S10[13]);
    ADDER_LUT_FULL ADD_S1E(IN[4][5],  IN[5][4],   IN[6][3],   C10[14], S10[14]);
    ADDER_LUT_FULL ADD_S1F(IN[5][5],  IN[6][4],   IN[7][3],   C10[15], S10[15]);
    ADDER_LUT_FULL ADD_S1G(C00[6],    S00[10],    IN[8][3],   C10[16], S10[16]);
    ADDER_LUT_FULL ADD_S1H(C00[7],    S00[11],    C00[10],    C10[17], S10[17]);
    ADDER_LUT_FULL ADD_S1I(C00[8],    C00[11],    IN[8][5],   C10[18], S10[18]);
    ADDER_LUT_FULL ADD_S1J(C00[9],    IN[7][7],   IN[8][6],   C10[19], S10[19]);
    ADDER_LUT_FULL ADD_S1K(IN[6][9],  IN[7][8],   IN[8][7],   C10[20], S10[20]);
    ADDER_LUT_FULL ADD_S1L(IN[8][8],  IN[9][7],   IN[10][6],  C10[21], S10[21]);
    ADDER_LUT_HALF ADD_S1M(IN[6][2],  IN[7][1],               C10[22], S10[22]);
    ADDER_LUT_FULL ADD_S1N(IN[7][2],  IN[8][1],   IN[9][0],   C10[23], S10[23]);
    ADDER_LUT_FULL ADD_S1O(IN[8][2],  IN[9][1],   IN[10][0],  C10[24], S10[24]);
    ADDER_LUT_FULL ADD_S1P(IN[9][2],  IN[10][1],  IN[11][0],  C10[25], S10[25]);
    ADDER_LUT_FULL ADD_S1R(IN[9][3],  IN[10][2],  IN[11][1],  C10[26], S10[26]);
    ADDER_LUT_FULL ADD_S1S(IN[9][4],  IN[10][3],  IN[11][2],  C10[27], S10[27]);
    ADDER_LUT_FULL ADD_S1T(IN[9][5],  IN[10][4],  IN[11][3],  C10[28], S10[28]);
    ADDER_LUT_FULL ADD_S1Q(IN[9][6],  IN[10][5],  IN[11][4],  C10[29], S10[29]);
    
    //Stage 3
    ADDER_LUT_HALF ADD_S20(IN[0][4],  IN[1][3],               C20[0], S20[0]);
    ADDER_LUT_FULL ADD_S21(IN[0][5],  IN[1][4],   IN[2][3],   C20[1], S20[1]);
    ADDER_LUT_FULL ADD_S22(S10[0],    IN[2][4],   IN[3][3],   C20[2], S20[2]);
    genvar i;
    for(i=0; i<DATA_WIDTH-'d2; i=i+'d1) begin
        ADDER_LUT_FULL ADD_S2X(S10[i+'d1],    C10[i],     S10[i+'d12],    C20[i+'d3], S20[i+'d3]);
    end
    ADDER_LUT_FULL ADD_S23(S10[11],   C10[10],    IN[9][8],   C20[13], S20[13]);
    ADDER_LUT_FULL ADD_S24(IN[7][11], C10[11],    IN[8][10],  C20[14], S20[14]);
    ADDER_LUT_FULL ADD_S25(IN[8][11], IN[9][10],  IN[10][9],  C20[15], S20[15]);
    ADDER_LUT_HALF ADD_S26(IN[3][2],  IN[4][1],               C20[16], S20[16]);
    ADDER_LUT_FULL ADD_S27(IN[4][2],  IN[5][1],   IN[6][0],   C20[17], S20[17]);
    ADDER_LUT_FULL ADD_S28(IN[5][2],  IN[6][1],   IN[7][0],   C20[18], S20[18]);
    ADDER_LUT_FULL ADD_S29(C10[12],   S10[22],    IN[8][0],   C20[19], S20[19]);
    for(i=0; i<DATA_WIDTH-'d5; i=i+1) begin
        ADDER_LUT_FULL ADD_S2Y(C10[i+'d13], S10[i+'d23], C10[i+'d22], C20[i+'d20], S20[i+'d20]);
    end
    ADDER_LUT_FULL ADD_S2A(C10[20],   IN[11][5],  C10[29],    C20[27], S20[27]);
    ADDER_LUT_FULL ADD_S2B(C10[21],   IN[10][7],  IN[11][6],  C20[28], S20[28]);
    ADDER_LUT_FULL ADD_S2C(IN[9][9],  IN[10][8],  IN[11][7],  C20[29], S20[29]);
    
    //Stage 4
    ADDER_LUT_HALF ADD_S30(IN[0][3], IN[1][2],                C30[0], S30[0]);
    ADDER_LUT_FULL ADD_S31(S20[0],   IN[2][2],    IN[3][1],   C30[1], S30[1]);
    for(i=0; i<DATA_WIDTH+'d2; i=i+1) begin
        ADDER_LUT_FULL ADD_S3X(S20[i+'d1],   C20[i],   S20[i+'d16], C30[i+'d2], S30[i+'d2]);
    end  
    ADDER_LUT_FULL ADD_S32(S20[15],   C20[14],    IN[11][8],  C30[16], S30[16]);
    ADDER_LUT_FULL ADD_S33(IN[9][11], C20[15],    IN[10][10], C30[17], S30[17]);   
    
    //Stage 5
    ADDER_LUT_HALF ADD_S40(IN[0][2],  IN[1][1],               C40[0], S40[0]);
    ADDER_LUT_FULL ADD_S41(S30[0],    IN[2][1],   IN[3][0],   C40[1], S40[1]);
    ADDER_LUT_FULL ADD_S42(S30[1],    C30[0],     IN[4][0],   C40[2], S40[2]);
    ADDER_LUT_FULL ADD_S43(S30[2],    C30[1],     IN[5][0],   C40[3], S40[3]);
    for(i=0; i<DATA_WIDTH+'d2; i=i+1) begin
        ADDER_LUT_FULL ADD_S4X(S30[i+'d3],   C30[i+'d2],   C20[i+'d16], C40[i+'d4], S40[i+'d4]);
    end  
    ADDER_LUT_FULL ADD_S44(S30[17],   C30[16],    IN[11][9],  C40[18], S40[18]);
    ADDER_LUT_FULL ADD_S45(IN[10][11],C30[17],    IN[11][10], C40[19], S40[19]);
    ADDER_LUT_HALF ADD_Q1(IN[1][0],   IN[0][1],               C50[0], Q[1]);
    ADDER_LUT_FULL ADD_Q2(S40[0],     IN[2][0],   C50[0],     C50[1], Q[2]);
    ADDER_LUT_FULL ADD_Q3(~IN[DATA_WIDTH-'d1][DATA_WIDTH-'d1], C40[2*DATA_WIDTH-'d5], C50[2*DATA_WIDTH-'d4], C50[2*DATA_WIDTH-'d3], Q[2*DATA_WIDTH-'d2]);
    
    //############################## Step 3: OUTPUT ##############################
    assign Q[0] = IN[0][0];
    genvar i0;
    for(i0 = 0; i0 < (2*DATA_WIDTH-'d5); i0=i0+'d1) begin
        ADDER_LUT_FULL ADD_QX(S40[i0+'d1], C40[i0], C50[i0+'d1], C50[i0+'d2], Q[i0+'d3]);
    end
    assign Q[2*DATA_WIDTH-'d1] = 1'b1 ^ C50[2*DATA_WIDTH-'d3]; 
       
endmodule
