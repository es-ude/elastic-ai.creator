//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     13.11.2020 09:22:46
// Copied on: 	    §{date_copy_created}
// Module Name:     LUT-based Multiplier for Signed Input
// Target Devices:  ASIC, FPGA
// Tool Versions:   1v0
// Description:     LUT-based multiplication with parametrized bitwidth
// Processing:      Direct processing
// Dependencies:    None
//
// State: 	        Works! (System Test done: 22.01.2025 on Arty A7-35T with 20% usage)
// Dependency:      ADDER_LUT_FULL, ADDER_LUT_HALF
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input data
//
//////////////////////////////////////////////////////////////////////////////////


module MULT_LUT_SIGNED#(
    parameter BITWIDTH = 6'd6
)(
    input wire signed [BITWIDTH-'d1:0]    A,
    input wire signed [BITWIDTH-'d1:0]    B,
    output wire signed [2*BITWIDTH-'d1:0] Q
);
    wire [BITWIDTH-'d1:0] IN [BITWIDTH-'d1:0];
    wire [BITWIDTH-'d1:0] PP [BITWIDTH-'d2:0];
    wire [BITWIDTH-'d1:0] CC [BITWIDTH-'d2:0];
        
    //################################### Step 1: Partial Products ###################################
    //Info: Val[x0][y0] --> x0 = Row, y0 = Column
    genvar x0 ,y0;
    for(x0=0; x0<BITWIDTH; x0=x0+1) begin
        for(y0=0; y0<BITWIDTH; y0=y0+1) begin
            if((y0 == BITWIDTH-'d1)||(x0 == BITWIDTH-'d1)) begin
                assign IN[x0][y0] = ~(A[y0] & B[x0]);
            end else begin
                assign IN[x0][y0] = A[y0] & B[x0];
            end            
        end
    end  
    //################################### Step 2: Accumulation ###################################
    //Row 2
    ADDER_LUT_HALF PP_row10(IN[0][1], IN[1][0],   CC[0][0],   PP[0][0]);
    for(x0=1; x0 < (BITWIDTH-'d1); x0 = x0+1) begin
        ADDER_LUT_FULL PP_row11(IN[0][x0+'d1], IN[1][x0], CC[0][x0-'d1], CC[0][x0], PP[0][x0]);
    end
    ADDER_LUT_FULL PP_row13(1'b1, IN[1][BITWIDTH-'d1],  CC[0][BITWIDTH-'d2],  CC[0][BITWIDTH-'d1],  PP[0][BITWIDTH-'d1]);
    assign Q[1:0] = {PP[0][0], IN[0][0]};
    
    //Row 3 to BITWIDTH-1
    for(y0 = 2; y0 < (BITWIDTH-'d1); y0 = y0 +'d1) begin
        ADDER_LUT_HALF PP_row20(PP[y0-'d2][1], IN[y0][0],   CC[y0-'d1][0],   PP[y0-'d1][0]);
        for(x0=1; x0 < (BITWIDTH-'d1); x0 = x0+1) begin
            ADDER_LUT_FULL PP_row21(PP[y0-'d2][x0+'d1], IN[y0][x0], CC[y0-'d1][x0-'d1], CC[y0-'d1][x0], PP[y0-'d1][x0]);
        end
        ADDER_LUT_FULL PP_row22(CC[y0-'d2][BITWIDTH-'d1], IN[y0][BITWIDTH-'d1],  CC[y0-'d1][BITWIDTH-'d2],  CC[y0-'d1][BITWIDTH-'d1],  PP[y0-'d1][BITWIDTH-'d1]);
        assign Q[y0] = PP[y0-'d1][0];
    end 
    //Last Row    
    ADDER_LUT_HALF PP_row20(PP[BITWIDTH-'d3][1], IN[BITWIDTH-'d1][0],   CC[BITWIDTH-'d2][0],   PP[BITWIDTH-'d2][0]);
    for(x0=1; x0 < (BITWIDTH-'d1); x0 = x0+1) begin
        ADDER_LUT_FULL PP_row21(PP[BITWIDTH-'d3][x0+'d1], IN[BITWIDTH-'d1][x0], CC[BITWIDTH-'d2][x0-'d1], CC[BITWIDTH-'d2][x0], PP[BITWIDTH-'d2][x0]);
    end
    ADDER_LUT_FULL PP_row22(CC[BITWIDTH-'d3][BITWIDTH-'d1], ~IN[BITWIDTH-'d1][BITWIDTH-'d1],  CC[BITWIDTH-'d2][BITWIDTH-'d2],  CC[BITWIDTH-'d2][BITWIDTH-'d1],  PP[BITWIDTH-'d2][BITWIDTH-'d1]);
        
    //Sign Row
    assign Q[2*BITWIDTH-'d1] = 1'b1 ^ CC[BITWIDTH-'d2][BITWIDTH-'d1];
        
    //################################### Step 3: Output ###################################
    assign Q[2*BITWIDTH-'d2:BITWIDTH-'d1] = PP[BITWIDTH-'d2][BITWIDTH-'d1:0];
    
endmodule