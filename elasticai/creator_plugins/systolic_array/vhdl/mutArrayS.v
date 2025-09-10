`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 13.11.2020 09:22:46
// Design Name: 
// Module Name: Array Multiplier (signed)
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module mutArrayS#(parameter DATA_WIDTH = 6)(
    input wire signed [DATA_WIDTH-'d1:0]    A,
    input wire signed [DATA_WIDTH-'d1:0]    B,
    output wire signed [2*DATA_WIDTH-'d1:0] Q
);
    wire [DATA_WIDTH-'d1:0] IN [DATA_WIDTH-'d1:0];
    wire [DATA_WIDTH-'d1:0] PP [DATA_WIDTH-'d2:0];
    wire [DATA_WIDTH-'d1:0] CC [DATA_WIDTH-'d2:0];
        
    //################################### Step 1: Partial Products ###################################
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
    //################################### Step 2: Summenbildung ###################################
    //Row 2
    mutHalfAdder PP_row10(IN[0][1], IN[1][0],   CC[0][0],   PP[0][0]);
    for(x0=1; x0 < (DATA_WIDTH-'d1); x0 = x0+1) begin
        mutFullAdder PP_row11(IN[0][x0+'d1], IN[1][x0], CC[0][x0-'d1], CC[0][x0], PP[0][x0]);
    end
    mutFullAdder PP_row13(1'b1, IN[1][DATA_WIDTH-'d1],  CC[0][DATA_WIDTH-'d2],  CC[0][DATA_WIDTH-'d1],  PP[0][DATA_WIDTH-'d1]);
    assign Q[1:0] = {PP[0][0], IN[0][0]};
    
    //Row 3 to DATA_WIDTH-1
    for(y0 = 2; y0 < (DATA_WIDTH-'d1); y0 = y0 +'d1) begin
        mutHalfAdder PP_row20(PP[y0-'d2][1], IN[y0][0],   CC[y0-'d1][0],   PP[y0-'d1][0]);
        for(x0=1; x0 < (DATA_WIDTH-'d1); x0 = x0+1) begin
            mutFullAdder PP_row21(PP[y0-'d2][x0+'d1], IN[y0][x0], CC[y0-'d1][x0-'d1], CC[y0-'d1][x0], PP[y0-'d1][x0]);
        end
        mutFullAdder PP_row22(CC[y0-'d2][DATA_WIDTH-'d1], IN[y0][DATA_WIDTH-'d1],  CC[y0-'d1][DATA_WIDTH-'d2],  CC[y0-'d1][DATA_WIDTH-'d1],  PP[y0-'d1][DATA_WIDTH-'d1]);
        assign Q[y0] = PP[y0-'d1][0];
    end 
    //Last Row    
    mutHalfAdder PP_row20(PP[DATA_WIDTH-'d3][1], IN[DATA_WIDTH-'d1][0],   CC[DATA_WIDTH-'d2][0],   PP[DATA_WIDTH-'d2][0]);
    for(x0=1; x0 < (DATA_WIDTH-'d1); x0 = x0+1) begin
        mutFullAdder PP_row21(PP[DATA_WIDTH-'d3][x0+'d1], IN[DATA_WIDTH-'d1][x0], CC[DATA_WIDTH-'d2][x0-'d1], CC[DATA_WIDTH-'d2][x0], PP[DATA_WIDTH-'d2][x0]);
    end
    mutFullAdder PP_row22(CC[DATA_WIDTH-'d3][DATA_WIDTH-'d1], ~IN[DATA_WIDTH-'d1][DATA_WIDTH-'d1],  CC[DATA_WIDTH-'d2][DATA_WIDTH-'d2],  CC[DATA_WIDTH-'d2][DATA_WIDTH-'d1],  PP[DATA_WIDTH-'d2][DATA_WIDTH-'d1]);
        
    //Sign Row
    assign Q[2*DATA_WIDTH-'d1] = 1'b1 ^ CC[DATA_WIDTH-'d2][DATA_WIDTH-'d1];
        
    //################################### Step 3: Ausgang ###################################
    assign Q[2*DATA_WIDTH-'d2:DATA_WIDTH-'d1] = PP[DATA_WIDTH-'d2][DATA_WIDTH-'d1:0];
    
endmodule

//Untermodule
module mutFullAdder(
    input A,
    input B,
    input Cin,
    output Cout,
    output Q
);
    wire C0, C1, Q0;
    assign Cout = C0 | C1;
    
    mutHalfAdder ADD0(A, B, C0, Q0);
    mutHalfAdder ADD1(Cin, Q0, C1, Q); 
    
endmodule

module mutHalfAdder(
    input A,
    input B,
    output Cout,
    output Q
);
    assign Cout = A & B;  
    assign Q = A ^ B; 

endmodule

