//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     13.11.2020 09:22:46
// Copied on: 	    §{date_copy_created}
// Module Name:     Clock- and bit-wise Multiplier for Signed Input
// Target Devices:  ASIC, FPGA
// Tool Versions:   1v0
// Description:     LUT-based multiplication with parametrized bitwidth
// Processing:      Direct processing
// Dependencies:    None
//
// State: 	        Works! (System Test done: 22.01.2025 on Arty A7-35T with 20% usage)
// Dependency:      None
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input data
//
//////////////////////////////////////////////////////////////////////////////////


module MULT_BOOTH_SIGNED#(
    parameter BITWIDTH = 5'd16
)(
    input wire CLK,
    input wire nRST,
    input wire START_FLAG,
    output wire DRDY,
    input wire signed [BITWIDTH-'d1:0] DATA_A,
    input wire signed [BITWIDTH-'d1:0] DATA_B,
    output reg signed [2*BITWIDTH-'d1:0] DOUT
);    
    localparam CNT_WIDTH = $clog2(BITWIDTH - 'd1);
    localparam INIT = 'd0, END = (INIT + BITWIDTH);
    
    reg [CNT_WIDTH:0] noIte;
    reg flag_busy;
    reg doCalc;
    assign DRDY = (noIte == END);
    wire calcCLK;
    assign calcCLK = CLK && (START_FLAG || doCalc);
    reg [BITWIDTH-'d1:0] HIGH;
    reg [BITWIDTH-'d1:0] LOW;   

    //Clock-Gating
    always@(posedge CLK) begin
        if(!nRST) begin
            flag_busy <= 1'd0;
            noIte <= 'd0;
            DOUT <= 'd0;
        end else begin
            flag_busy <= (START_FLAG) ? 1'd1 : ((noIte == END) ? 'd0 : flag_busy);
            noIte <= (flag_busy) ? (noIte + 'd1) : 'd0;
            DOUT <= (noIte == END) ? {HIGH, LOW} : DOUT;
        end
    end
    
    //Calculation
    reg LOW0;
    wire [1:0] check;
    assign check = {LOW[0], LOW0};
    wire [BITWIDTH-'d1:0] SUM;
    assign SUM = HIGH + DATA_A;
    wire [BITWIDTH-'d1:0] DIF;
    assign DIF = HIGH - DATA_A;    
    
    always@(negedge calcCLK or negedge nRST) begin
        if(!nRST) begin
            HIGH <= 0;
            LOW <= 0;
            LOW0 <= 0;
            doCalc <= 1'd0;
        end else begin
        if(noIte == INIT) begin
            HIGH <= 'd0;
            LOW <= DATA_B;
            LOW0 <= 1'b0;
            doCalc <= 1'd1;
        end else begin
            if(check[1]^check[0]) begin
                if(check[1]) begin
                    {HIGH, LOW, LOW0} <= {DIF[BITWIDTH-'d1], DIF, LOW};
                end else begin
                    {HIGH, LOW, LOW0} <= {SUM[BITWIDTH-'d1], SUM, LOW};
                end
            end else begin
                {HIGH, LOW, LOW0} <= {HIGH[BITWIDTH-'d1], HIGH, LOW};
            end
            doCalc <= (noIte == END) ? 1'd0 : doCalc;
        end
    end
end
endmodule
