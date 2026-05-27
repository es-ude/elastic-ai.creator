`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 15.10.2021 12:24:34
// Design Name: 
// Module Name: I22C_Slave
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

//MSB first transmission
//For reading data, DATA_RDY is earlier high (no second request necessary)

module I22C_Slave#(parameter BIT_ADR = 5'd3, parameter BIT_REG = 5'd3, parameter BIT_DATA = 5'd6)(
    //Internal control lines
    input wire nRST,
    input wire [BIT_ADR-'d1:0] modADR,
    //Ringbuffer 
    input wire SCL,
    input wire SDI,
    output wire SCL_buf,
    output wire SDI_buf,
    //Data to handle
    output wire DATA_RDY,
    output reg RnW,
    output reg [BIT_REG-'d1:0] REG,
    input  wire [BIT_DATA-'d1:0] DIN,
    output wire [BIT_DATA-'d1:0] DOUT
);
    localparam BIT_SHIFT = 'd2 + BIT_ADR + BIT_REG + BIT_DATA ;
    wire idle_slave;    
    reg right_module;
    reg [BIT_SHIFT-'d1:0] shift_register;
    reg [4:0] cnt_cyc;
    reg [4:0] cnt_data;
    reg test;
 
    assign idle_slave = (cnt_cyc == 5'd0);  
    assign DATA_RDY = (idle_slave || RnW);
    assign DOUT = (DATA_RDY && ~RnW) ? shift_register[cnt_data] : 'd0;  

    assign SCL_buf = SCL;
    assign SDI_buf = (test) ? DIN[cnt_data] : SDI;
    
    //Datenverarbeitung und Clock-Zählung
    always@(negedge SCL or negedge nRST) begin
        if(!nRST) begin
            shift_register <= 'd0;
            cnt_cyc <= 1'b0;     
            RnW <= 1'd0;
            REG <= 'd0;
        end else begin
            shift_register <= {shift_register[BIT_SHIFT-'d1:0], SDI};
            cnt_cyc <= cnt_cyc + 5'd1;
            case(cnt_cyc)
                0: begin
                    RnW <= 1'd0;
                end
                BIT_SHIFT-BIT_DATA-'d1: begin
                    REG <= (right_module) ? shift_register[BIT_REG:1] : 'd0;
                    RnW <= right_module && shift_register[0];
                end
                BIT_SHIFT-'d1: begin
                    cnt_cyc <= 5'd0;
                end
            endcase
        end
    end
    wire reset;     assign reset = !nRST || idle_slave;
    always@(negedge SCL or posedge reset) begin
        right_module <= (reset) ? 1'd0 : ((cnt_cyc == BIT_ADR) ? (modADR == shift_register[BIT_ADR-1:0]) : right_module);
    end
    
    wire CLK;       assign CLK = SCL && right_module && RnW;
    always@(posedge CLK or posedge reset) begin
        test <= ~reset;
        cnt_data <= (reset) ? BIT_DATA : cnt_data - 'd1;
    end
endmodule
