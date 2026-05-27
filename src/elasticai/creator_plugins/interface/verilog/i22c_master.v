`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 15.10.2021 12:13:53
// Design Name: 
// Module Name: I22C_Master
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

module I22C_Master#(parameter BIT_ADR = 'd3, parameter BIT_REG = 'd3, parameter BIT_DATA = 'd6)(
    //Internal control signals
    input wire   CLK_sys,
    input wire 	 nRST,
    input wire	start_flag,
    output reg DATA_RDY,
    output wire [BIT_DATA-'d1:0] DATA_I22C,
    //Data input signals
    input wire [BIT_ADR-'d1:0]   ADR,
    input wire              RnW,
    input wire [BIT_REG-'d1:0]   REG,
    input wire [BIT_DATA-'d1:0]  DATA,
    //Communication signals (Ringbuffer-Aufbau)
    input   SDI,
    output  reg SCL,
    output  reg SDO
);
    localparam BIT_SHIFT = 'd2 + BIT_ADR + BIT_DATA + BIT_REG;
    localparam Idle = 3'd0, Prep = 3'd1, Write = 3'd2, Sample = 3'd3, Ends = 3'd4;
    
    reg [2:0] state;   
    reg SDI_buf;
    reg [5:0] cnt_cyc;
    reg [BIT_SHIFT-'d1:0] shift_register;

    assign DATA_I22C = (DATA_RDY) ? shift_register : 'd0;
    
    always@(posedge CLK_sys or negedge nRST) begin
        if(!nRST) begin
            DATA_RDY <= 1'd0;
            state <= Idle;
            cnt_cyc <= 6'd0;
            shift_register <= 'd0;
            SCL <= 1'd0;
            SDO <= 1'd0;
            SDI_buf <= 1'd0;
        end else begin
            SDI_buf <= SDI;
            case(state)
                Idle: begin
                    state <= (start_flag) ? 2'd1 : 2'd0;
                end
                Prep: begin
                    state <= Write;
                    DATA_RDY <= 1'd0;
                    cnt_cyc <= 6'd0;
                    shift_register <= {ADR, REG, RnW, 1'd0, DATA};
                    SDO <= 1'd0;
                end
                Write: begin
                    state <= Sample;
                    SCL <= 1'd1;
                    SDO <= shift_register[BIT_SHIFT-'d1];
                    shift_register <= shift_register;
                    cnt_cyc <= cnt_cyc;
                end
                Sample: begin
                    state <= (cnt_cyc == BIT_SHIFT -'d1) ? Ends : Write;
                    SCL <= 1'd0;
                    SDO <= SDO;
                    shift_register <= {shift_register[BIT_SHIFT-'d2:0], SDI_buf};
                    cnt_cyc <= cnt_cyc + 6'd1;
                end
                Ends: begin
                    state <= Idle;
                    DATA_RDY <= 1'd1;
                    SCL <= 1'd0;
                    SDO <= 1'd0;
                end
            endcase
        end
    end   
endmodule
