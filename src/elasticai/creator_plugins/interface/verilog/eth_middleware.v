`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 08.01.2024 09:35:44
// Design Name: 
// Module Name: ETH_DataHandler
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


module ETH_DataHandler#(
    parameter ETH_DATA_BYTE = 16'd16,
    parameter BIT_CNT = 8'd12,
    parameter BIT_CNT_OVR = 8'd4
)(
    // Global input
    input wire CLK_SYS,
    input wire nRST,
    input wire ETH_MOD_RDY,
    input wire ETH_BUSY_TX,
    // Configurating send module
    input wire ETH_SEND_SHOT,
    input wire ETH_SEND_CONT,
    input wire [BIT_CNT-'d1:0] ETH_CYC_WAIT,
    // Control ETH module
    output wire ETH_START_FLAG,
    output wire [8*ETH_DATA_BYTE-'d1:0] ETH_DATA_TX
);

    reg dly_send_shot;
    reg [BIT_CNT+BIT_CNT_OVR-'d1:0] cnt_wait;
    
    wire active_device;
    wire do_send_shot, do_send_cont;
    assign active_device = ETH_MOD_RDY && ~ETH_BUSY_TX;
    assign do_send_shot = active_device && ((ETH_SEND_SHOT && ~dly_send_shot) || (~ETH_SEND_SHOT && dly_send_shot));
    assign do_send_cont = active_device && ETH_SEND_CONT && (cnt_wait == {ETH_CYC_WAIT, {(BIT_CNT_OVR){1'd1}}});

    assign ETH_START_FLAG = do_send_shot || do_send_cont;
    assign ETH_DATA_TX = 'd100;
    
    always@(posedge CLK_SYS) begin
        if(~nRST) begin
            dly_send_shot <= 1'd0;
            cnt_wait <= 'd0;
        end else begin
            dly_send_shot <= ETH_SEND_SHOT;
            cnt_wait <= (active_device && ETH_SEND_CONT && ~do_send_cont) ? cnt_wait + 'd1 : 'd0;
        end     
    end   
    
endmodule
