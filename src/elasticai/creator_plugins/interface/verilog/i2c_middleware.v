`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10.05.2020 13:41:09
// Design Name: 
// Module Name: I2C_DataHandler
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
//			i2c_reg == 8'hFF bleiben frei für TESTZWECKE!
// 
//////////////////////////////////////////////////////////////////////////////////

module COM_DataHandler#(
    parameter FIFO_BYTE_DATA = 4'd2,
    parameter ID_CODE_DEV = 16'h0AF1
)(
    input wire                      CLK_SYS,
    input wire                      nRST,
    // DATA PROTOCOLE
    input wire                      I2C_NXT_RD_CHCK,
    output wire                     I2C_NXT_RD_DO,
    input wire                      I2C_RNW,
    input wire [7:0]                I2C_REG,
    input wire [8*FIFO_BYTE_DATA-'d1:0] DATA_RX,
    output wire [8*FIFO_BYTE_DATA-'d1:0] DATA_TX,
    input wire                      DATA_RDY, 
    // DATA
	output reg 						LED_TEST
);

//--- Controlling the I2C Handler (Big-Endian Encoding)
wire [15:0] i2c_data_get;
wire [15:0] i2c_data_return;

assign i2c_data_get = {DATA_RX[7:0], DATA_RX[15:8]};
assign DATA_TX = {i2c_data_return[7:0], i2c_data_return[15:8]};

//--- Controlling the data return path
assign I2C_NXT_RD_DO = (I2C_NXT_RD_CHCK && I2C_REG[7] && I2C_REG != 8'hFF);
assign i2c_data_return = (I2C_NXT_RD_DO) ? ((I2C_REG == 8'h80) ? ID_CODE_DEV : 16'h1001) : 16'h0000;

//--- Positive edge detection of DATA_RDY
reg shift_drdy;
wire drdy_posflag;
assign drdy_posflag = DATA_RDY && !shift_drdy && !I2C_RNW;

//--- Handler for writing
always@(posedge CLK_SYS) begin
    if(!nRST) begin
        shift_drdy  <= 1'd0;
        LED_TEST <= 1'd0;
    end else begin
        shift_drdy  <= DATA_RDY;
        // ---------------------------- Test Strukturen ----------------------------
        //LED Control
        LED_TEST        <= (drdy_posflag && (I2C_REG == 8'h00)) ? ((i2c_data_get[15]) ? ~LED_TEST : i2c_data_get[14]) : LED_TEST;       
    end  
 end
endmodule
