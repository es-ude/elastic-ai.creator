`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 17.10.2021 15:23:02
// Design Name: 
// Module Name: I22C_DataHandler
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


module I22C_DataHandler#(parameter BIT_REG = 5'd3, parameter BIT_WIDTH = 5'd6)(
    //Signals of I22C-Slave
    input                   nRST,
    input                   DATA_RDY,
    inout                   RnW,
    input [BIT_REG-'d1:0]   REG,
    input [BIT_DATA-'d1:0]  DATA_FROM_MASTER,
    output [BIT_DATA-'d1:0] DATA_FOR_MASTER
    //Signals and memory for module
);
    
    
    always@(posedge DATA_RDY or negedge nRST) begin
        if(!nRST) begin
                
        end else begin
            
        end
    end    

endmodule
