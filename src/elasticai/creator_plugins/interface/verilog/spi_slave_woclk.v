//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     16:03:34 03/01/2019
// Copied on: 	    §{date_copy_created}
// Module Name:     SPI Slave Module (not using System Clock)
// Target Devices:  ASIC
// Tool Versions:   1v0
// Processing:      Data Transmission of SPI Protocol with Processing after Transmission
// Dependencies:    None
//
// State: 	        Tested!
// Improvements:    - If read then catch data from middleware
//                  - Actual only MODE 1 is supported. Adding configuration
// Parameters:      BITWIDTH    --> Bitwidth/size of spi packet
//                  CPOL        --> Polarity of the SPI SCLK
//                  CPHA        --> Polarity for sensing and writing MISO/MOSI
//                  MSB         --> Transmission mode if MSB first (1) else LSB first (0)
//////////////////////////////////////////////////////////////////////////////////


module SPI_SLAVE_WOCLK#(
    parameter BITWIDTH = 6'd8,
    parameter CPOL = 1'b0,
    parameter CPHA = 1'b0,
    parameter MSB = 1'b0
)(
    input wire RSTN,
    input wire [BITWIDTH-'d1:0] DFROM_MIDDLEWARE,
	output wire [BITWIDTH-'d1:0] DFOR_MIDDLEWARE,
	output wire DRDY,
    //Signal for communication
	input wire CSN,
    input wire SCLK,
	input wire MOSI,
	output wire MISO	
);

reg writeData, miso_int;
reg [BITWIDTH-'d1:0] buffer_spi;
wire write_flag;
wire delete_flag;

assign DRDY = !RSTN ? 1'b0 : CSN;
assign MISO = (CSN) ? 1'bZ : miso_int;
assign DFOR_MIDDLEWARE = (DRDY) ? buffer_spi : 'd0;
assign write_flag = ~(delete_flag || writeData);
assign delete_flag = !RSTN || CSN;

//Set Write Flag (Data For Master)
always@(posedge SCLK or posedge delete_flag) begin
	if(delete_flag) begin
		writeData <= 1'b0;
	end else begin	
		writeData <= 1'b1;
	end
end

//Update MISO
always@(posedge SCLK or posedge CSN or negedge RSTN) begin
	if(!RSTN) begin
		miso_int <= 1'b0;
	end else begin
		if(CSN) begin
			miso_int <= 1'b0;
		end else begin
			miso_int <= (MSB) ? buffer_spi[BITWIDTH-'d1] : buffer_spi[0];
		end
	end
end

//Sample MOSI
always@(negedge SCLK or posedge write_flag) begin
	if(write_flag) begin
		buffer_spi <= DFROM_MIDDLEWARE;
	end else begin
		if(!SCLK) begin
			buffer_spi <= (MSB) ? {buffer_spi[BITWIDTH-'d2:0], MOSI} : {MOSI, buffer_spi[BITWIDTH-'d1:1]};
		end else begin
			buffer_spi <= buffer_spi;
		end
	end
end
endmodule
