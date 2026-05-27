//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     14:15:47 03/05/2019 
// Copied on: 	    §{date_copy_created}
// Module Name:     SPI Master Module (using System Clock) 
// Target Devices:  FPGA
// Tool Versions:   1v0
// Processing:      Data Transmission of SPI Protocol with Processing after Transmission
// Dependencies:    None
//
// State: 	        Tested!
// Improvements:    None
// Comments:        Sampling and Updating takes place at the end of each state. Maybe thats weird!
// Parameters:      BITWIDTH    --> Bitwidth of SPI packet
//                  CPOL        --> Polarity of the SPI SCLK
//                  CPHA        --> Polarity for sensing and writing MISO/MOSI
//                  MSB         --> Transmission mode if MSB first (1) else LSB first (0)
//                  SPI_DIV_CLK --> Divison Number for Generating the SPI SCLK
//////////////////////////////////////////////////////////////////////////////////


module SPI_MASTER#(
    parameter BITWIDTH = 8'd24,
    parameter CPOL = 1'd0,
    parameter CPHA = 1'd0,
    parameter MSB = 1'd1,
    parameter SPI_DIV_CLK = 8'd4 
)(
    input wire CLK_SYS,
	input wire RSTN,
	input wire START_FLAG,
    input wire [BITWIDTH-'d1:0] DATA_TX,
    output wire [BITWIDTH-'d1:0] DATA_RX,
	output reg DRDY,
	input wire MISO,
	output reg MOSI,
	output reg SCLK,
	output reg CSN
);

localparam STATE_IDLE = 3'd0, STATE_SELECT = 3'd1, STATE_WRITE = 3'd2, STATE_SAMPLE = 3'd3, STATE_CNT = 3'd4;

reg [BITWIDTH-'d1:0] buffer_spi;
reg [$clog2(BITWIDTH)-'d1:0] cnt_bit; 
reg [$clog2(SPI_DIV_CLK):0] cnt_sclk;
reg [2:0] state;

assign DATA_RX = (DRDY) ? buffer_spi : {(BITWIDTH){1'b0}};

//State-Machine for SPI communication
always@(posedge CLK_SYS) begin
	if(!RSTN) begin
		SCLK <= CPOL;
		MOSI <= 1'd0;
		CSN <= 1'd1;
		DRDY <= 1'd0;
		buffer_spi <= {(BITWIDTH){1'b0}};
		cnt_bit <= 'd0;
		cnt_sclk <= 'd0;
		state <= STATE_IDLE;
	end else begin
		case(state)// synopsys full_case
		STATE_IDLE: begin
			buffer_spi <= buffer_spi;
			cnt_bit <= 'd0;
			state <= (START_FLAG) ? STATE_SELECT : STATE_IDLE;
		end
		STATE_SELECT: begin
		    MOSI <= (MSB) ? DATA_TX[BITWIDTH-'d1] : DATA_TX['d0];
			CSN <= 1'd0;
			DRDY <= 1'd0;
			SCLK <= CPOL;
			buffer_spi <= DATA_TX;
			cnt_bit <= 'd0;
			cnt_sclk <= 'd0;
			state <= (CPHA == 'd0) ? STATE_SAMPLE : STATE_WRITE;
		end
		STATE_WRITE: begin
		    CSN <= CSN;
		    cnt_bit <= cnt_bit;
		    if(cnt_sclk == SPI_DIV_CLK) begin
                SCLK <= ~SCLK;
                MOSI <= (MSB) ? buffer_spi[BITWIDTH-'d1] : buffer_spi['d0];
                cnt_sclk <= 'd0;
                state <= (CPHA == 'd0) ? STATE_CNT : STATE_SAMPLE;
            end else begin
                SCLK <= SCLK;
                MOSI <= MOSI;
                cnt_sclk <= cnt_sclk + 'd1;
                state <= state;
            end   
		end
		STATE_SAMPLE: begin
		    CSN <= CSN;
		    MOSI <= MOSI;
		    cnt_bit <= cnt_bit;
			if(cnt_sclk == SPI_DIV_CLK) begin
			     SCLK <= ~SCLK;
			     buffer_spi <= (MSB) ? {buffer_spi[BITWIDTH-'d2:'d0], MISO} : {MISO, buffer_spi[BITWIDTH-'d1:'d1]};
			     cnt_sclk <= 'd0;
			     state <= (CPHA == 'd0) ? STATE_WRITE: STATE_CNT;
            end else begin
                SCLK <= SCLK;
                buffer_spi <= buffer_spi;
                cnt_sclk <= cnt_sclk + 'd1;
                state <= state;
            end     
		end
		STATE_CNT: begin
		  cnt_sclk <= 'd0;
		  MOSI <= 1'd0;
		  if(cnt_bit == BITWIDTH-'d1) begin
		      CSN <= 1'd1;
		      MOSI <= 1'd0;
		      SCLK <= CPOL;
		      DRDY <= 1'd1;
		      cnt_bit <= 'd0;
		      state <= STATE_IDLE;
		  end else begin
		      CSN <= 1'd0;
		      MOSI <= MOSI;
		      SCLK <= SCLK;
		      DRDY <= 1'd0;
		      cnt_bit <= cnt_bit + 'd1;
		      state <= (CPHA == 'd0) ? STATE_SAMPLE : STATE_WRITE;
		  end	
		end
		endcase
	end
end
endmodule
