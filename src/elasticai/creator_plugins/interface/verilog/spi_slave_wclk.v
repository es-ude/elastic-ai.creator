//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     10:27:18 02/28/2019 
// Copied on: 	    §{date_copy_created}
// Module Name:     SPI Slave Module (using System Clock)
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v0
// Processing:      Data Transmission of SPI Protocol with Processing after Transmission
// Dependencies:    Works until CLK_SPI < CLK_SYS / 4 (coming from sampling process and edge detection)
//
// State: 	        Works!
// Improvements:    - If read then catch data from middleware
//                  - Adapt structure for using higher CLK_SPI
// Parameters:      BITWIDTH    --> Bitwidth of SPI packet
//                  CPOL        --> Polarity of the SPI SCLK
//                  CPHA        --> Polarity for sensing and writing MISO/MOSI
//                  MSB         --> Transmission mode if MSB first (1) else LSB first (0)
//////////////////////////////////////////////////////////////////////////////////


module SPI_SLAVE_WCLK#(
    parameter BITWIDTH = 6'd20,
    parameter CPOL = 1'b0,
    parameter CPHA = 1'b0,
    parameter MSB = 1'd1
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire CSN,
    input wire SCLK,
    input wire MOSI,
    output wire MISO,
    output reg DRDY,
    input wire [BITWIDTH - 1'b1:0] DFROM_MIDDLEWARE,
	output reg [BITWIDTH - 1'b1:0] DFOR_MIDDLEWARE
    );


localparam STATE_IDLE = 2'd0, STATE_SAMPLE = 2'd1, STATE_WRITE = 2'd2, STATE_DONE = 2'd3;
reg [1:0] state;
reg [BITWIDTH-'d1:0] buffer;
reg sclk_buf, mosi_buf, sclk_dly, miso_bit;
wire sclk_falling, sclk_rising, sclk_set, sclk_smp;

// Change to default design (lattice does not support tri-state buffer directly)
assign MISO = miso_bit;

// --- Edge detection for different SPI modes
assign sclk_falling = sclk_dly && !sclk_buf;
assign sclk_rising = !sclk_dly && sclk_buf;
assign sclk_set = (!CPOL) ? ((CPHA) ? sclk_rising : sclk_falling) : ((CPHA) ? sclk_falling : sclk_rising);
assign sclk_smp = (!CPOL) ? ((CPHA) ? sclk_falling : sclk_rising) : ((CPHA) ? sclk_rising : sclk_falling);

// --- State Machine for SPI communication
always@(posedge CLK_SYS) begin
	if(!RSTN) begin
		DFOR_MIDDLEWARE <= {(BITWIDTH){1'b0}};
		state <= STATE_IDLE;
		//Initial sampling to prevent false edge detection
		sclk_dly <= CPOL;
		miso_bit <= 1'd0;
		DRDY <= 1'd0;
		buffer <= {(BITWIDTH){1'b0}};
	end else begin
	    mosi_buf <= MOSI;
	    sclk_buf <= SCLK;
		sclk_dly <= sclk_buf;
		
		case(state)
		STATE_IDLE: begin
			if(!CSN) begin
				state <= STATE_SAMPLE;
				buffer <= DFROM_MIDDLEWARE;
				miso_bit <= (MSB) ? DFROM_MIDDLEWARE[BITWIDTH-'d1] : DFROM_MIDDLEWARE[0];
				DRDY <= 1'd0;
			end else begin
				miso_bit <= 1'd0;
				state <= STATE_IDLE;
			end
		end
		STATE_WRITE: begin
			if(CSN == 1'b1) begin
				state <= STATE_DONE;
			end else if(sclk_set) begin
				miso_bit <= (MSB) ? buffer[BITWIDTH-'d1] : buffer[0];
				state <= STATE_SAMPLE;
			end else begin
				state <= STATE_WRITE;
			end
		end
		STATE_SAMPLE: begin
			if(CSN == 1'b1) begin
				state <= STATE_DONE;
			end else if(sclk_smp) begin
				buffer <= (MSB) ? {buffer[BITWIDTH-'d2:0], mosi_buf} : {mosi_buf, buffer[BITWIDTH-'d1:'d1]};
				state <= STATE_WRITE;
			end else begin
				state <= STATE_SAMPLE;
			end
		end
		STATE_DONE: begin
			DRDY <= 1'b1;
			DFOR_MIDDLEWARE <= buffer;
			state <= STATE_IDLE;
		end
		endcase
	end
end
endmodule