////////////////////////////////////////////////////////////////////////////////
//
// Company:         UDE-IES
// Engineer:        AE
//
// Create Date:     10:00 - 27.10.2023
// Design Name:     I2C_Slave
// Target Device:   FPGA
// Tool versions:   
// Description:     ToDo: Processing DATA_IN, state clearly
//
////////////////////////////////////////////////////////////////////////////////

module I2C_Slave#(
    parameter MOD_ADR = 7'h30, 
    parameter DATA_BYTES = 3'd2,
    parameter CNT_WAIT_DONE = 12'd127
)(
	// Global variables
	input wire                         CLK_SYS,
	input wire                         nRST,
	// Communications lines (int.)
	input wire [8* DATA_BYTES-'d1:0]   DATA_IN,
	output wire [8* DATA_BYTES-'d1:0]  DATA_OUT,
	output reg                         I2C_RNW,
	output reg [7:0]                   I2C_REG,
	input wire                         I2C_NXT_RD_DO,
	output reg                         I2C_NXT_RD_CHCK,
	output reg                         I2C_RDY,	
	// Communications lines (ext.)
	inout wire                         SDA, 
	input wire                         SCL
);

//Falling edge: Set data
//Rising edge: Read data 

localparam STATE_IDLE = 3'd0, STATE_ADR = 3'd1, STATE_ACK0 = 3'd2, STATE_REG = 3'd3, STATE_ACK1 = 3'd4, STATE_DATA = 3'd5, STATE_ACK2 = 3'd6, STATE_DONE = 3'd7;
reg [8* DATA_BYTES-'d1:0] i2c_buffer;
reg [3:0] cnt_frame;
reg [2:0] cnt_state;
reg [$clog2(DATA_BYTES)-'d1:0] cnt_data;
reg [$clog2(CNT_WAIT_DONE)-'d1:0] cnt_wait;
reg mod_selected, dly_sda;
reg [1:0] dly_scl;

//Control logic for SDA bidirectional buffer
wire i2c_din, i2c_dout, i2c_active_ack, i2c_active_out;
assign i2c_active_ack = mod_selected && (cnt_state == STATE_ACK0 || cnt_state == STATE_ACK1 || cnt_state == STATE_ACK2);
assign i2c_active_out = mod_selected && (cnt_state == STATE_DATA) && I2C_RNW;
assign i2c_dout = (i2c_active_out) ? i2c_buffer[8* DATA_BYTES-'d1] : 1'd0;
bidirec_buffer SDA_BUF(
    .oe(i2c_active_out || i2c_active_ack),
    .data_wr(i2c_dout),
    .data_rd(i2c_din), 
    .bidir(SDA)
);

assign DATA_OUT = (~(mod_selected || I2C_RNW)) ? i2c_buffer : 'd0;
//Control logic for processing data
wire scl_rising_edge, scl_falling_edge, chng_state_ack;
assign scl_rising_edge = dly_scl[0] && !dly_scl[1];
assign scl_falling_edge = !dly_scl[0] && dly_scl[1];
assign chng_state_ack = scl_falling_edge && (cnt_frame == 4'd8);

always@(posedge CLK_SYS) begin
	if(~nRST) begin
	   dly_scl <= 2'd0;
	   dly_sda <= 1'd0;
	   cnt_wait <= 'd0;
	   cnt_frame <= 4'd0;
	   cnt_data <= 'd0;
	   cnt_state <= STATE_IDLE;
	   mod_selected <= 1'd0;
	   i2c_buffer <= 'd0;
	   I2C_RNW <= 1'd0;
	   I2C_RDY <= 1'd1;	   
	   I2C_REG <= 8'd0;
	   I2C_NXT_RD_CHCK <= 1'd0;
	end else begin
	   dly_scl <= {dly_scl[0], SCL};
	   dly_sda <= i2c_din;
	   // Control scheme for I2C module
	   case(cnt_state)
	       STATE_IDLE: begin
	           cnt_state <= (cnt_wait == 'd19) ? STATE_ADR : STATE_IDLE;
	           cnt_wait <= (!dly_sda && SCL) ? cnt_wait + 'd1 : 'd0;
	       end
	       // Getting the ADR from Master module
	       STATE_ADR: begin
	           cnt_wait <= 3'd0;
	           i2c_buffer <= (scl_rising_edge) ? {i2c_buffer[6:0], dly_sda} : i2c_buffer;  
	           cnt_state <= (chng_state_ack) ? STATE_ACK0 : cnt_state;  
	           cnt_frame <= (scl_rising_edge) ? cnt_frame + 4'd1 : cnt_frame;
	           mod_selected <= (chng_state_ack) ? (i2c_buffer[7:1] == MOD_ADR[6:0]) : mod_selected;
	           I2C_RNW <= (chng_state_ack) ? i2c_buffer[0] : I2C_RNW;
	       end	
	       // Acknowledgement #1
	       STATE_ACK0: begin
	           i2c_buffer <= (I2C_NXT_RD_DO && I2C_RNW && mod_selected) ? DATA_IN : i2c_buffer;
	           cnt_state <= (scl_falling_edge) ? ((mod_selected) ? ((I2C_NXT_RD_DO && I2C_RNW) ? STATE_DATA : STATE_REG) : STATE_DONE) : cnt_state;
	           cnt_frame <= 4'd0;
	       end  
	       // Getting the Register address from master
	       STATE_REG: begin
	           i2c_buffer <= (scl_rising_edge) ? {i2c_buffer[6:0], dly_sda} : i2c_buffer;  
	           cnt_state <= (chng_state_ack) ? STATE_ACK1 : cnt_state;  
	           cnt_frame <= (scl_rising_edge) ? cnt_frame + 4'd1 : cnt_frame;
	           I2C_REG <= (chng_state_ack) ? i2c_buffer[7:0] : I2C_REG;
	           I2C_NXT_RD_CHCK <= (chng_state_ack);
	       end
	       // Acknowledgement #2
	       STATE_ACK1: begin
	           cnt_state <= (scl_falling_edge) ? ((I2C_REG == 8'hFF || I2C_NXT_RD_DO) ? STATE_DONE : STATE_DATA) : cnt_state;
	           cnt_frame <= 4'd0;
	           I2C_NXT_RD_CHCK <= I2C_NXT_RD_DO;
	       end 
	       STATE_DATA: begin	           
	           i2c_buffer <= ((scl_rising_edge && ~I2C_RNW) || (scl_falling_edge && I2C_RNW)) ? {i2c_buffer[8* DATA_BYTES-'d2:0], dly_sda} : i2c_buffer;
	           cnt_state <= (chng_state_ack) ? STATE_ACK2 : cnt_state;  
	           cnt_frame <= (scl_rising_edge) ? cnt_frame + 4'd1 : cnt_frame;
	           I2C_RDY <= 1'd0;
	       end
	       // Acknowledgement #3
	       STATE_ACK2: begin   
	           cnt_frame <= 4'd0;
	           if(scl_falling_edge) begin
	               cnt_state <= (cnt_data == DATA_BYTES-'d1) ? STATE_DONE : STATE_DATA;	   
	               cnt_data <= (cnt_data == DATA_BYTES-'d1) ? 'd0 : cnt_data + 'd1;
	           end else begin
	               cnt_state <= cnt_state;
	               cnt_data <= cnt_data;
	           end
	       end 
	       // Done
	       STATE_DONE: begin
	           cnt_frame <= 4'd0;
	           mod_selected <= 1'd0;
	           I2C_NXT_RD_CHCK <= I2C_NXT_RD_CHCK;
	           I2C_RNW <= I2C_RNW;
	           if(cnt_wait == CNT_WAIT_DONE) begin
	               cnt_state <= STATE_IDLE;
	               cnt_wait <= 'd0;
	               cnt_data <= 'd0;
	               I2C_RDY <= 1'd1;
	           end else begin
	               cnt_state <= cnt_state;
	               cnt_wait <= (i2c_din && SCL) ? cnt_wait + 'd1 : 'd0;
	               cnt_data = cnt_data;
	               I2C_RDY <= I2C_RDY;
	           end
	       end 
	   endcase	   
	end
end
endmodule


//Addon for bidirec buffer
module bidirec_buffer#(parameter BIT_SIZE=4'd1)(
    //Output signal lines
    input wire oe,
    input wire [BIT_SIZE-'d1:0] data_wr,
    //Input signal line
    output wire [BIT_SIZE-'d1:0] data_rd,
    //Bidirectional signal line
    inout wire [BIT_SIZE-'d1:0] bidir
);
    
    assign data_rd = bidir;
    assign bidir = (oe) ? data_wr : 'dZ;
endmodule
    