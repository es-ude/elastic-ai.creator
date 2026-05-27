//////////////////////////////////////////////////////////////////////////////////
// Company:         UDE-ES
// Engineer:        AE
// 
// Create Date:    14:15:47 03/05/2019 
// Design Name: 
// Module Name:    I2C_Master 
// Project Name: 
// Target Devices:  FPGA
// Tool versions: 
// Description: 
//
// Dependencies:    Calculation of VAL_CNT = CLK_SYS / (2 * CLK_I2C) - 'd1
//                  Datenraten-Bereich: 0,1 bis 3,4 Mbit/s
//                  MSB transmission of the data
//                  Size of ADR Frame = 8 bit [ADR, RnW]
//                  Size of DATA = 8 bit (max. frames of 8)
//
//////////////////////////////////////////////////////////////////////////////////
//Falling edge: Set data from Master
//Rising edge: Read in slave

module I2C_Master#( 
    parameter DATA_FRAMES = 4'd2, 
    parameter CLK_I2C = 8'd13,
    parameter CLK_CYC_WAIT = 8'd9
)(
    //Global input data
    input wire CLK,
    input wire nRST,
    input wire START_FLAG,
    input wire DO_READ,
    input wire DO_READ_DATA,
    input wire [6:0] ADR,
    input wire [7:0] REG,
    input wire [DATA_FRAMES*8-'d1:0] DATA_IN,
    output wire [DATA_FRAMES*8-'d1:0] DATA_OUT,
    output reg SLAVE_USED,
    output wire I2C_RDY,
    //Module output to external device
	inout wire I2C_SDA, 
	output wire I2C_SCL
);

localparam STATE_IDLE = 4'd0, STATE_START = 4'd1, STATE_ADR = 4'd2, STATE_ACK0 = 4'd3;
localparam STATE_REG = 4'd4, STATE_DATA = 4'd5, STATE_ACK1 = 4'd6, STATE_STOP = 4'd7, STATE_WAIT = 4'd8;

reg rd_first_transfer_done;
reg [3:0] state;
reg [$clog2(CLK_CYC_WAIT)-'d1:0] cnt_clk_cnt;
reg [$clog2(CLK_I2C)-'d1:0] cnt_clk;
reg [2:0] cnt_i2c, cnt_frames;
reg ack_done, clk_scl, reg_send;
reg [8*DATA_FRAMES-'d1:0] i2c_data_slave;

wire sda_active, check_ack, do_tx, do_rx;
wire [7:0] i2c_din;
wire i2c_rnw;

//Slicing the input
wire [7:0] sliced_din [DATA_FRAMES-'d1:0];
genvar i0;
for (i0=0; i0 < DATA_FRAMES; i0 = i0+1) begin
    assign sliced_din[i0] = DATA_IN[8*i0+:8];
end

assign check_ack = (state == STATE_ACK0) || (state == STATE_ACK1);
assign sda_active = (state == STATE_IDLE) || check_ack;
assign do_tx = (cnt_clk == 'd0) && clk_scl;
assign do_rx = (cnt_clk == 'd0) && !clk_scl && (state == STATE_DATA || check_ack);
assign DATA_OUT = (I2C_RDY && DO_READ && rd_first_transfer_done) ? i2c_data_slave : 'd0;

assign i2c_rnw = (DO_READ && DO_READ_DATA) ? rd_first_transfer_done : DO_READ;
assign i2c_din = (state == STATE_ADR) ? {ADR, i2c_rnw} : ((state == STATE_DATA) ? sliced_din[cnt_frames] : 8'd0);
assign I2C_RDY = (state == STATE_IDLE);
assign I2C_SDA = (state == STATE_START) ? 1'd0 :
                 (state == STATE_ADR) ? i2c_din[cnt_i2c] : 
                 (state == STATE_REG) ? REG[cnt_i2c] : 
                 (state == STATE_DATA && !(DO_READ && rd_first_transfer_done)) ? i2c_din[cnt_i2c] : 
                 (state == STATE_STOP) ? 1'd0 : 1'dZ; 
assign I2C_SCL = (state == STATE_START || state == STATE_WAIT) ? 1'd1 :
                 (state != STATE_IDLE) ? clk_scl : 1'dZ;

//CLK control
always@(posedge CLK) begin
    if(!nRST) begin
        clk_scl <= 1'd1;
        cnt_clk <= clk_scl;
    end else begin  
        clk_scl <= (cnt_clk == 8'd0) ? ~clk_scl : ((I2C_RDY) ? 1'd1 : clk_scl);
        cnt_clk <= (!I2C_RDY && cnt_clk != 'd0) ? cnt_clk - 'd1 : CLK_I2C;
    end
end

//Control for I2C transmission
always@(posedge CLK) begin
    if(!nRST) begin
        rd_first_transfer_done <= 1'd0;
        state <= STATE_IDLE;
        cnt_i2c <= 3'd7;
        cnt_frames <= 3'd0;
        SLAVE_USED <= 1'd0;
        ack_done <= 1'd0;
        i2c_data_slave <= 'd0;
        reg_send <= 1'd0;
        cnt_clk_cnt <= 'd0;
    end else begin        
        case(state)
            STATE_IDLE: begin
                state <= (START_FLAG) ? STATE_START : STATE_IDLE;
                cnt_i2c <= cnt_i2c;
            end
            STATE_START: begin
                if(do_tx) begin
                    state <= STATE_ADR;
                    cnt_i2c <= 3'd7;
                    cnt_frames <= DATA_FRAMES - 3'd1;
                end else begin
                    state <= state;
                    cnt_i2c <= cnt_i2c;
                    cnt_frames <= cnt_frames;
                end
            end
            STATE_ADR: begin
                if(do_tx) begin
                    state <= (cnt_i2c == 3'd0) ? STATE_ACK0 : state;
                    cnt_i2c <= (cnt_i2c == 3'd0) ? 3'd7 : cnt_i2c - 3'd1; 
                    cnt_frames <= cnt_frames;
                end else begin
                    state <= state;
                    cnt_i2c <= cnt_i2c;
                    cnt_frames <= cnt_frames;
                end
            end
            STATE_ACK0: begin
                SLAVE_USED <= (do_rx) ? !I2C_SDA : SLAVE_USED;
                if(do_tx) begin
                    state <= (SLAVE_USED) ? ((DO_READ && rd_first_transfer_done) ? STATE_DATA : STATE_REG) : STATE_STOP;
                    cnt_i2c <= cnt_i2c;
                    cnt_frames <= cnt_frames;
                end else begin
                    state <= state;
                    cnt_i2c <= cnt_i2c;
                    cnt_frames <= cnt_frames;
                end
            end
            STATE_REG: begin
                //--- Processing 
                if(do_tx) begin
                    state <= (cnt_i2c == 3'd0) ? STATE_ACK1 : state;
                    cnt_i2c <= (cnt_i2c == 3'd0) ? 3'd7 : cnt_i2c - 3'd1;
                    cnt_frames <= cnt_frames;
                    reg_send <= reg_send;
                end else begin
                    state <= state;
                    cnt_i2c <= cnt_i2c;
                    cnt_frames <= cnt_frames;
                    reg_send <= reg_send;
                end
            end
            STATE_DATA: begin
				//--- Processing data input
				if(do_rx && DO_READ && rd_first_transfer_done) begin
					i2c_data_slave <= {i2c_data_slave[8*DATA_FRAMES-'d2:0], I2C_SDA};
				end else begin
					i2c_data_slave <= i2c_data_slave;
				end
				//--- Processing 
                if(do_tx) begin
                    state <= (cnt_i2c == 3'd0) ? STATE_ACK1 : state;
                    cnt_i2c <= (cnt_i2c == 3'd0) ? 3'd7 : cnt_i2c - 3'd1;
                    cnt_frames <= cnt_frames;
                end else begin
                    state <= state;
                    cnt_i2c <= cnt_i2c;
                    cnt_frames <= cnt_frames;
                end
            end
            STATE_ACK1: begin
                SLAVE_USED <= (do_rx) ? !I2C_SDA : SLAVE_USED;
                if(do_tx) begin
                    state <= (cnt_frames == 3'd0) ? STATE_STOP : ((REG == 8'hFF || (DO_READ && !rd_first_transfer_done)) ? STATE_STOP : STATE_DATA);
                    cnt_i2c <= cnt_i2c;
                    cnt_frames <= (cnt_frames == 3'd0) ? DATA_FRAMES - 3'd1 : ((reg_send || (DO_READ && rd_first_transfer_done)) ? cnt_frames - 3'd1 : cnt_frames);
                    reg_send <= 1'd1;
                end else begin
                    state <= state;
                    cnt_i2c <= cnt_i2c;
                    cnt_frames <= cnt_frames;
                    reg_send <= reg_send;
                end
            end
            STATE_STOP: begin
                if(do_tx) begin
                    state <= (DO_READ && !rd_first_transfer_done && (REG != 8'hFF)) ? STATE_WAIT : STATE_IDLE;
                    cnt_i2c <= cnt_i2c;
                    cnt_frames <= cnt_frames;
                    reg_send <= 1'd0;
                    rd_first_transfer_done <= DO_READ && (REG != 8'hFF);
                end else begin
                    state <= state;
                    cnt_i2c <= cnt_i2c;
                    cnt_frames <= cnt_frames;
                    reg_send <= reg_send;
                    rd_first_transfer_done <= rd_first_transfer_done;
                end
            end
            STATE_WAIT: begin
                if(cnt_clk_cnt == 'd9 && do_tx) begin
                    cnt_clk_cnt <= 'd0;
                    state <= STATE_START;
                end else begin
                    cnt_clk_cnt <= cnt_clk_cnt + ((do_tx) ? 'd1 : 'd0);
                    state <= state;
                end
            end
        endcase
    end
end

endmodule