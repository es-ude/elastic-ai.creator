//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     14:15:47 03/05/2019 
// Copied on: 	    §{date_copy_created}
// Module Name:     UART Physical Implementation 
// Target Devices:  FPGA
// Tool Versions:   1v1
// Processing:      Oversampling the RX/TX lines for protocol handling (LSB first)
// Dependencies:    None
//
// State: 	        Works on hardware!
// Improvements:    None
// Comments:        None
// Parameters:      BITRATE     --> Factor for building the BAUDRATE
//                  BITWIDTH    --> BITWIDTH FOR DATA TRANSMISSION
//                  NSAMP       --> Oversampling ratio of UART input
//////////////////////////////////////////////////////////////////////////////////
// Calculating the variable: cntCYC = f_sys/(n* BAUDRATE), mit f_sys = 100 MHz (=Sampling rate)
// Example #1: BAUDRATE = 115.200 (n = 4) --> 217
// Example #2: BAUDRATE = 921.600 (n = 4) --> 27


module UART_COM#(
    parameter BITRATE = 'd26,
    parameter BITWIDTH = 'd8,
    parameter NSAMP = 'd4
)(
    input wire CLK_SYS,
    input wire RSTN,
    // --- Communication signals (from external device)
    input wire RX,
    output reg TX,
    // --- Controlling the Middleware / FIFO buffer (FPGA internal)
    input wire UART_START_FLAG,
    input wire [BITWIDTH-'d1:0] UART_DIN,
    output reg [BITWIDTH-'d1:0] UART_DOUT,
    output wire UART_RDY
);
    localparam STATE_IDLE = 2'd0, STATE_START = 2'd1, STATE_RW = 2'd2, STATE_STOP = 2'd3;
    
    reg [1:0] state;
    reg [BITWIDTH-'d1:0] bufferUART;
    reg [$clog2(BITWIDTH)+'d1:0] bit_transfer;
    reg [$clog2(NSAMP)-'d1:0] valRX, cnt_ovs;
    reg [$clog2(BITRATE)-'d1:0] cnt_dt;
    wire ovs_done_flag, cnt_done_flag, uart_done_flag;

    assign UART_RDY = (state == STATE_IDLE);
    assign cnt_done_flag = (cnt_dt == BITRATE-'d1);
    assign ovs_done_flag = (cnt_ovs == NSAMP-'d1);
    assign uart_done_flag = (bit_transfer == BITWIDTH);
    
    always@(posedge CLK_SYS) begin
        if(!RSTN) begin
            bufferUART <= 'd0;
            UART_DOUT <= 'd0;
            state <= STATE_IDLE;
            cnt_dt <= 'd0;
            cnt_ovs <= 'd0;
            bit_transfer <= 'd0;
            TX <= 1'd1;
            valRX <= 'd0;
        end else begin
            case(state)
                STATE_IDLE: begin
                    state <= (UART_START_FLAG || !RX) ? STATE_START : STATE_IDLE;
                end
                STATE_START: begin
                    TX <= 1'd0;
                    bufferUART <= UART_DIN;
                    if(cnt_done_flag) begin
                        cnt_dt <= 'd0;
                        cnt_ovs <= (ovs_done_flag) ? 'd0 : cnt_ovs + 'd1;
                        state <= (ovs_done_flag) ? STATE_RW : STATE_START;
                        bit_transfer <= bit_transfer + ((ovs_done_flag) ? 'd1 : 'd0);
                    end else begin
                        cnt_dt <= cnt_dt + 'd1;
                        cnt_ovs <= cnt_ovs;
                        state <= STATE_START;
                        bit_transfer <= bit_transfer;
                    end
                end
                STATE_RW: begin
                    TX <= bufferUART[0];
                    if(cnt_done_flag) begin
                        cnt_dt <= 'd0;
                        cnt_ovs <= (ovs_done_flag) ? 'd0 : cnt_ovs + 'd1;
                        state <= (uart_done_flag && ovs_done_flag) ? STATE_STOP : STATE_RW; 
                        bit_transfer <= bit_transfer + ((ovs_done_flag) ? 'd1 : 'd0);
                        valRX <= (ovs_done_flag) ? 'd0 : valRX + RX;
                        bufferUART <= (ovs_done_flag) ? {valRX[1], bufferUART[BITWIDTH-'d1:1]} : bufferUART;
                    end else begin
                        cnt_dt <= cnt_dt + 'd1;
                        cnt_ovs <= cnt_ovs;
                        state <= state; 
                        bit_transfer <= bit_transfer;
                        valRX <= valRX;
                        bufferUART <= bufferUART;
                    end
                end
                STATE_STOP: begin
                    TX <= 1'd1;
                    if(cnt_done_flag) begin
                        cnt_dt <= 'd0;
                        cnt_ovs <= (ovs_done_flag) ? 'd0 : cnt_ovs + 'd1;
                        state <= (ovs_done_flag) ? STATE_IDLE : STATE_STOP;
                        UART_DOUT <= bufferUART;
                        bit_transfer <= 'd0;
                    end else begin
                        cnt_dt <= cnt_dt + 'd1;
                        cnt_ovs <= cnt_ovs;
                        state <= STATE_STOP;
                        UART_DOUT <= UART_DOUT;
                        bit_transfer <= bit_transfer;
                    end
                end  
            endcase
        end
    end
endmodule
