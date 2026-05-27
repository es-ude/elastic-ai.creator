//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     19.04.2023 18:25:50
// Copied on: 	    §{date_copy_created}
// Module Name:     UART FIFO Module for building complete communication protocols
// Target Devices:  FPGA
// Tool Versions:   1v1
// Dependencies:    None
//
// State: 	        Tested!
// Improvements:    None
// Comments:        Little-endian transmission, but big-endian processing
// Parameters:      FIFO_SIZE  --> Number of bytes for saving FIFO
//                  BITWIDTH    --> BITWIDTH FOR DATA TRANSMISSION
//////////////////////////////////////////////////////////////////////////////////


module UART_FIFO#(
    parameter FIFO_SIZE = 'd4,
    parameter BITWIDTH = 'd8
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire UART_RDY_FLAG,
    input wire START_FLAG,
    //Sliced data for UART module
    input wire [BITWIDTH-'d1:0] UART_DIN,
    output wire [BITWIDTH-'d1:0] UART_DOUT,
    output wire UART_START_FLAG,
    //Transmitted data for FPGA
    input wire [BITWIDTH*FIFO_SIZE-'d1:0] FIFO_IN,
    output reg [BITWIDTH*FIFO_SIZE-'d1:0] FIFO_OUT,
    output wire FIFO_RDY
);

localparam STATE_IDLE = 2'd0, STATE_START = 2'd1, STATE_FIFO = 2'd2, STATE_STOP = 2'd3;

reg [1:0] state;
reg int_trigger;
reg [$clog2(FIFO_SIZE+'d1)-'d1:0] cnt_uart;
reg [2:0] shift_uart;
reg [1:0] shift_start;
wire do_flag_pos, do_flag_neg;

assign FIFO_RDY = (state == STATE_IDLE);
assign UART_START_FLAG = int_trigger && ((state == STATE_START) || (do_flag_pos && (cnt_uart < FIFO_SIZE-'d1)));
assign UART_DOUT = FIFO_OUT[(FIFO_SIZE-'d1)*BITWIDTH+:'d8];

assign do_flag_pos = (&shift_uart[1:0] && !shift_uart[2]);
assign do_flag_neg = (~&shift_uart[1:0] && shift_uart[1]);
assign do_start = (shift_start[0] && !shift_start[1]);

// FIFO-Controller
always@(posedge CLK_SYS) begin
    if(!RSTN) begin
        shift_uart <= 3'b111;
        shift_start <= 2'd0;
        state <= STATE_IDLE;
        cnt_uart <= 'd0;
        FIFO_OUT <= 'd0;
        int_trigger <= 1'd0;
    end else begin
        shift_uart <= {shift_uart[1:0], UART_RDY_FLAG};
        shift_start <= {shift_start[0], START_FLAG};
        case(state)
            STATE_IDLE: begin
                state <= (do_flag_neg ^ do_start) ? STATE_START : STATE_IDLE;
                int_trigger <= do_start;
            end
            STATE_START: begin
                state <= STATE_FIFO;
                FIFO_OUT <= FIFO_IN;
            end
            STATE_FIFO: begin
                if(cnt_uart == FIFO_SIZE) begin
                    state <= STATE_STOP;
                    cnt_uart <= 'd0;
                    FIFO_OUT <= FIFO_OUT;
                end else begin
                    state <= STATE_FIFO;
                    cnt_uart <= cnt_uart + ((do_flag_pos) ? 'd1 : 'd0);
                    FIFO_OUT <= (do_flag_pos) ? {FIFO_OUT[0+:(FIFO_SIZE-'d1)*BITWIDTH], UART_DIN} : FIFO_OUT;
                end
            end
            STATE_STOP: begin
                state <= STATE_IDLE;
            end
        endcase
    end
end
endmodule
