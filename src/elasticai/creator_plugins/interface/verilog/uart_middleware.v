//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     19.04.2023 21:15:14
// Copied on: 	    §{date_copy_created}
// Module Name:     UART FIFO Module for building complete communication procotocols
// Target Devices:  FPGA
// Tool Versions:   1v0
// Dependencies:    None
//
// State: 	        Tested!
// Improvements:    None
// Comments:        None
// Parameters:      BITWIDTH_DATA   --> Number of bits for data handling of the procotol
//                  BITWIDTH_ADR    --> Number of bits for address width
//                  BITWIDTH_CMDS   --> Number of bits for commando
//                  BITWIDTH        --> BITWIDTH FOR DATA TRANSMISSION
//                  FIFO_SIZE       --> Number of bytes getting from UART
//////////////////////////////////////////////////////////////////////////////////
// Example of the implemented data structure
// ------------------------------- DATA FRAME ---------------------------------------------------------------
// # ---- CMD (2 bits)----  # ---- ADR (BITWIDTH_ADR) ----          # ---- DATA (BITWIDTH_DATA) ----        #
// # 0: REG_DUT_CNTL        # TOGGLE_LED | CHANGE_LED | DO_TEST     # xxxx | LED                            #
// # 1: REG_DUT_WR          # ---- ADR ----                         # ---- DATA ----                        #
// # 2: REG_DUT_RD          # ---- ADR ----                         # xxxxxx                                #
// # 3: Reserved            # ---- ADR ----                         # xxxxxx                                #
// ------------------------------- DATA FRAME ---------------------------------------------------------------

module UART_MIDDLEWARE#(
    parameter FIFO_SIZE = 5'd3,
    parameter BITWIDTH = 5'd8,
    parameter BITWIDTH_CMDS = 5'd2,
    parameter BITWIDTH_ADR = 5'd6,
    parameter BITWIDTH_DATA = 5'd16
)(
    // Global signals
    input wire                              CLK_SYS,
    input wire                              RSTN,
    // Control lines between middlware and FIFO
    input wire                              FIFO_RDY,
    input wire [BITWIDTH* FIFO_SIZE-'d1:0]  FIFO_DIN,
    output reg [BITWIDTH* FIFO_SIZE-'d1:0]  FIFO_DOUT,
    // Output signals
    output reg                              LED_CONTROL,
    output wire                             DUT_DO_TEST,
    output reg [BITWIDTH_DATA-'d1:0]        DUT_DIN,
    input wire [BITWIDTH_DATA-'d1:0]        DUT_DOUT,
    output wire [BITWIDTH_ADR-'d1:0]        DUT_ADR,
    output wire                             DUT_RnW
);
    localparam REG_DUT_CNTL = 2'd0, REG_DUT_WR = 2'd1, REG_DUT_RD = 2'd2, REG_HEADER = 2'd3;

    reg [1:0] shift_drdy;
    reg [1:0] shift_test;
    reg trigger_test;
    wire do_update_data;

    wire [BITWIDTH_CMDS-'d1:0] sel_cmds;
    wire [BITWIDTH_DATA-'d1:0] sel_data;

    // --- Data handler for Test module
    assign do_update_data = shift_drdy[0] && ~shift_drdy[1];
    assign DUT_DO_TEST = shift_test[0] ^ shift_test[1];
    assign DUT_RnW = ~(sel_cmds == REG_DUT_WR);
    assign sel_cmds = FIFO_DIN[(BITWIDTH_DATA+BITWIDTH_ADR)+:BITWIDTH_CMDS];
    assign DUT_ADR = FIFO_DIN[BITWIDTH_DATA+:BITWIDTH_ADR];
    assign sel_data =  FIFO_DIN[0+:BITWIDTH_DATA];

    // --- Implemented data protocol
    always@(posedge CLK_SYS) begin
        if(~RSTN) begin
            shift_drdy <= 2'd3;
            shift_test <= 2'd0;
            FIFO_DOUT <= 'd0;

            LED_CONTROL <= 1'd0;
            trigger_test <= 1'd0;
            DUT_DIN <= 'd0;
        end else begin
            shift_drdy <= {shift_drdy[0], FIFO_RDY};
            shift_test <= {shift_test[0], trigger_test};
            FIFO_DOUT <= (do_update_data) ? FIFO_DIN : FIFO_DOUT;

            LED_CONTROL     <= (do_update_data && sel_cmds == REG_DUT_CNTL && |DUT_ADR[2:1]) ? ((DUT_ADR[2]) ? ~LED_CONTROL : sel_data[0])  : LED_CONTROL;
            trigger_test    <= (do_update_data && sel_cmds == REG_DUT_CNTL && DUT_ADR[0])   ? ~trigger_test : trigger_test;
            DUT_DIN         <= (do_update_data && sel_cmds == REG_DUT_WR)                   ? sel_data      : DUT_DIN;
        end
    end
endmodule
