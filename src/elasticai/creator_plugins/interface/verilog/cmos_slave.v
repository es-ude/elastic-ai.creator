//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     07.12.2022 08:18:52
// Copied on: 	    {$date_copy_created}
// Module Name:     CMOS Interface Slave Modul for Data Transmission
// Target Devices:  FPGA
// Tool Versions:   1v1
// Description:     Performing a data transmission from source to target
// Processing:      Transmitting LSB first, setting on rising edge, sensing on falling edge
// Dependencies:    None
//
// State: 	        Works!
// Improvements:    None
// Parameters:      BIT_WIDTH --> Bitwidth of input signed data
//                  NUMBER_CH --> Number of channels for data transmission
//////////////////////////////////////////////////////////////////////////////////


module CMOS_INTERFACE_SLAVE#(
    parameter BIT_WIDTH = 5'd12,
    parameter NUMBER_CH = 5'd7
)(
    input wire CLK,
    input wire nRST,
    input wire EN,
    input wire TX_CLK,
    input wire [NUMBER_CH-'d1:0] TX_DATA,
    output wire [NUMBER_CH* BIT_WIDTH-'d1:0] DATA_OUT,
    output wire EOC
);
    localparam STATE_IDLE = 1'd0, STATE_RECEIVE = 1'd1;

    reg state;
    reg clk_dly;
    reg [$clog2(BIT_WIDTH):0] cnt_trans;
    reg [BIT_WIDTH-'d1:0] data [NUMBER_CH-'d1:0];
    
    wire start_transmission;
    assign EOC = (state == STATE_IDLE);         

    integer i0;
    always@(posedge CLK) begin
        if(!nRST) begin
            state <= STATE_IDLE;
            cnt_trans <= 'd0;
            clk_dly <= 1'd0;
            for(i0 = 'd0; i0 < NUMBER_CH; i0 + 'd1) begin
                data[i0] <= 'd0;
            end
        end else begin           
            case(state)
                STATE_IDLE: begin
                    state <= (EN) ? STATE_RECEIVE : STATE_IDLE;
                end
                STATE_RECEIVE: begin
                    clk_dly <= TX_CLK;
                    if(!TX_CLK && clk_dly) begin
                        cnt_trans <= cnt_trans + ((TX_CLK) ? 'd1 : 'd0);
                        state <= (cnt_trans == BIT_WIDTH-'d1) ? STATE_IDLE : STATE_RECEIVE;
                        data <= data;
                        for(i0 = 'd0; i0 < NUMBER_CH; i0 + 'd1) begin
                            data[i0] <= {TX_DATA[i0], data[i0][BIT_WIDTH-'d1:'d1]};
                        end
                    end else begin
                        cnt_trans <= cnt_trans;
                        state <= state;
                        for(i0 = 'd0; i0 < NUMBER_CH; i0 + 'd1) begin
                            data[i0] <= data[i0];
                        end
                    end                           
                end
            endcase    
        end    
    end
endmodule
