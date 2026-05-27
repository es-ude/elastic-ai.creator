//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     16.01.2023 12:44:18
// Copied on: 	    {$date_copy_created}
// Module Name:     CMOS Interface Master Modul for Data Transmission
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v1
// Description:     Performing a data transmission from source to target
// Processing:      Transmitting LSB first, setting on rising edge, sensing on falling edge
// Dependencies:    None
//
// State: 	        Works! (ASIC Implemented on 18/03/2023)
// Improvements:    None
// Parameters:      BIT_WIDTH --> Bitwidth of input signed data
//                  NUMBER_CH --> Number of channels for data transmission
//////////////////////////////////////////////////////////////////////////////////

// Data must concatenated like
// wire CMOS_DATA [BIT_WIDTH* NUMBER_CH] data_to_send;
// for(idx = 'd0; idx < NUMBER_CH; idx + idx + 'd1) begin
//      assign CMOS_DATA[idx*BIT_WIDTH+:BIT_WIDTH] = data[idx];
// end


module CMOS_INTERFACE_MASTER#(
    parameter BIT_WIDTH = 5'd12,
    parameter NUMBER_CH = 5'd7
)(
    input wire CLK,
    input wire nRST,
    input wire EN,
    input wire [BIT_WIDTH* NUMBER_CH-'d1:0] DATA_CH,
    input wire [NUMBER_CH-'d1:0] EOC,
    output reg [NUMBER_CH-'d1:0] TX_DATA,
    output reg TX_CLK
);
    localparam STATE_IDLE = 1'd0, STATE_TRANSMIT = 1'd1;
    
    reg sync_eoc;
    reg state;
    reg [$clog2(BIT_WIDTH):0] cnt_trans;
    
    wire start_transmission;
    assign start_transmission = !sync_eoc && (EOC == (NUMBER_CH){1'b1}) && EN;
        
    genvar i0;
    always@(posedge CLK) begin
        if(!nRST) begin
            sync_eoc <= 1'd0;
            cnt_trans <= 'd0;
            state <= STATE_IDLE;
            TX_DATA <= 'd0;
            TX_CLK <= 1'd0;
        end else begin
            sync_eoc <= {(EOC == (NUMBER_CH){1'b1})};
            
            case(state)
                STATE_IDLE: begin
                    state <= (start_transmission) ? STATE_TRANSMIT : STATE_IDLE;
                end
                STATE_TRANSMIT: begin
                    if(cnt_trans == BIT_WIDTH) begin
                        TX_CLK <= 1'd0;
                        TX_DATA <= 'd0;
                        state <= STATE_IDLE; 
                        cnt_trans <= 'd0;
                    end else begin
                        TX_CLK <= ~TX_CLK;
                        for(i0 = 'd0; i0 < NUMBER_CH; i0 = i0 + 'd1) begin
                            TX_DATA[i0] <= DATA_CH[i0* BIT_WIDTH + cnt_trans];
                        end
                        state <= STATE_TRANSMIT;
                        cnt_trans <= cnt_trans + ((TX_CLK) ? 'd1 : 'd0);
                    end                                      
                end
            endcase    
        end    
    end
endmodule
