//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     19.03.2022 20:03:48
// Copied on: 	    §{date_copy_created}
// Module Name:     Middleware / Datahandler of each SPI Data Trasmission
// Target Devices:  ASIC
// Tool Versions:   1v0
// Processing:      Protocol Handling
// Dependencies:    None
//
// State: 	        Tested!
// Improvements:    None
// Parameters:      BITWIDTH    --> Bitwidth/size of spi packet
//                  MSB         --> Transmission mode if MSB first (1) else LSB first (0)
//////////////////////////////////////////////////////////////////////////////////

//`define USE_ASIC_IMPLEMENTATION

module SPI_MIDDLEWARE#(
    parameter BITWIDTH = 6'd8,
    parameter MSB = 1'd1
)(
    // Global signal
    input wire CLK_SYS,
    input wire RSTN,
    input wire SPI_DRDY,
    input wire [BITWIDTH-'d1:0] SPI_DATA_RX,
    output wire [BITWIDTH-'d1:0] SPI_DATA_TX,
    // Data Handling
    output reg LED_TEST
);

    //---------- Code-Block for FPGA
    `ifndef USE_ASIC_IMPLEMENTATION
        reg [1:0] shift_reg;
        wire spi_new_data;
        wire [BITWIDTH-'d1:0] data_spi_rx_used;
        reg [BITWIDTH-'d1:0] data_spi_tx_used;
        assign spi_new_data = shift_reg[0] && !shift_reg[1];

        if(MSB) begin
            assign data_spi_rx_used = SPI_DATA_RX;
            assign SPI_DATA_TX = data_spi_tx_used;
        end else begin
            genvar i0;
            for (i0 = 'd0; i0 < BITWIDTH; i0 = i0 + 'd1) begin
                assign data_spi_rx_used[i0] = SPI_DATA_RX[BITWIDTH -i0 -'d1];
                assign SPI_DATA_TX[i0] = data_spi_tx_used[BITWIDTH -i0 -'d1];
            end
        end

        always@(posedge CLK_SYS) begin
            if(!RSTN) begin
                shift_reg <= 2'd0;
                data_spi_tx_used <= {(BITWIDTH){1'd0}};
                LED_TEST <= 1'd0;
            end else begin
                shift_reg <= {shift_reg[0], SPI_DRDY};
                // Use-case-specific code
                data_spi_tx_used <= (spi_new_data) ? data_spi_rx_used : data_spi_tx_used;
                LED_TEST <= (spi_new_data && data_spi_rx_used[0]) ? ~LED_TEST : LED_TEST;
            end
        end
    `else
        //---------- Code-Block for ASIC
        always@(posedge SPI_DRDY or negedge RSTN) begin
            if~(RSTN) begin
                data_spi_tx_used = {(BITWIDTH){1'd0}};
                LED_TEST <= 1'd0;
            end else begin
                // Use-case-specific code
                data_spi_tx_used = data_spi_rx_used;
                LED_TEST <= (data_spi_rx_used[0]) ? ~LED_TEST : LED_TEST;
            end
        end
    `endif
endmodule
