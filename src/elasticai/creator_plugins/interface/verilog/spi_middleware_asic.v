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
    // Use-specific GPIO / Data Handling
    output reg LED
);
    // ---------- Data Preparing (Packet Structure)
    localparam NUM_BITS_HEADER = 'd8;
    localparam NUM_BITS_DATA = BITWIDTH - NUM_BITS_HEADER;

    wire [NUM_BITS_HEADER-'d1:0] register;
    assign register = data_spi_rx_used[(BITWIDTH-'d1)-:NUM_BITS_HEADER];
    wire [NUM_BITS_DATA-'d1:0] data;
    assign data = data_spi_rx_used[0+:NUM_BITS_DATA];

    // ---------- Data Preparing (MSB)
    wire [BITWIDTH-'d1:0] data_spi_rx_used;
    reg [BITWIDTH-'d1:0] data_spi_tx_used;

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

    // ---------- Data Processing and Data Handling
    // --- Data Protocol: Use-specific code
    localparam REG_ECHO = 'd0, REG_SYSTEM = 'd1;

    always@(posedge SPI_DRDY or negedge RSTN) begin
        if(~RSTN) begin
            data_spi_tx_used = {(BITWIDTH){1'd0}};
            LED <= 1'd0;
        end else begin
            data_spi_tx_used <= {register, data};
            //data_spi_tx_used <= (register == REG_ECHO) ? {register, data} : data_spi_tx_used;
            LED <= (register == REG_SYSTEM) ? ((data[1]) ? ~LED : data[0]) : LED;
        end
    end
endmodule
