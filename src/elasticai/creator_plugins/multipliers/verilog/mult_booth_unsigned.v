//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     13.11.2020 09:22:46
// Copied on: 	    §{date_copy_created}
// Module Name:     Clock- and bit-wise Multiplier for unsigned Input
// Target Devices:  ASIC, FPGA
// Tool Versions:   1v0
// Description:     LUT-based multiplication with parametrized bitwidth
// Processing:      Direct processing
// Dependencies:    None
//
// State: 	        Works!
// Dependency:      None
// Improvements:    None
// Parameters:      BITWIDTH --> Bitwidth of input data
//
//////////////////////////////////////////////////////////////////////////////////


module MULT_UNSIGNED#(
    parameter BITWIDTH = 16
)(
    input  wire                     CLK,
    input  wire                     RSTN,
    input  wire                     START_FLAG,

    input  wire [BITWIDTH-1:0]      A,
    input  wire [BITWIDTH-1:0]      B,

    output reg                      DRDY,
    output reg [2*BITWIDTH-1:0]     Q
);

    localparam CNT_WIDTH = $clog2(BITWIDTH);

    reg [2*BITWIDTH-1:0] multiplicand;
    reg [BITWIDTH-1:0]   multiplier;
    reg [2*BITWIDTH-1:0] product;
    reg [CNT_WIDTH:0] count;
    reg busy;

    always @(posedge CLK) begin
        if (!RSTN) begin
            multiplicand <= 0;
            multiplier   <= 0;
            product      <= 0;
            count <= 0;
            busy  <= 0;
            DRDY <= 0;
            Q    <= 0;
        end else begin
            // Start multiplication
            if (START_FLAG && !busy) begin
                DRDY <= 1'b0;
                multiplicand <= {{BITWIDTH{1'b0}}, A};
                multiplier   <= B;
                product <= 0;
                count <= 0;
                busy  <= 1'b1;
            end
            // Iterative multiply
            else if (busy) begin
                // Add current multiplicand if LSB is 1
                if (multiplier[0]) begin
                    product <= product + multiplicand;
                end
                // Shift operands
                multiplicand <= multiplicand << 1;
                multiplier   <= multiplier >> 1;
                // Iteration counter
                count <= count + 1'b1;
                // Finish
                if (count == BITWIDTH-1) begin
                    busy <= 1'b0;
                    if (multiplier[0])
                        Q <= product + multiplicand;
                    else
                        Q <= product;
                    DRDY <= 1'b1;
                end
            end
        end
    end
endmodule