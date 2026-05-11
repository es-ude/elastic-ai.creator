//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     13.11.2020 09:22:46
// Copied on: 	    §{date_copy_created}
// Module Name:     Clock- and bit-wise Multiplier for Signed Input
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


module MULT_SIGNED#(
    parameter BITWIDTH = 16
)(
    input  wire                         CLK,
    input  wire                         RSTN,
    input  wire                         START_FLAG,
    input  wire signed [BITWIDTH-1:0]  A,
    input  wire signed [BITWIDTH-1:0]  B,
    output reg                          DRDY,
    output reg signed [2*BITWIDTH-1:0] Q
);

    localparam CNT_WIDTH = $clog2(BITWIDTH+1);

    reg signed [BITWIDTH-1:0] HIGH;
    reg signed [BITWIDTH-1:0] LOW;
    reg                       LOW0;

    reg [CNT_WIDTH-1:0] count;
    reg                  busy;

    reg signed [BITWIDTH:0] temp;

    wire [1:0] booth_bits;
    assign booth_bits = {LOW[0], LOW0};

    always @(posedge CLK) begin
        if (!RSTN) begin
            HIGH <= 0;
            LOW  <= 0;
            LOW0 <= 0;
            count <= 0;
            busy  <= 0;
            DRDY <= 0;
            Q    <= 0;
        end else begin
            // Start multiplication
            if (START_FLAG && !busy) begin
                DRDY <= 1'b0;
                HIGH <= 0;
                LOW  <= B;
                LOW0 <= 0;
                count <= 0;
                busy  <= 1'b1;
            end
            // Booth iteration
            else if (busy) begin
                // Add/Subtract step
                case (booth_bits)
                    2'b01: temp = HIGH + A;
                    2'b10: temp = HIGH - A;
                    default: temp = HIGH;
                endcase
                // Arithmetic shift right
                {HIGH, LOW, LOW0} <=
                    $signed({temp, LOW, LOW0}) >>> 1;
                // Counter
                count <= count + 1'b1;
                // Done
                if (count == BITWIDTH-1) begin
                    busy <= 1'b0;
                    Q <= {
                        $signed({temp, LOW} >>> 1)
                    };
                    DRDY <= 1'b1;
                end
            end
        end
    end
endmodule