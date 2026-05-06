//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     27.04.2026, 14:31
// Copied on: 	    §{date_copy_created}
// Module Name:     Ripple-Carry Adder, signed
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v0
// Processing:      Logical Design
//
// State: 	        Works!
// Dependencies:    ADDER_HALF, ADDER_FULL
// Improvements:    None
// Parameters:      BITWIDTH
//
//////////////////////////////////////////////////////////////////////////////////


module ADDER_RIPPLE_CARRY_SIGNED#(
	parameter BITWIDTH = 'd4
)(
	input wire signed [BITWIDTH-'d1:0] A, B,
	output wire signed [BITWIDTH:0] Q
);
	wire [BITWIDTH-'d1:0] carry;
	assign Q[BITWIDTH] = carry[BITWIDTH-'d1] ^ A[BITWIDTH-'d1] ^ B[BITWIDTH-'d1];

	ADDER_HALF HA (A[0], B[0], carry[0], Q[0]);
	genvar i0;
	generate
		for(i0 = 'd1; i0 < BITWIDTH; i0 = i0 + 'd1) begin
			ADDER_FULL FA (
                .A(A[i0]),
                .B(B[i0]),
                .Cin(carry[i0-'d1]),
                .Q(Q[i0]),
                .Cout(carry[i0])
            );
		end
	endgenerate
endmodule
