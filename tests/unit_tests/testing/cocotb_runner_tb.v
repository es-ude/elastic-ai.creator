//`define DEFINE_TEST


module COCOTB_TEST#(
    parameter BITWIDTH = 5'd12,
    `ifdef DEFINE_TEST
        parameter OFFSET = 5'd4,
    `endif
    parameter SCALE = 5'd1
)(
    input wire [BITWIDTH-'d1:0] A,
    output wire [BITWIDTH-'d1:0] Q
);
    `ifdef DEFINE_TEST
        assign Q = SCALE * A + OFFSET;
    `else
        assign Q = SCALE * A ;
    `endif
endmodule
