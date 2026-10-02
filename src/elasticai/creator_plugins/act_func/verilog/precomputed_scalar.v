module ACT_PRECOMPUTED_SCALAR #(
    parameter integer BITWIDTH_IN  = 4,
    parameter integer BITWIDTH_OUT = 4
)(
    input  wire signed [BITWIDTH_IN-1:0]  A,
    output reg  signed [BITWIDTH_OUT-1:0] Q
);

    localparam integer NUM_VALUES = 3;
    localparam signed [(NUM_VALUES-1)*BITWIDTH_IN-1:0] VALUES_X = { 4'sd4, -4'sd6 };
    localparam signed [NUM_VALUES*BITWIDTH_OUT-1:0] VALUES_Y = { 4'sd1, 4'sd0, -4'sd1 };

    integer i;
    reg found;
    reg signed [BITWIDTH_IN-1:0]  threshold_i;
    reg signed [BITWIDTH_OUT-1:0] value_i;

    always @(*) begin
        found = 1'b0;
        Q     = VALUES_Y[(NUM_VALUES*BITWIDTH_OUT-1) -: BITWIDTH_OUT];
        for (i = 0; i < NUM_VALUES-1; i = i + 1) begin
            threshold_i = VALUES_X[i*BITWIDTH_IN +: BITWIDTH_IN];
            value_i     = VALUES_Y[i*BITWIDTH_OUT +: BITWIDTH_OUT];
            if (!found && A <= threshold_i) begin
                Q     = value_i;
                found = 1'b1;
            end
        end
    end
endmodule