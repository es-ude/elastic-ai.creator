//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     02.01.2024 20:49:02
// Copied on: 	    §{date_copy_created}
// Module Name:     Template for building a Pulse Density Modulator (PDM, First order)
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v0
// Processing:      Working on posedge of CLK
// Dependencies:    None
//
// State: 	        Works!
// Improvements:    None
// Parameters:      MOD_ORDER --> Modulator Order
//                  CNTWIDTH_CLK --> Bitsize of the counter for generating streaming clock
//////////////////////////////////////////////////////////////////////////////////

module PULSE_DENSITY_MODULATOR#(
    parameter MOD_ORDER=5'd4,
    parameter CNTWIDTH_CLK=5'd12
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire [MOD_ORDER-'d1:0] REF_VAL,
    input wire [CNTWIDTH_CLK-'d1:0] REF_CLK,
    output wire CLK_STREAM,
    output wire PDM_STREAM
);

    //--- CLK generation
    reg [CNTWIDTH_CLK-'d1:0] cnt_clk;
    assign CLK_STREAM = EN && (cnt_clk == 'd0);
    always@(posedge CLK_SYS) begin
        if(~(RSTN && EN)) begin
            cnt_clk <= 'd0;
        end else begin
            if(cnt_clk == REF_CLK - 'd1) begin
                cnt_clk <= 'd0;
            end else begin
                cnt_clk <= cnt_clk + 'd1;
            end
        end
    end

    // --- Pulse Pattern Generation
    reg signed [MOD_ORDER:0] mod_out;
    assign PDM_STREAM = EN && ~mod_out[MOD_ORDER];
    always@(posedge CLK_SYS) begin
        if(~RSTN) begin
            mod_out <= 'd0;
        end else begin
            mod_out <= (CLK_STREAM) ? (mod_out + {1'd0, REF_VAL} - {PDM_STREAM, {(MOD_ORDER){1'd0}}}) : mod_out;
        end
    end
endmodule
