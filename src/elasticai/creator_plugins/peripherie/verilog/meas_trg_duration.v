//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     08.01.2024 16:55:01
// Copied on: 	    §{date_copy_created}
// Module Name:     Module for measuring the period of a digital trigger signal
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v0
// Processing:      Logical Design
//
// State: 	        Works!
// Dependencies:    None
// Improvements:    None
// Parameters:      None
//
//////////////////////////////////////////////////////////////////////////////////


module MEAS_TRG_DURATION#(
    parameter CNTWIDTH_TRG = 6'd10,
    parameter CNTWIDTH_OVR = 10'd2
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire START,
    input wire SIG,
    output reg [CNTWIDTH_OVR-'d1:0] CNT_RPT,
    output reg [CNTWIDTH_TRG + CNTWIDTH_OVR-'d1:0] CNT_PERIOD
);
    reg [1:0] dly_sig;
    reg [CNTWIDTH_OVR - 'd1:0] cnt_repeat;
    reg [CNTWIDTH_TRG + CNTWIDTH_OVR - 'd1:0] cnt_period;
    wire end_rpt;
    
    assign end_rpt = (dly_sig[1] && ~dly_sig[0]);

    always@(posedge CLK_SYS) begin
        if(~RSTN) begin
            dly_sig <= 2'd0;  
            CNT_RPT <= 'd0;
            CNT_PERIOD <= 'd0;
        end else begin
            dly_sig <= {dly_sig[0], SIG};
            CNT_RPT <= CNT_RPT + ((end_rpt) ? 'd1 : 'd0);
            CNT_PERIOD <= CNT_PERIOD + ((dly_sig[1]) ? 'd1 : 'd0);
        end
    end
endmodule
