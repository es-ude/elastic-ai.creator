//////////////////////////////////////////////////////////////////////////////////
// Company:         University of Duisburg-Essen, Intelligent Embedded Systems Lab
// Engineer:        AE
//
// Create Date:     01.06.2019 17:06:19
// Copied on: 	    §{date_copy_created}
// Module Name:     Template for building a Pulse Width Modulator (PWM)
// Target Devices:  ASIC / FPGA
// Tool Versions:   1v0
// Processing:      Working on posedge of CLK
// Dependencies:    None
//
// State: 	        Works!
// Improvements:    None
// Parameters:      PERIOD_NUM_CYCLE --> Number of cycles to build PWM period
//////////////////////////////////////////////////////////////////////////////////

module PULSE_WIDTH_MODULATOR#(
    parameter PERIOD_NUM_CYCLE = 8'd10
)(
    input wire CLK_SYS,
    input wire RSTN,
    input wire EN,
    input wire [$clog2(PERIOD_NUM_CYCLE)-'d1:0] DUTY_CYCLE,
    output reg PWM_STREAM
);
    
    reg [$clog2(PERIOD_NUM_CYCLE)-'d1:0] pwm_cnt;
    
    always@(posedge CLK_SYS) begin
        if(~(EN && RSTN)) begin
            pwm_cnt <= 'd0;
            PWM_STREAM <= 1'd0;
        end else begin
            if(pwm_cnt == (PERIOD_NUM_CYCLE - 'd1)) begin
                pwm_cnt <= 'd0;
                PWM_STREAM <= 1'd1;
            end else begin
                pwm_cnt <= pwm_cnt + 'd1;
                PWM_STREAM <= (pwm_cnt == DUTY_CYCLE - 'd1) ? 1'd0 : PWM_STREAM;
            end
        end
    end
endmodule
