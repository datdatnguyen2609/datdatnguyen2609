// BCD to 7-Segment Display Controller with PWM Brightness Control
// This module combines BCD to 7-segment conversion with PWM brightness control
// Features:
// - BCD input (4-bit) to 7-segment display conversion
// - PWM brightness control with 8 levels (0-7)
// - 1kHz PWM frequency to avoid visible flicker
// - Switch-controlled brightness levels

module bcd2led7seg #(
    parameter CLK_FREQ = 125_000_000,  // Input clock frequency (125MHz)
    parameter PWM_FREQ = 1000          // PWM frequency (1kHz)
)(
    input clk,                         // System clock (125MHz)
    input reset,                       // Reset signal
    input [3:0] bcd_input,             // BCD input (0-9)
    input [2:0] brightness_switches,   // Brightness control switches (0-7)
    output [7:0] seg_out,              // 7-segment display output
    output [7:0] seg_out_pwm,          // PWM-controlled 7-segment output
    output pwm_enable                  // PWM enable signal (for debugging)
);

    // Internal signals
    wire [7:0] seg_decoded;            // Decoded 7-segment patterns
    wire pwm_signal;                   // PWM brightness control signal
    
    // Instantiate the seven segment controller (reusing existing module)
    seven_seg_controller seg_controller (
        .value(bcd_input),
        .seg_out(seg_decoded)
    );
    
    // Instantiate the PWM brightness controller
    pwm_brightness_controller #(
        .CLK_FREQ(CLK_FREQ),
        .PWM_FREQ(PWM_FREQ),
        .RESOLUTION(8)
    ) pwm_controller (
        .clk(clk),
        .reset(reset),
        .brightness_level(brightness_switches),
        .pwm_out(pwm_signal)
    );
    
    // Apply PWM to the 7-segment output
    // When PWM is high, display the segment pattern
    // When PWM is low, turn off all segments
    assign seg_out_pwm = pwm_signal ? seg_decoded : 8'b00000000;
    
    // Also provide non-PWM output for comparison/debugging
    assign seg_out = seg_decoded;
    
    // Expose PWM enable signal for debugging
    assign pwm_enable = pwm_signal;

endmodule