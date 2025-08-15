// Demonstration Module for PWM Brightness Control
// This module shows a simple example of using the BCD to 7-segment 
// converter with PWM brightness control

module pwm_brightness_demo #(
    parameter CLK_FREQ = 125_000_000,
    parameter PWM_FREQ = 1000
)(
    input clk,                      // System clock (125MHz)
    input reset,                    // Reset signal
    input [2:0] brightness_switches, // Brightness control switches (0-7)
    input [3:0] bcd_switches,       // BCD input switches (0-15)
    output [7:0] seg_display,       // Original 7-segment output
    output [7:0] seg_display_pwm,   // PWM brightness controlled output
    output pwm_enable,              // PWM enable signal
    output [2:0] brightness_leds,   // LEDs showing current brightness level
    output valid_bcd                // LED indicating valid BCD (0-9)
);

    // BCD to 7-segment converter with PWM brightness control
    bcd2led7seg #(
        .CLK_FREQ(CLK_FREQ),
        .PWM_FREQ(PWM_FREQ)
    ) display_controller (
        .clk(clk),
        .reset(reset),
        .bcd_input(bcd_switches),
        .brightness_switches(brightness_switches),
        .seg_out(seg_display),
        .seg_out_pwm(seg_display_pwm),
        .pwm_enable(pwm_enable)
    );
    
    // Output brightness level on LEDs for visual feedback
    assign brightness_leds = brightness_switches;
    
    // Indicate valid BCD input (0-9)
    assign valid_bcd = (bcd_switches <= 4'd9);

endmodule