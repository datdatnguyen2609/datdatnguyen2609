// PWM Brightness Controller Module
// Generates PWM signals for 7-segment display brightness control
// Supports 8 brightness levels (0-7) with 1kHz PWM frequency

module pwm_brightness_controller #(
    parameter CLK_FREQ = 125_000_000,  // Input clock frequency (125MHz)
    parameter PWM_FREQ = 1000,         // PWM frequency (1kHz)
    parameter RESOLUTION = 8           // PWM resolution (8 levels = 3 bits)
)(
    input clk,                         // System clock (125MHz)
    input reset,                       // Reset signal
    input [2:0] brightness_level,      // Brightness level (0-7)
    output reg pwm_out                 // PWM output signal
);

    // Calculate PWM period and step counts
    localparam PWM_PERIOD = CLK_FREQ / PWM_FREQ;  // 125,000 clock cycles for 1kHz
    localparam PWM_STEP = PWM_PERIOD / RESOLUTION; // 15,625 clocks per brightness step
    
    // Internal counters
    reg [$clog2(PWM_PERIOD)-1:0] pwm_counter;
    reg [$clog2(PWM_PERIOD)-1:0] duty_threshold;
    
    // Calculate duty cycle threshold based on brightness level
    always @(*) begin
        case (brightness_level)
            3'd0: duty_threshold = 0;                    // 0% duty cycle
            3'd1: duty_threshold = PWM_STEP;             // 12.5% duty cycle
            3'd2: duty_threshold = 2 * PWM_STEP;         // 25% duty cycle
            3'd3: duty_threshold = 3 * PWM_STEP;         // 37.5% duty cycle
            3'd4: duty_threshold = 4 * PWM_STEP;         // 50% duty cycle
            3'd5: duty_threshold = 5 * PWM_STEP;         // 62.5% duty cycle
            3'd6: duty_threshold = 6 * PWM_STEP;         // 75% duty cycle
            3'd7: duty_threshold = 7 * PWM_STEP;         // 87.5% duty cycle
            default: duty_threshold = 0;
        endcase
    end
    
    // PWM counter and output generation
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            pwm_counter <= 0;
            pwm_out <= 0;
        end else begin
            // Increment counter
            if (pwm_counter >= PWM_PERIOD - 1) begin
                pwm_counter <= 0;
            end else begin
                pwm_counter <= pwm_counter + 1;
            end
            
            // Generate PWM output
            if (brightness_level == 3'd0) begin
                pwm_out <= 0;  // Always off for brightness level 0
            end else if (brightness_level == 3'd7) begin
                pwm_out <= 1;  // Always on for maximum brightness
            end else begin
                pwm_out <= (pwm_counter < duty_threshold) ? 1'b1 : 1'b0;
            end
        end
    end

endmodule