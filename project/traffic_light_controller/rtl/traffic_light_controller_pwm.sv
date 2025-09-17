// Enhanced Traffic Light Controller with PWM Brightness Control
// This module integrates the original traffic light functionality with PWM brightness control
// for the 7-segment display

module traffic_light_controller_pwm(
    input clk,                      // System clock (125MHz)
    input rst,                      // Reset signal
    input [2:0] brightness_switches, // Brightness control switches (0-7)
    output reg [2:0] car_light,     // Traffic light outputs
    output reg pedestrian_light,    // Pedestrian light output
    output wire [7:0] seg_display,  // Original 7-segment display
    output wire [7:0] seg_display_pwm, // PWM brightness controlled display
    output wire pwm_enable          // PWM enable signal (for debugging)
);

    parameter GREEN = 2'b00, YELLOW = 2'b01, RED = 2'b10;
    reg [1:0] state, next_state;
    
    reg [3:0] max_time;
    wire [3:0] seconds;
    wire done;
    wire [3:0] display_value; 
    
    // Original counter instance
    counter count_inst (
        .clk(clk),
        .reset(rst),
        .max_count(max_time),
        .seconds(seconds),
        .done(done)
    );

    // Traffic light state machine (unchanged)
    always @(posedge clk) begin
        if (rst) begin
            state <= GREEN;
        end else begin
            state <= next_state;
        end
    end

    always @(*) begin
        case (state)
            GREEN:  next_state = (done) ? YELLOW : GREEN;
            YELLOW: next_state = (done) ? RED : YELLOW;
            RED:    next_state = (done) ? GREEN : RED;
            default: next_state = GREEN;
        endcase
    end

    always @(*) begin
        case (state)
            GREEN: begin
                car_light = 3'b100;  
                pedestrian_light = 0; 
                max_time = 9;
            end
            YELLOW: begin
                car_light = 3'b010;  
                pedestrian_light = 0; 
                max_time = 3;
            end
            RED: begin
                car_light = 3'b001;  
                pedestrian_light = 1; 
                max_time = 6;
            end
            default: begin
                car_light = 3'b000;
                pedestrian_light = 0;
                max_time = 9;
            end
        endcase
    end
    
    assign display_value = max_time - seconds;
    
    // Enhanced 7-segment display with PWM brightness control
    bcd2led7seg #(
        .CLK_FREQ(125_000_000),
        .PWM_FREQ(1000)
    ) enhanced_display (
        .clk(clk),
        .reset(rst),
        .bcd_input(display_value),
        .brightness_switches(brightness_switches),
        .seg_out(seg_display),
        .seg_out_pwm(seg_display_pwm),
        .pwm_enable(pwm_enable)
    );

endmodule