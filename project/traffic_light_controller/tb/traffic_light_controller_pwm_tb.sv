// Testbench for Enhanced Traffic Light Controller with PWM Brightness Control

`timescale 1ns / 1ps

module traffic_light_controller_pwm_tb;

    // Test parameters
    localparam CLK_PERIOD = 8;  // 8ns = 125MHz clock
    
    // Testbench signals
    reg clk;
    reg rst;
    reg [2:0] brightness_switches;
    wire [2:0] car_light;
    wire pedestrian_light;
    wire [7:0] seg_display;
    wire [7:0] seg_display_pwm;
    wire pwm_enable;
    
    // Instantiate the Device Under Test (DUT)
    traffic_light_controller_pwm dut (
        .clk(clk),
        .rst(rst),
        .brightness_switches(brightness_switches),
        .car_light(car_light),
        .pedestrian_light(pedestrian_light),
        .seg_display(seg_display),
        .seg_display_pwm(seg_display_pwm),
        .pwm_enable(pwm_enable)
    );
    
    // Clock generation (125MHz)
    always #(CLK_PERIOD/2) clk = ~clk;
    
    // Function to decode 7-segment display to digit
    function [3:0] decode_7seg;
        input [7:0] seg;
        begin
            case (seg)
                8'b00111111: decode_7seg = 0;
                8'b00000110: decode_7seg = 1;
                8'b01011011: decode_7seg = 2;
                8'b01001111: decode_7seg = 3;
                8'b01100110: decode_7seg = 4;
                8'b01101101: decode_7seg = 5;
                8'b01111101: decode_7seg = 6;
                8'b00000111: decode_7seg = 7;
                8'b01111111: decode_7seg = 8;
                8'b01101111: decode_7seg = 9;
                default: decode_7seg = 15; // Invalid/blank
            endcase
        end
    endfunction
    
    // Test stimulus
    initial begin
        // Initialize signals
        clk = 0;
        rst = 1;
        brightness_switches = 3'd4; // Start with medium brightness
        
        // Wait for a few clock cycles
        #(10 * CLK_PERIOD);
        
        // Release reset
        rst = 0;
        #(10 * CLK_PERIOD);
        
        $display("=== Enhanced Traffic Light Controller with PWM Brightness Test ===");
        $display("Testing traffic light states with PWM-controlled 7-segment display");
        $display("");
        $display("Time\t\tState\tCar_Light\tPed_Light\tDisplay\tBrightness\tPWM");
        
        // Test for a few seconds to see state transitions
        test_traffic_states();
        
        // Test brightness control during operation
        test_brightness_control();
        
        $display("\n=== Test Completed ===");
        $finish;
    end
    
    // Task to test traffic light states
    task test_traffic_states();
        integer cycle_count;
        reg [7:0] last_seg_display;
        begin
            $display("\n--- Testing Traffic Light States ---");
            cycle_count = 0;
            last_seg_display = 8'hFF;
            
            // Monitor for 20 seconds (enough to see full cycle)
            repeat (20 * 125_000_000 / 10000) begin // Reduced time scale for simulation
                @(posedge clk);
                cycle_count = cycle_count + 1;
                
                // Print state every 1000 cycles
                if (cycle_count % 1000 == 0) begin
                    $display("%0t\t%s\t%b\t\t%0d\t\t%0d\t%0d\t\t%0d", 
                             $time, 
                             (car_light == 3'b100) ? "GREEN " : 
                             (car_light == 3'b010) ? "YELLOW" : 
                             (car_light == 3'b001) ? "RED   " : "OFF   ",
                             car_light, pedestrian_light, 
                             decode_7seg(seg_display), brightness_switches, pwm_enable);
                end
            end
        end
    endtask
    
    // Task to test brightness control
    task test_brightness_control();
        integer i;
        begin
            $display("\n--- Testing Brightness Control During Operation ---");
            
            // Test all brightness levels
            for (i = 0; i <= 7; i = i + 1) begin
                brightness_switches = i;
                #(10000 * CLK_PERIOD); // Wait a bit
                $display("Brightness Level %0d: Display=%0d, PWM=%0d", 
                         i, decode_7seg(seg_display), pwm_enable);
            end
            
            // Test rapid brightness changes
            $display("\nTesting rapid brightness changes:");
            repeat (16) begin
                brightness_switches = brightness_switches + 1;
                #(5000 * CLK_PERIOD);
                if (brightness_switches <= 7)
                    $display("Brightness=%0d, PWM=%0d", brightness_switches, pwm_enable);
            end
        end
    endtask
    
    // Continuous monitoring for debugging
    always @(posedge clk) begin
        // Log significant events
        // Uncomment for detailed debugging
        /*
        if ($time % 1000000 == 0) begin
            $display("DEBUG: Time=%0t, State=%b, Display=%0d, Brightness=%0d, PWM=%0d", 
                     $time, car_light, decode_7seg(seg_display), brightness_switches, pwm_enable);
        end
        */
    end

endmodule