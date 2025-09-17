// Testbench for BCD to 7-Segment Display with PWM Brightness Control
// Tests PWM functionality, brightness levels, and BCD conversion

`timescale 1ns / 1ps

module bcd2led7seg_tb;

    // Test parameters
    localparam CLK_PERIOD = 8;  // 8ns = 125MHz clock
    localparam TEST_DURATION = 50_000_000; // 50ms test duration
    
    // Testbench signals
    reg clk;
    reg reset;
    reg [3:0] bcd_input;
    reg [2:0] brightness_switches;
    wire [7:0] seg_out;
    wire [7:0] seg_out_pwm;
    wire pwm_enable;
    
    // Instantiate the Device Under Test (DUT)
    bcd2led7seg #(
        .CLK_FREQ(125_000_000),
        .PWM_FREQ(1000)
    ) dut (
        .clk(clk),
        .reset(reset),
        .bcd_input(bcd_input),
        .brightness_switches(brightness_switches),
        .seg_out(seg_out),
        .seg_out_pwm(seg_out_pwm),
        .pwm_enable(pwm_enable)
    );
    
    // Clock generation (125MHz)
    always #(CLK_PERIOD/2) clk = ~clk;
    
    // Test stimulus
    initial begin
        // Initialize signals
        clk = 0;
        reset = 1;
        bcd_input = 4'd0;
        brightness_switches = 3'd0;
        
        // Wait for a few clock cycles
        #(10 * CLK_PERIOD);
        
        // Release reset
        reset = 0;
        #(10 * CLK_PERIOD);
        
        $display("=== Starting BCD to 7-Segment PWM Brightness Control Test ===");
        $display("Time\t\tBCD\tBrightness\tSeg_Out\t\tSeg_PWM\t\tPWM_En");
        
        // Test different BCD values with various brightness levels
        test_bcd_values();
        
        // Test PWM brightness levels
        test_brightness_levels();
        
        // Test edge cases
        test_edge_cases();
        
        $display("=== Test Completed ===");
        $finish;
    end
    
    // Task to test different BCD values
    task test_bcd_values();
        integer i;
        begin
            $display("\n--- Testing BCD Values (0-9) with Medium Brightness ---");
            brightness_switches = 3'd4; // 50% brightness
            
            for (i = 0; i <= 9; i = i + 1) begin
                bcd_input = i;
                #(1000 * CLK_PERIOD); // Wait 1000 clock cycles
                $display("%0t\t%0d\t%0d\t\t%b\t%b\t%0d", 
                         $time, bcd_input, brightness_switches, seg_out, seg_out_pwm, pwm_enable);
            end
            
            // Test invalid BCD (should display blank)
            bcd_input = 4'd15;
            #(1000 * CLK_PERIOD);
            $display("%0t\t%0d\t%0d\t\t%b\t%b\t%0d", 
                     $time, bcd_input, brightness_switches, seg_out, seg_out_pwm, pwm_enable);
        end
    endtask
    
    // Task to test different brightness levels
    task test_brightness_levels();
        integer i;
        reg [31:0] pwm_high_count, pwm_low_count;
        begin
            $display("\n--- Testing Brightness Levels (0-7) with BCD=8 ---");
            bcd_input = 4'd8; // Display '8' for all segments
            
            for (i = 0; i <= 7; i = i + 1) begin
                brightness_switches = i;
                pwm_high_count = 0;
                pwm_low_count = 0;
                
                // Monitor PWM for 1ms (one PWM period at 1kHz)
                repeat (125_000) begin // 125,000 clocks = 1ms at 125MHz
                    @(posedge clk);
                    if (pwm_enable)
                        pwm_high_count = pwm_high_count + 1;
                    else
                        pwm_low_count = pwm_low_count + 1;
                end
                
                $display("Brightness Level %0d: PWM High=%0d, Low=%0d, Duty=%.1f%%", 
                         i, pwm_high_count, pwm_low_count, 
                         (pwm_high_count * 100.0) / (pwm_high_count + pwm_low_count));
            end
        end
    endtask
    
    // Task to test edge cases
    task test_edge_cases();
        begin
            $display("\n--- Testing Edge Cases ---");
            
            // Test reset during operation
            bcd_input = 4'd5;
            brightness_switches = 3'd6;
            #(5000 * CLK_PERIOD);
            $display("Before reset: BCD=%0d, Brightness=%0d, PWM=%0d", 
                     bcd_input, brightness_switches, pwm_enable);
            
            reset = 1;
            #(100 * CLK_PERIOD);
            $display("During reset: PWM should be 0: PWM=%0d", pwm_enable);
            
            reset = 0;
            #(1000 * CLK_PERIOD);
            $display("After reset: PWM restored: PWM=%0d", pwm_enable);
            
            // Test rapid brightness changes
            $display("\nTesting rapid brightness changes:");
            bcd_input = 4'd3;
            repeat (8) begin
                brightness_switches = brightness_switches + 1;
                #(10000 * CLK_PERIOD); // 10000 clocks = 80μs
                $display("Brightness=%0d, PWM=%0d", brightness_switches, pwm_enable);
            end
        end
    endtask
    
    // Monitor for debugging
    always @(posedge clk) begin
        // This can be enabled for detailed debugging
        // $monitor("Time=%0t, BCD=%0d, Brightness=%0d, Seg=%b, PWM_Seg=%b, PWM=%0d", 
        //          $time, bcd_input, brightness_switches, seg_out, seg_out_pwm, pwm_enable);
    end

endmodule