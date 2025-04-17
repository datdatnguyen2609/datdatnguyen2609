`timescale 1ns/1ps

module uart_tb;
    // Parameters
    parameter CLK_PERIOD = 20; // 50MHz clock (20ns period)
    parameter BIT_PERIOD = 104167; // 9600 baud (104.167 μs per bit)
    
    // Testbench signals
    reg clk;
    reg reset;
    reg tx_start;
    reg [7:0] tx_data;
    wire tx_done;
    wire rx_done;
    wire [7:0] rx_data;
    wire tx_to_rx; // Loopback connection
    
    // Instantiate UART top module
    uart_top uart_dut(
        .clk(clk),
        .reset(reset),
        .tx_start(tx_start),
        .tx_data(tx_data),
        .tx_done(tx_done),
        .rx_done(rx_done),
        .rx_data(rx_data),
        .tx_line(tx_to_rx),
        .rx_line(tx_to_rx) // Loopback: TX output connected to RX input
    );
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Test sequence
    initial begin
        // Initialize signals
        reset = 1;
        tx_start = 0;
        tx_data = 8'h00;
        
        // Reset the system
        #100;
        reset = 0;
        #100;
        
        // Test case 1: Send byte 0xA5
        tx_data = 8'hA5;
        tx_start = 1;
        #50;
        tx_start = 0;
        
        // Wait for transmission and reception to complete
        wait(rx_done);
        #1000;
        
        // Verify received data
        if (rx_data === 8'hA5) 
            $display("Test case 1: PASSED - Received data: 0x%h", rx_data);
        else
            $display("Test case 1: FAILED - Expected: 0xA5, Received: 0x%h", rx_data);
        
        // Test case 2: Send byte 0x3C
        #10000;
        tx_data = 8'h3C;
        tx_start = 1;
        #50;
        tx_start = 0;
        
        // Wait for transmission and reception to complete
        wait(rx_done);
        #1000;
        
        // Verify received data
        if (rx_data === 8'h3C) 
            $display("Test case 2: PASSED - Received data: 0x%h", rx_data);
        else
            $display("Test case 2: FAILED - Expected: 0x3C, Received: 0x%h", rx_data);
            
        // Test case 3: Send byte 0xFF
        #10000;
        tx_data = 8'hFF;
        tx_start = 1;
        #50;
        tx_start = 0;
        
        // Wait for transmission and reception to complete
        wait(rx_done);
        #1000;
        
        // Verify received data
        if (rx_data === 8'hFF) 
            $display("Test case 3: PASSED - Received data: 0x%h", rx_data);
        else
            $display("Test case 3: FAILED - Expected: 0xFF, Received: 0x%h", rx_data);
        
        // Test case 4: Send byte 0x00
        #10000;
        tx_data = 8'h00;
        tx_start = 1;
        #50;
        tx_start = 0;
        
        // Wait for transmission and reception to complete
        wait(rx_done);
        #1000;
        
        // Verify received data
        if (rx_data === 8'h00) 
            $display("Test case 4: PASSED - Received data: 0x%h", rx_data);
        else
            $display("Test case 4: FAILED - Expected: 0x00, Received: 0x%h", rx_data);
        
        // End simulation
        #10000;
        $display("UART Testing Complete");
        $finish;
    end
    
    // Generate VCD waveform file for viewing in waveform viewer
    initial begin
        $dumpfile("uart_test.vcd");
        $dumpvars(0, uart_tb);
    end

endmodule
