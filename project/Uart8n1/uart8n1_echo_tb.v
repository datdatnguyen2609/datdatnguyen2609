`timescale 1ns/1ps

module uart_echo_tb;
    // Parameters
    parameter CLK_PERIOD = 20; // 50MHz clock (20ns period)
    parameter BIT_PERIOD = 104167; // 9600 baud (104.167 μs per bit)
    
    // Testbench signals
    reg clk;
    reg reset;
    reg tb_tx_start;
    reg [7:0] tb_tx_data;
    wire tb_tx_done;
    wire echo_tx_done;
    
    wire rx_done;
    wire [7:0] rx_data;
    
    wire tb_to_dut;  // Test bench to DUT
    wire dut_to_tb;  // DUT to Test bench
    
    reg echo_tx_start;
    reg [7:0] echo_tx_data;
    
    // Test statistics
    integer sent_bytes = 0;
    integer received_bytes = 0;
    integer echoed_bytes = 0;
    integer correct_echo = 0;
    
    // Test data array
    reg [7:0] test_data [0:9];
    integer i;
    
    // Instantiate main UART system (DUT)
    uart_top dut_uart(
        .clk(clk),
        .reset(reset),
        .tx_start(echo_tx_start),    // Echo transmit start
        .tx_data(echo_tx_data),      // Echo transmit data
        .tx_done(echo_tx_done),      // Echo transmit done
        .rx_done(rx_done),           // Receive done
        .rx_data(rx_data),           // Received data
        .tx_line(dut_to_tb),         // TX line going to testbench (echo)
        .rx_line(tb_to_dut)          // RX line coming from testbench
    );
    
    // Instantiate test bench UART transmitter for stimulus
    uart_tx tb_transmitter(
        .clk(clk),
        .reset(reset),
        .baud_tick(baud_tick),
        .tx_start(tb_tx_start),
        .tx_data(tb_tx_data),
        .tx_done(tb_tx_done),
        .tx(tb_to_dut)
    );
    
    // Instantiate baud generator for testbench
    baud_generator tb_baud_gen(
        .clk(clk),
        .reset(reset),
        .baud_tick(baud_tick)
    );
    
    // Instantiate UART receiver for monitoring echo responses
    uart_rx tb_receiver(
        .clk(clk),
        .reset(reset),
        .baud_tick(baud_tick),
        .rx(dut_to_tb),
        .rx_done(tb_rx_done),
        .rx_data(tb_rx_data)
    );
    
    // Echo logic - when receiving data, send it back
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            echo_tx_start <= 0;
            echo_tx_data <= 0;
        end else begin
            echo_tx_start <= 0; // Default state
            
            if (rx_done) begin
                echo_tx_data <= rx_data; // Echo back what was received
                echo_tx_start <= 1;      // Start transmitting
                received_bytes = received_bytes + 1;
                $display("DUT Received: 0x%h at time %t", rx_data, $time);
            end
        end
    end
    
    // Monitor echoed data received by testbench
    always @(posedge clk) begin
        if (tb_rx_done) begin
            echoed_bytes = echoed_bytes + 1;
            $display("Echo Received: 0x%h at time %t", tb_rx_data, $time);
            
            // Verify if echo matches the original transmitted data
            if (tb_rx_data === test_data[echoed_bytes-1]) begin
                correct_echo = correct_echo + 1;
            end else begin
                $display("ERROR: Echo mismatch! Expected: 0x%h, Got: 0x%h", 
                         test_data[echoed_bytes-1], tb_rx_data);
            end
        end
    end
    
    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    // Test sequence
    initial begin
        // Initialize test data
        test_data[0] = 8'hA5;
        test_data[1] = 8'h3C;
        test_data[2] = 8'hFF;
        test_data[3] = 8'h00;
        test_data[4] = 8'h55;
        test_data[5] = 8'hAA;
        test_data[6] = 8'h12;
        test_data[7] = 8'h87;
        test_data[8] = 8'h39;
        test_data[9] = 8'hE4;
        
        // Initialize signals
        reset = 1;
        tb_tx_start = 0;
        tb_tx_data = 8'h00;
        
        // Reset the system
        #100;
        reset = 0;
        #100;
        
        // Send test bytes and wait for echo
        for (i = 0; i < 10; i = i + 1) begin
            // Send test byte
            tb_tx_data = test_data[i];
            tb_tx_start = 1;
            sent_bytes = sent_bytes + 1;
            $display("Sending: 0x%h at time %t", tb_tx_data, $time);
            
            #50;
            tb_tx_start = 0;
            
            // Wait for transmit to complete
            wait(tb_tx_done);
            
            // Wait for echo response
            wait(tb_rx_done);
            
            // Add delay between bytes
            #(BIT_PERIOD * 20);
        end
        
        // Wait for any remaining transactions
        #(BIT_PERIOD * 50);
        
        // Print test results
        $display("\n----- UART Echo Test Results -----");
        $display("Bytes sent: %d", sent_bytes);
        $display("Bytes received by DUT: %d", received_bytes);
        $display("Bytes echoed back: %d", echoed_bytes);
        $display("Correct echoes: %d", correct_echo);
        if (correct_echo == sent_bytes)
            $display("TEST PASSED: All bytes were correctly echoed");
        else
            $display("TEST FAILED: Some bytes were not correctly echoed");
        $display("---------------------------------\n");
        
        // End simulation
        $finish;
    end
    
    // Generate VCD waveform file
    initial begin
        $dumpfile("uart_echo_test.vcd");
        $dumpvars(0, uart_echo_tb);
    end

endmodule
