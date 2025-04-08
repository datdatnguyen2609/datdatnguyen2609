`timescale 1ns/1ps

module uart8n1_tb;

  // Parameters
  parameter CLOCK_RATE = 100_000_000; // 100 MHz
  parameter BAUD_RATE = 9600;
  parameter CLOCK_PERIOD = 1_000_000_000 / CLOCK_RATE; // in ns (because timescale is 1ns)

  // Testbench Signals
  reg clk = 0;
  reg rxEn = 1;
  reg rx = 1;
  wire rxBusy;
  wire rxDone;
  wire rxErr;
  wire [7:0] out;
  reg txEn = 1;
  reg txStart = 0;
  reg [7:0] in = 8'h00;
  wire txBusy;
  wire txDone;
  wire tx;

  // Instantiate the UART module
  uart8n1 #(
    .CLOCK_RATE(CLOCK_RATE),
    .BAUD_RATE(BAUD_RATE)
  ) uut (
    .clk(clk),
    .rxEn(rxEn),
    .rx(rx),
    .rxBusy(rxBusy),
    .rxDone(rxDone),
    .rxErr(rxErr),
    .out(out),
    .txEn(txEn),
    .txStart(txStart),
    .in(in),
    .txBusy(txBusy),
    .txDone(txDone),
    .tx(tx)
  );

  // Clock Generation
  always #(CLOCK_PERIOD / 2) clk = ~clk;

  // Test Procedure
  initial begin
    // Wait for system to stabilize
    #(10 * CLOCK_PERIOD);

    // Transmit a byte
    in = 8'hA5; // Example data byte
    txStart = 1;
    #(CLOCK_PERIOD);
    txStart = 0;

    // Wait for transmission to complete
    wait (txDone);

    // Simulate reception by looping back transmitted data
    rx = tx;

    // Wait for reception to complete
    wait (rxDone);

    // Check received data
    if (out == 8'hA5) begin
      $display("Test Passed: Received data matches transmitted data.");
    end else begin
      $display("Test Failed: Received data does not match transmitted data.");
    end

    // End simulation
    $finish;
  end

endmodule
