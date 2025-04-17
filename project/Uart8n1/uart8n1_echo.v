// UART Echo System Module
module uart_echo_system(
    input wire clk,              // System clock
    input wire reset,            // Reset signal
    input wire rx_in,            // External serial input
    output wire tx_out,          // External serial output
    output wire [7:0] led_out    // LED output for visualization (optional)
);
    // Internal signals
    wire rx_done;
    wire [7:0] rx_data;
    reg tx_start;
    wire tx_done;
    reg [7:0] tx_data;
    
    // Optional LED output to display the last received byte
    assign led_out = rx_data;
    
    // Instantiate UART top module
    uart_top uart_module(
        .clk(clk),
        .reset(reset),
        .tx_start(tx_start),
        .tx_data(tx_data),
        .tx_done(tx_done),
        .rx_done(rx_done),
        .rx_data(rx_data),
        .tx_line(tx_out),
        .rx_line(rx_in)
    );
    
    // Echo logic - when receiving data, send it back
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            tx_start <= 0;
            tx_data <= 0;
        end else begin
            tx_start <= 0; // Default state
            
            if (rx_done) begin
                tx_data <= rx_data; // Echo back what was received
                tx_start <= 1;      // Start transmitting
            end
        end
    end
endmodule
