module uart_8n1 (
    input wire clk,          // System clock
    input wire rst,          // Reset signal
    input wire rx,           // UART receive line
    output wire tx,          // UART transmit line
    input wire [7:0] tx_data,// Data to be transmitted
    input wire tx_start,     // Signal to start transmission
    output reg tx_busy,      // Transmission busy flag
    output reg [7:0] rx_data,// Received data
    output reg rx_ready      // Data received flag
);

    parameter BAUD_RATE = 9600;
    parameter CLOCK_FREQ = 50000000; // 50 MHz clock
    localparam BAUD_COUNTER_MAX = CLOCK_FREQ / BAUD_RATE;

    reg [15:0] baud_counter;
    reg [3:0] bit_counter;
    reg [7:0] shift_reg;
    reg rx_reg, rx_sync;
    reg tx_reg;
    reg [7:0] rx_shift_reg;

    // Baud rate generator
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            baud_counter <= 0;
        end else if (baud_counter == BAUD_COUNTER_MAX - 1) begin
            baud_counter <= 0;
        end else begin
            baud_counter <= baud_counter + 1;
        end
    end

    // UART receiver
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            rx_reg <= 1'b1;
            rx_sync <= 1'b1;
            bit_counter <= 0;
            rx_ready <= 0;
        end else if (baud_counter == BAUD_COUNTER_MAX - 1) begin
            rx_sync <= rx;
            rx_reg <= rx_sync;
            if (rx_reg == 0 && bit_counter == 0) begin
                bit_counter <= 1;
            end else if (bit_counter > 0 && bit_counter < 9) begin
                rx_shift_reg <= {rx_reg, rx_shift_reg[7:1]};
                bit_counter <= bit_counter + 1;
            end else if (bit_counter == 9) begin
                rx_data <= rx_shift_reg;
                rx_ready <= 1;
                bit_counter <= 0;
            end
        end
    end

    // UART transmitter
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            tx_reg <= 1'b1;
            tx_busy <= 0;
            bit_counter <= 0;
        end else if (tx_start && !tx_busy) begin
            tx_busy <= 1;
            shift_reg <= tx_data;
            bit_counter <= 1;
        end else if (tx_busy && baud_counter == BAUD_COUNTER_MAX - 1) begin
            if (bit_counter == 1) begin
                tx_reg <= 0; // Start bit
                bit_counter <= bit_counter + 1;
            end else if (bit_counter > 1 && bit_counter < 10) begin
                tx_reg <= shift_reg[0];
                shift_reg <= shift_reg >> 1;
                bit_counter <= bit_counter + 1;
            end else if (bit_counter == 10) begin
                tx_reg <= 1; // Stop bit
                tx_busy <= 0;
                bit_counter <= 0;
            end
        end
    end

    assign tx = tx_reg;

endmodule
