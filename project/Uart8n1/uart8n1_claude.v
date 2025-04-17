// Baud Rate Generator
module baud_generator(
    input wire clk,          // System clock input
    input wire reset,        // Reset signal
    output reg baud_tick     // Baud rate tick output
);
    // For 9600 bps from a 50MHz clock, we need a divisor of 50,000,000 / 9600 / 16 = ~326
    // Using 16x oversampling for better reliability
    parameter DIVISOR = 326;
    reg [15:0] counter;
    
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            counter <= 0;
            baud_tick <= 0;
        end else begin
            if (counter == DIVISOR - 1) begin
                counter <= 0;
                baud_tick <= 1;
            end else begin
                counter <= counter + 1;
                baud_tick <= 0;
            end
        end
    end
endmodule

// UART Transmitter (TX)
module uart_tx(
    input wire clk,          // System clock
    input wire reset,        // Reset signal
    input wire baud_tick,    // Baud rate tick
    input wire tx_start,     // Start transmission
    input wire [7:0] tx_data, // Data to transmit
    output reg tx_done,      // Transmission complete
    output reg tx            // Serial output
);
    // States
    parameter IDLE = 2'b00;
    parameter START = 2'b01;
    parameter DATA = 2'b10;
    parameter STOP = 2'b11;
    
    reg [1:0] state;
    reg [2:0] bit_counter;   // Counts 0-7 (8 bits)
    reg [3:0] tick_counter;  // Counts ticks (16 per bit)
    reg [7:0] data_reg;      // Register to hold the data
    
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            bit_counter <= 0;
            tick_counter <= 0;
            tx_done <= 1;
            tx <= 1;         // Line is high when idle
        end else begin
            case (state)
                IDLE: begin
                    tx <= 1;  // Line is high when idle
                    tick_counter <= 0;
                    bit_counter <= 0;
                    
                    if (tx_start && tx_done) begin
                        data_reg <= tx_data;
                        tx_done <= 0;
                        state <= START;
                    end
                end
                
                START: begin
                    tx <= 0;  // Start bit is low
                    
                    if (baud_tick) begin
                        if (tick_counter == 15) begin
                            tick_counter <= 0;
                            state <= DATA;
                        end else begin
                            tick_counter <= tick_counter + 1;
                        end
                    end
                end
                
                DATA: begin
                    tx <= data_reg[bit_counter];  // Transmit each bit
                    
                    if (baud_tick) begin
                        if (tick_counter == 15) begin
                            tick_counter <= 0;
                            
                            if (bit_counter == 7) begin  // All 8 bits sent
                                bit_counter <= 0;
                                state <= STOP;
                            end else begin
                                bit_counter <= bit_counter + 1;
                            end
                        end else begin
                            tick_counter <= tick_counter + 1;
                        end
                    end
                end
                
                STOP: begin
                    tx <= 1;  // Stop bit is high
                    
                    if (baud_tick) begin
                        if (tick_counter == 15) begin
                            tx_done <= 1;
                            state <= IDLE;
                        end else begin
                            tick_counter <= tick_counter + 1;
                        end
                    end
                end
            endcase
        end
    end
endmodule

// UART Receiver (RX)
module uart_rx(
    input wire clk,          // System clock
    input wire reset,        // Reset signal
    input wire baud_tick,    // Baud rate tick
    input wire rx,           // Serial input
    output reg rx_done,      // Reception complete
    output reg [7:0] rx_data // Received data
);
    // States
    parameter IDLE = 2'b00;
    parameter START = 2'b01;
    parameter DATA = 2'b10;
    parameter STOP = 2'b11;
    
    reg [1:0] state;
    reg [2:0] bit_counter;   // Counts 0-7 (8 bits)
    reg [3:0] tick_counter;  // Counts ticks (16 per bit)
    reg [7:0] data_reg;      // Register to hold the data
    
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            bit_counter <= 0;
            tick_counter <= 0;
            rx_done <= 0;
            rx_data <= 0;
        end else begin
            case (state)
                IDLE: begin
                    rx_done <= 0;
                    tick_counter <= 0;
                    bit_counter <= 0;
                    
                    if (rx == 0) begin  // Start bit detected
                        state <= START;
                    end
                end
                
                START: begin
                    if (baud_tick) begin
                        if (tick_counter == 7) begin  // Sample at middle of start bit
                            if (rx == 0) begin  // Confirm it's a start bit
                                tick_counter <= 0;
                                state <= DATA;
                            end else begin
                                state <= IDLE;  // False start bit
                            end
                        end else begin
                            tick_counter <= tick_counter + 1;
                        end
                    end
                end
                
                DATA: begin
                    if (baud_tick) begin
                        if (tick_counter == 15) begin  // Sample at middle of data bit
                            tick_counter <= 0;
                            data_reg[bit_counter] <= rx;
                            
                            if (bit_counter == 7) begin  // All 8 bits received
                                bit_counter <= 0;
                                state <= STOP;
                            end else begin
                                bit_counter <= bit_counter + 1;
                            end
                        end else begin
                            tick_counter <= tick_counter + 1;
                        end
                    end
                end
                
                STOP: begin
                    if (baud_tick) begin
                        if (tick_counter == 15) begin
                            rx_done <= 1;
                            rx_data <= data_reg;
                            state <= IDLE;
                        end else begin
                            tick_counter <= tick_counter + 1;
                        end
                    end
                end
            endcase
        end
    end
endmodule

// Top UART Module
module uart_top(
    input wire clk,           // System clock
    input wire reset,         // Reset signal
    input wire tx_start,      // Start transmission
    input wire [7:0] tx_data, // Data to transmit
    output wire tx_done,      // Transmission complete
    output wire rx_done,      // Reception complete
    output wire [7:0] rx_data,// Received data
    output wire tx_line,      // Serial output
    input wire rx_line        // Serial input
);
    wire baud_tick;
    
    // Instantiate baud rate generator
    baud_generator baud_gen(
        .clk(clk),
        .reset(reset),
        .baud_tick(baud_tick)
    );
    
    // Instantiate UART transmitter
    uart_tx transmitter(
        .clk(clk),
        .reset(reset),
        .baud_tick(baud_tick),
        .tx_start(tx_start),
        .tx_data(tx_data),
        .tx_done(tx_done),
        .tx(tx_line)
    );
    
    // Instantiate UART receiver
    uart_rx receiver(
        .clk(clk),
        .reset(reset),
        .baud_tick(baud_tick),
        .rx(rx_line),
        .rx_done(rx_done),
        .rx_data(rx_data)
    );
endmodule
