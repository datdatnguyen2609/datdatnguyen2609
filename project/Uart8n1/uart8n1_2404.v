// Bo tao xung Baud
module baud_generator(
    input wire clk,          // Xung dong ho he thong
    input wire reset,        // Tin hieu reset
    input wire en,           // Tin hieu cho phep
    output reg baud_tick     // Xung baud rate
);
    // Voi 9600 bps tu dong ho 50MHz, can bo chia 50,000,000 / 9600 / 16 = ~326
    // Su dung lay mau x16 de tang do tin cay
    parameter DIVISOR = 326;
    reg [15:0] counter, counter_next;
    reg baud_tick_next;
    
    // Logic tuan tu
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            counter <= 0;
            baud_tick <= 0;
        end else begin
            counter <= counter_next;
            baud_tick <= baud_tick_next;
        end
    end
    
    // Logic to hop
    always @* begin
        counter_next = counter;
        baud_tick_next = 0;
        
        if (en) begin
            if (counter == DIVISOR - 1) begin
                counter_next = 0;
                baud_tick_next = 1;
            end else begin
                counter_next = counter + 1;
                baud_tick_next = 0;
            end
        end
    end
endmodule

// Bo phat UART (TX) voi cho phep va trang thai ban
module uart_tx(
    input wire clk,          // Dong ho he thong
    input wire reset,        // Tin hieu reset
    input wire en,           // Tin hieu cho phep
    input wire baud_tick,    // Xung baud rate
    input wire tx_start,     // Bat dau truyen
    input wire [7:0] tx_data, // Du lieu can truyen
    output reg tx_busy,      // Dang ban
    output reg tx_done,      // Truyen xong
    output reg tx            // Ngo ra noi tiep
);
    // Cac trang thai
    parameter IDLE = 2'b00;
    parameter START = 2'b01;
    parameter DATA = 2'b10;
    parameter STOP = 2'b11;
    
    // Khai bao thanh ghi
    reg [1:0] state_reg, state_next;
    reg [2:0] bit_counter_reg, bit_counter_next;
    reg [3:0] tick_counter_reg, tick_counter_next;
    reg [7:0] data_reg, data_next;
    reg tx_busy_next, tx_done_next, tx_next;
    
    // Logic tuan tu
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state_reg <= IDLE;
            bit_counter_reg <= 0;
            tick_counter_reg <= 0;
            data_reg <= 0;
            tx_busy <= 0;
            tx_done <= 1;
            tx <= 1;
        end else begin
            state_reg <= state_next;
            bit_counter_reg <= bit_counter_next;
            tick_counter_reg <= tick_counter_next;
            data_reg <= data_next;
            tx_busy <= tx_busy_next;
            tx_done <= tx_done_next;
            tx <= tx_next;
        end
    end
    
    // Logic to hop
    always @* begin
        // Mac dinh: giu nguyen gia tri hien tai
        state_next = state_reg;
        bit_counter_next = bit_counter_reg;
        tick_counter_next = tick_counter_reg;
        data_next = data_reg;
        tx_busy_next = tx_busy;
        tx_done_next = tx_done;
        tx_next = tx;
        
        if (en) begin
            case (state_reg)
                IDLE: begin
                    tx_next = 1;  // Duong truyen cao khi nghi
                    tick_counter_next = 0;
                    bit_counter_next = 0;
                    tx_busy_next = 0;
                    
                    if (tx_start && tx_done) begin
                        data_next = tx_data;
                        tx_done_next = 0;
                        tx_busy_next = 1;
                        state_next = START;
                    end
                end
                
                START: begin
                    tx_next = 0;  // Bit bat dau la thap
                    tx_busy_next = 1;
                    
                    if (baud_tick) begin
                        if (tick_counter_reg == 15) begin
                            tick_counter_next = 0;
                            state_next = DATA;
                        end else begin
                            tick_counter_next = tick_counter_reg + 1;
                        end
                    end
                end
                
                DATA: begin
                    tx_next = data_reg[bit_counter_reg];  // Truyen tung bit
                    tx_busy_next = 1;
                    
                    if (baud_tick) begin
                        if (tick_counter_reg == 15) begin
                            tick_counter_next = 0;
                            
                            if (bit_counter_reg == 7) begin  // Da gui du 8 bit
                                bit_counter_next = 0;
                                state_next = STOP;
                            end else begin
                                bit_counter_next = bit_counter_reg + 1;
                            end
                        end else begin
                            tick_counter_next = tick_counter_reg + 1;
                        end
                    end
                end
                
                STOP: begin
                    tx_next = 1;  // Bit ket thuc la cao
                    tx_busy_next = 1;
                    
                    if (baud_tick) begin
                        if (tick_counter_reg == 15) begin
                            tx_done_next = 1;
                            tx_busy_next = 0;
                            state_next = IDLE;
                        end else begin
                            tick_counter_next = tick_counter_reg + 1;
                        end
                    end
                end
            endcase
        end
    end
endmodule

// Bo nhan UART (RX) voi cho phep va trang thai ban
module uart_rx(
    input wire clk,          // Dong ho he thong
    input wire reset,        // Tin hieu reset
    input wire en,           // Tin hieu cho phep
    input wire baud_tick,    // Xung baud rate
    input wire rx,           // Ngo vao noi tiep
    output reg rx_busy,      // Dang nhan
    output reg rx_done,      // Nhan xong
    output reg [7:0] rx_data // Du lieu da nhan
);
    // Cac trang thai
    parameter IDLE = 2'b00;
    parameter START = 2'b01;
    parameter DATA = 2'b10;
    parameter STOP = 2'b11;
    
    // Khai bao thanh ghi
    reg [1:0] state_reg, state_next;
    reg [2:0] bit_counter_reg, bit_counter_next;
    reg [3:0] tick_counter_reg, tick_counter_next;
    reg [7:0] data_reg, data_next;
    reg rx_busy_next, rx_done_next;
    reg [7:0] rx_data_next;
    
    // Thanh ghi dong bo cho tin hieu rx
    reg [1:0] rx_sync_reg, rx_sync_next;
    
    // Bit rx da duoc dong bo
    wire rx_bit = rx_sync_reg[1];
    
    // Logic tuan tu
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state_reg <= IDLE;
            bit_counter_reg <= 0;
            tick_counter_reg <= 0;
            data_reg <= 0;
            rx_busy <= 0;
            rx_done <= 0;
            rx_data <= 0;
            rx_sync_reg <= 2'b11;
        end else begin
            state_reg <= state_next;
            bit_counter_reg <= bit_counter_next;
            tick_counter_reg <= tick_counter_next;
            data_reg <= data_next;
            rx_busy <= rx_busy_next;
            rx_done <= rx_done_next;
            rx_data <= rx_data_next;
            rx_sync_reg <= rx_sync_next;
        end
    end
    
    // Logic to hop
    always @* begin
        // Mac dinh: giu nguyen gia tri hien tai
        state_next = state_reg;
        bit_counter_next = bit_counter_reg;
        tick_counter_next = tick_counter_reg;
        data_next = data_reg;
        rx_busy_next = rx_busy;
        rx_done_next = rx_done;
        rx_data_next = rx_data;
        
        // Dong bo hoa mac dinh
        if (en) begin
            rx_sync_next = {rx_sync_reg[0], rx};
        end else begin
            rx_sync_next = rx_sync_reg;
        end
        
        // Logic may trang thai
        if (en) begin
            case (state_reg)
                IDLE: begin
                    rx_done_next = 0;
                    tick_counter_next = 0;
                    bit_counter_next = 0;
                    rx_busy_next = 0;
                    
                    if (rx_bit == 0) begin  // Phat hien bit bat dau
                        state_next = START;
                        rx_busy_next = 1;
                    end
                end
                
                START: begin
                    rx_busy_next = 1;
                    
                    if (baud_tick) begin
                        if (tick_counter_reg == 7) begin  // Lay mau o giua bit bat dau
                            if (rx_bit == 0) begin  // Xac nhan la bit bat dau
                                tick_counter_next = 0;
                                state_next = DATA;
                            end else begin
                                state_next = IDLE;
                                rx_busy_next = 0;
                            end
                        end else begin
                            tick_counter_next = tick_counter_reg + 1;
                        end
                    end
                end
                
                DATA: begin
                    rx_busy_next = 1;
                    
                    if (baud_tick) begin
                        if (tick_counter_reg == 15) begin  // Lay mau o giua bit du lieu
                            tick_counter_next = 0;
                            data_next[bit_counter_reg] = rx_bit;
                            
                            if (bit_counter_reg == 7) begin  // Da nhan du 8 bit
                                bit_counter_next = 0;
                                state_next = STOP;
                            end else begin
                                bit_counter_next = bit_counter_reg + 1;
                            end
                        end else begin
                            tick_counter_next = tick_counter_reg + 1;
                        end
                    end
                end
                
                STOP: begin
                    rx_busy_next = 1;
                    
                    if (baud_tick) begin
                        if (tick_counter_reg == 15) begin
                            // Chi chap nhan du lieu neu bit ket thuc hop le (cao)
                            if (rx_bit == 1) begin
                                rx_done_next = 1;
                                rx_data_next = data_reg;
                            end
                            
                            rx_busy_next = 0;
                            state_next = IDLE;
                        end else begin
                            tick_counter_next = tick_counter_reg + 1;
                        end
                    end
                end
            endcase
        end
    end
endmodule

// Module UART tong the voi cho phep
module uart_top(
    input wire clk,           // Dong ho he thong
    input wire reset,         // Tin hieu reset
    input wire en,            // Cho phep toan bo
    
    // Giao dien TX
    input wire tx_start,      // Bat dau truyen
    input wire [7:0] tx_data, // Du lieu can truyen
    output wire tx_busy,      // Chi thi TX dang ban
    output wire tx_done,      // Truyen xong
    
    // Giao dien RX
    output wire rx_busy,      // Chi thi RX dang ban
    output wire rx_done,      // Nhan xong
    output wire [7:0] rx_data,// Du lieu da nhan
    
    // Duong truyen UART
    output wire tx_line,      // Ngo ra noi tiep
    input wire rx_line        // Ngo vao noi tiep
);
    wire baud_tick;
    
    // Khoi tao bo tao xung baud
    baud_generator baud_gen(
        .clk(clk),
        .reset(reset),
        .en(en),
        .baud_tick(baud_tick)
    );
    
    // Khoi tao bo phat UART
    uart_tx transmitter(
        .clk(clk),
        .reset(reset),
        .en(en),
        .baud_tick(baud_tick),
        .tx_start(tx_start),
        .tx_data(tx_data),
        .tx_busy(tx_busy),
        .tx_done(tx_done),
        .tx(tx_line)
    );
    
    // Khoi tao bo nhan UART
    uart_rx receiver(
        .clk(clk),
        .reset(reset),
        .en(en),
        .baud_tick(baud_tick),
        .rx(rx_line),
        .rx_busy(rx_busy),
        .rx_done(rx_done),
        .rx_data(rx_data)
    );
endmodule
