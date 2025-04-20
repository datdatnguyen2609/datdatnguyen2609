`timescale 1ns/1ps

module uart8n1_2404_tb;
    
    // Tinh hieu testbench
    reg clk;                   // Xung clock
    reg reset;                 // Tin hieu reset
    reg en;                    // Tin hieu cho phep
    reg tx_start;              // Tin hieu bat dau truyen
    reg [7:0] tx_data;         // Du lieu truyen di
    wire tx_busy;              // Tin hieu TX dang ban
    wire tx_done;              // Tin hieu TX hoan thanh
    wire rx_busy;              // Tin hieu RX dang ban
    wire rx_done;              // Tin hieu RX hoan thanh
    wire [7:0] rx_data;        // Du lieu nhan duoc
    wire tx_line, rx_line;     // Duong truyen UART

    // Ket noi TX voi RX de test loopback
    assign rx_line = tx_line;

    // Khoi tao module UART top
    uart_top dut (
        .clk(clk),
        .reset(reset),
        .en(en),
        .tx_start(tx_start),
        .tx_data(tx_data),
        .tx_busy(tx_busy),
        .tx_done(tx_done),
        .rx_busy(rx_busy),
        .rx_done(rx_done),
        .rx_data(rx_data),
        .tx_line(tx_line),
        .rx_line(rx_line)
    );

    // Tao xung clock
    always begin
        #10 clk = ~clk;  // Tao xung clock 50MHz
    end

    // Chuoi test
    initial begin
        // Khoi tao cac tin hieu
        clk = 0;
        reset = 1;
        en = 0;
        tx_start = 0;
        tx_data = 8'h00;

        // Reset va cho phep UART hoat dong
        #100 reset = 0;
        #50 en = 1;

        // Truyen tung ky tu trong cum tu
        #50;
        tx_data = "H"; tx_start = 1; #20 tx_start = 0; wait (tx_done);  // Truyen 'H'
        #50;
        tx_data = "e"; tx_start = 1; #20 tx_start = 0; wait (tx_done);  // Truyen 'e'
        #50;
        tx_data = "l"; tx_start = 1; #20 tx_start = 0; wait (tx_done);  // Truyen 'l'
        #50;
        tx_data = "l"; tx_start = 1; #20 tx_start = 0; wait (tx_done);  // Truyen 'l'
        #50;
        tx_data = "o"; tx_start = 1; #20 tx_start = 0; wait (tx_done);  // Truyen 'o'
        #50;
        tx_data = "!"; tx_start = 1; #20 tx_start = 0; wait (tx_done);  // Truyen '!'

        // Truyen cac chu so
        #50;
        tx_data = 8'd48; tx_start = 1; #20 tx_start = 0; wait (tx_done);  // Truyen '0'
        #50;
        tx_data = 8'd49; tx_start = 1; #20 tx_start = 0; wait (tx_done);  // Truyen '1'
        #50;
        tx_data = 8'd50; tx_start = 1; #20 tx_start = 0; wait (tx_done);  // Truyen '2'
        #50;
        tx_data = 8'd51; tx_start = 1; #20 tx_start = 0; wait (tx_done);  // Truyen '3'

        // Truyen cac ky tu dac biet
        #50;
        tx_data = 8'h00; tx_start = 1; #20 tx_start = 0; wait (tx_done);  // Truyen NULL
        #50;
        tx_data = 8'hFF; tx_start = 1; #20 tx_start = 0; wait (tx_done);  // Truyen 0xFF
        #50;
        tx_data = 8'hAA; tx_start = 1; #20 tx_start = 0; wait (tx_done);  // Truyen 0xAA
        #50;
        tx_data = 8'h55; tx_start = 1; #20 tx_start = 0; wait (tx_done);  // Truyen 0x55

        // Cho hoan tat truyen du lieu cuoi cung
        #200;

        // Ket thuc mo phong
        $display("UART Test hoan thanh");
        $finish;
    end

    // Ghi nhan du lieu nhan duoc
    always @(posedge clk) begin
        if (rx_done) begin
            $display("Thoi gian %0t: Nhan du lieu: 0x%h (%c)", $time, rx_data, rx_data);
        end
    end

endmodule
