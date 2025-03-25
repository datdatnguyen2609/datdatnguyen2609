// NguyenThanhDat - datdatnguyen2609@gmail.com
module traffic_light_controller(
    input clk,    // Tin hieu clock
    input rst,    // Tin hieu reset
    output reg [2:0] car_light, // Den giao thong cho xe (Xanh - Vang - Do)
    output reg pedestrian_light, // Den giao thong cho nguoi di bo (Bat/Tat)
    output wire [7:0] seg_display // Hien thi thoi gian con lai tren LED 7 doan
);

    // Dinh nghia cac trang thai cua den giao thong
    parameter GREEN = 2'b00, YELLOW = 2'b01, RED = 2'b10;
    reg [1:0] state, next_state; // Luu tru trang thai hien tai va trang thai tiep theo
    
    reg [3:0] max_time; // Thoi gian toi da cho moi trang thai
    wire [3:0] seconds; // Gia tri dem thoi gian
    wire done; // Tin hieu ket thuc thoi gian cua trang thai hien tai
    wire [3:0] display_value; // Gia tri hien thi tren LED 7 doan

    // Module bo dem thoi gian
    counter count_inst (
        .clk(clk),
        .reset(rst),
        .max_count(max_time),
        .seconds(seconds),
        .done(done)
    );

    // Cap nhat trang thai hien tai theo clock
    always @(posedge clk) begin
        if (rst) begin
            state <= GREEN; // Khi reset, bat dau tu trang thai Xanh
        end else begin
            state <= next_state;
        end
    end

    // Xac dinh trang thai tiep theo dua vao trang thai hien tai va tin hieu 'done'
    always @(*) begin
        case (state)
            GREEN:  next_state = (done) ? YELLOW : GREEN;
            YELLOW: next_state = (done) ? RED : YELLOW;
            RED:    next_state = (done) ? GREEN : RED;
            default: next_state = GREEN;
        endcase
    end

    // Dieu khien den giao thong va dat thoi gian toi da tuong ung moi trang thai
    always @(*) begin
        case (state)
            GREEN: begin
                car_light = 3'b100;  // Den xanh bat
                pedestrian_light = 0; // Den nguoi di bo tat
                max_time = 9; // Thoi gian xanh 9 giay
            end
            YELLOW: begin
                car_light = 3'b010;  // Den vang bat
                pedestrian_light = 0; // Den nguoi di bo tat
                max_time = 3; // Thoi gian vang 3 giay
            end
            RED: begin
                car_light = 3'b001;  // Den do bat
                pedestrian_light = 1; // Den nguoi di bo bat
                max_time = 6; // Thoi gian do 6 giay
            end
            default: begin
                car_light = 3'b000;
                pedestrian_light = 0;
                max_time = 9;
            end
        endcase
    end
    
    // Tinh toan gia tri hien thi tren LED 7 doan
    assign display_value = max_time - seconds;

    // Module dieu khien hien thi LED 7 doan
    seven_seg_controller seg_inst (
        .value(display_value),
        .seg_out(seg_display)
    );
endmodule
/*
  Giải thích chức năng:
  Đây là đoạn code thể hiện chức năng tạo ra mạch điều khiển đèn giao thông (xanh 9s, vàng 3s, đỏ 6s) và đèn dành cho người đi bộ (xanh khi đèn cho phương tiện giao thông chuyển đỏ, và ngược lại)
  
  Đoạn code này còn có chức năng kết nối với bộ điều khiển led 7 thanh khi kết hợp với module seven_seg_controller để hiển thị, và kết nối mạch đếm (có cờ done mỗi khi đếm xong giá trị của 1 đèn)
  để tính toán thời gian hiển thị lên led 7 thanh và chuyển mốc tín hiệu từ xanh -> vàng -> đỏ

  Phần code được thiết kế dưới dạng FSM dạng Mealy (để đảm bảo tính tức thời của hệ thống khi đầu ra có thể nhận ngay tín hiệu thay đổi của đầu vào): 
  - FSM sử dụng tín hiệu "done" (đến từ bộ đếm) để quyết định trạng thái tiếp theo.
  - Các tín hiệu đầu ra (car_light, pedestrian_light, max_time) thay đổi ngay khi trạng thái thay đổi (không có bộ thanh ghi lưu trạng thái output riêng).
  - Dưới đây là mô hình hóa hoạt động của FSM này:
  
          [S0: GREEN]   
          car_light = 3'b100   
          pedestrian_light = 0   
          max_time = 9s   
                |    
                | (done = 1)  
                v  
          [S1: YELLOW]  
          car_light = 3'b010  
          pedestrian_light = 0  
          max_time = 3s  
                |  
                | (done = 1)  
                v  
          [S2: RED]  
          car_light = 3'b001  
          pedestrian_light = 1  
          max_time = 6s  
                |  
                | (done = 1)  
                v  
          (Quay lại S0)  
*/
