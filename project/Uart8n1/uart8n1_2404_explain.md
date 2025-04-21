# Giải thích chi tiết code UART (Verilog)

Đoạn code này mô tả một hệ thống UART hoàn chỉnh gồm các module con: bộ tạo xung baud (`baud_generator`), bộ phát UART (`uart_tx`), bộ nhận UART (`uart_rx`), và module tổng hợp (`uart_top`). Đây là giải thích chi tiết từng phần:

---

## 1. **Bộ tạo xung Baud (`baud_generator`)**

### Mục đích
Tạo ra xung baud (baud tick) từ xung đồng hồ hệ thống. Xung baud này điều tiết tốc độ truyền/nhận dữ liệu UART.

### Nguyên lý hoạt động
- **Tính toán bộ chia:**  
  - Giả sử dùng clock 50MHz, baudrate 9600bps, lấy mẫu x16:  
    `DIVISOR = 50,000,000 / 9600 / 16 ≈ 326`
- **Hoạt động:**  
  - Nếu `en` (enable) và `counter` đạt giá trị `DIVISOR-1`, phát xung baud (`baud_tick = 1`) rồi reset counter về 0.
  - Nếu không, counter tăng dần.

### Lưu ý
- Lấy mẫu 16 lần/bit để tăng độ tin cậy (giảm lỗi lấy mẫu).
- `baud_tick` chỉ lên mức 1 trong 1 chu kỳ clock khi đến thời điểm lấy mẫu kế tiếp.

---

## 2. **Bộ phát UART (`uart_tx`)**

### Mục đích
Chuyển đổi dữ liệu song song 8 bit thành tín hiệu UART nối tiếp (gồm: bit start, 8 bit data, bit stop).

### Thành phần
- **Các trạng thái FSM:**  
  - `IDLE`: Chờ lệnh truyền (`tx_start`).
  - `START`: Gửi bit start (logic 0).
  - `DATA`: Gửi 8 bit dữ liệu (LSB trước).
  - `STOP`: Gửi bit stop (logic 1).
- **Biến điều khiển:**  
  - `tx_busy`: Đang truyền.
  - `tx_done`: Đã truyền xong.

### Hoạt động
- Khi `tx_start` được kích hoạt ở trạng thái `IDLE`, nạp dữ liệu từ `tx_data` và chuyển sang gửi bit start.
- Mỗi trạng thái duy trì đủ số lượng tick (16 tick cho mỗi bit), rồi chuyển tiếp trạng thái.
- Khi truyền xong bit stop, báo hiệu `tx_done = 1` và trở lại trạng thái `IDLE`.

---

## 3. **Bộ nhận UART (`uart_rx`)**

### Mục đích
Nhận tín hiệu nối tiếp UART, chuyển về dữ liệu song song 8 bit.

### Thành phần
- **Các trạng thái FSM:**  
  - `IDLE`: Chờ tín hiệu start.
  - `START`: Xác nhận bit start (lấy mẫu ở giữa bit).
  - `DATA`: Nhận từng bit dữ liệu (8 bit, LSB trước).
  - `STOP`: Xác nhận bit stop, nếu hợp lệ thì báo nhận xong.
- **Đồng bộ tín hiệu:**  
  - Có bộ đồng bộ 2 tầng để chống metastability khi lấy tín hiệu `rx` ngoại vi vào clock nội bộ.

### Hoạt động
- Khi phát hiện cạnh xuống (bit start), chuyển sang trạng thái `START`, lấy mẫu giữa bit để chắc chắn.
- Với mỗi bit dữ liệu, lấy mẫu ở giữa bit, lưu vào thanh ghi dữ liệu.
- Sau khi nhận đủ 8 bit, kiểm tra bit stop (logic 1). Nếu hợp lệ thì `rx_done = 1` và xuất dữ liệu.

---

## 4. **Module tổng hợp UART (`uart_top`)**

### Mục đích
Kết nối các module con thành hệ thống UART hoàn chỉnh.

### Thành phần
- **Kết nối:**  
  - Khởi tạo và kết nối `baud_generator`, `uart_tx`, `uart_rx` với nhau.
  - Đầu ra gồm: tín hiệu busy/done cho cả TX và RX, dữ liệu nhận được, đường truyền UART (TX/RX lines).

### Sơ đồ kết nối:
```
clk, reset, en
   |         |
   |         v
   |  +---------------+
   |  | baud_generator| ---------> baud_tick
   |  +---------------+
   |           |                |
   |           v                v
   |  +---------------+   +---------------+
   |  |   uart_tx     |   |   uart_rx     |
   |  +---------------+   +---------------+
   |        |     |            |     |
   |       tx_line|           rx_line|
   |              v                  ^
   +-------------->------------------+
```

---

## Tổng kết

- **baud_generator**: Chia clock tạo nhịp baud định kỳ.
- **uart_tx**: Chuyển parallel thành serial (gửi).
- **uart_rx**: Chuyển serial thành parallel (nhận).
- **uart_top**: Ghép nối các module trên, tạo giao diện sử dụng cho hệ thống UART.

---

### **Giải thích từ khóa**
- **FSM (Finite State Machine)**: Máy trạng thái hữu hạn, là mô hình luồng điều khiển gồm các trạng thái chuyển tiếp.
- **LSB (Least Significant Bit)**: Bit ít quan trọng nhất (bit thấp nhất).
- **Metastability**: Hiện tượng bất ổn định khi tín hiệu đồng bộ hóa không đúng với clock nội bộ.

---

### **Ứng dụng**
Module UART này dùng để truyền nhận dữ liệu nối tiếp giữa vi mạch số (FPGA/MCU) và các thiết bị ngoại vi như máy tính, module Bluetooth, GPS, v.v.  
Chỉ cần nối `tx_line` và `rx_line` đến thiết bị UART ngoài.
