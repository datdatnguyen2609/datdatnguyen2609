
// 125Mhz -> 1hz
module clock_divider #(
    parameter N = 125_000_000 
)(
    input clk,        // Clock 125MHz
    input reset,      // Reset
    output reg clk_out // Clock 1Hz
);
    reg [$clog2(N/2)-1:0] counter;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            counter <= 0;
            clk_out <= 0;
        end else if (counter == (N/2 - 1)) begin
            counter <= 0;
            clk_out <= ~clk_out; 
        end else begin
            counter <= counter + 1;
        end
    end
endmodule

// ================================================================

module counter (
    input clk,          // clock form clock div
    input reset,
    input [3:0] max_count,  // time to count
    output reg [3:0] seconds, // remaining time
    output wire done    // flag when counting done
);
    reg [3:0] counter;

    always @(posedge clk) begin
        if(reset) begin
            counter <= 0;
        end
        else if ( counter < max_count) begin
            counter <= counter + 1;
        end  else if (counter == max_count)begin
            counter <= 0;
        end
    end
    assign seconds = counter;
    assign done = (counter == max_count);
endmodule

// ================================================================

module seven_seg_controller (
    input [3:0] value,    
    output reg [7:0] seg_out 
);
    always @(*) begin
        case (value)
            4'd0: seg_out = 8'b00111111; // 0
            4'd1: seg_out = 8'b00000110; // 1
            4'd2: seg_out = 8'b01011011; // 2
            4'd3: seg_out = 8'b01001111; // 3
            4'd4: seg_out = 8'b01100110; // 4
            4'd5: seg_out = 8'b01101101; // 5
            4'd6: seg_out = 8'b01111101; // 6
            4'd7: seg_out = 8'b00000111; // 7
            4'd8: seg_out = 8'b01111111; // 8
            4'd9: seg_out = 8'b01101111; // 9
            default: seg_out = 8'b00000000; // off
        endcase
    end
endmodule

// ================================================================

module traffic_light_controller(
    input clk,    
    input rst,       
    output reg [2:0] car_light, 
    output reg pedestrian_light, 
    output wire [7:0] seg_display 
);

    parameter GREEN = 2'b00, YELLOW = 2'b01, RED = 2'b10;
    reg [1:0] state, next_state;
    
    reg [3:0] max_time;
    wire [3:0] seconds;
    wire done;
    wire [3:0] display_value; 
    counter count_inst (
        .clk(clk),
        .reset(rst),
        .max_count(max_time),
        .seconds(seconds),
        .done(done)
    );

    always @(posedge clk) begin
        if (rst) begin
            state <= GREEN;
        end else begin
            state <= next_state;
        end
    end

    always @(*) begin
        case (state)
            GREEN:  next_state = (done) ? YELLOW : GREEN;
            YELLOW: next_state = (done) ? RED : YELLOW;
            RED:    next_state = (done) ? GREEN : RED;
            default: next_state = GREEN;
        endcase
    end

    always @(*) begin
        case (state)
            GREEN: begin
                car_light = 3'b100;  
                pedestrian_light = 0; 
                max_time = 9;
            end
            YELLOW: begin
                car_light = 3'b010;  
                pedestrian_light = 0; 
                max_time = 3;
            end
            RED: begin
                car_light = 3'b001;  
                pedestrian_light = 1; 
                max_time = 6;
            end
            default: begin
                car_light = 3'b000;
                pedestrian_light = 0;
                max_time = 9;
            end
        endcase
    end
    
    assign display_value = max_time - seconds;
    seven_seg_controller seg_inst (
        .value(display_value),
        .seg_out(seg_display)
    );
endmodule 