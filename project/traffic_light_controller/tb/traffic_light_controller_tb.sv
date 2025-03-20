
module traffic_light_controller_tb;
    reg clk;
    reg rst;
    wire [2:0] car_light;
    wire pedestrian_light;
    wire [7:0] seg_display;

    traffic_light_controller traffic_light_controller_inst (
        .clk(clk),
        .rst(rst),
        .car_light(car_light),
        .pedestrian_light(pedestrian_light),
        .seg_display(seg_display)
    );

    always #5_000_000 clk = ~clk;

    initial begin
        clk = 0;
        rst = 1;
        #20_000_000; 
        
        rst = 0; 
        #200_000_000;    
        
        $finish;   
    end

    initial begin
        $monitor("Time = %0t | Car Light = %b | Pedestrian Light = %b | 7-seg = %b", 
                 $time, car_light, pedestrian_light, seg_display);
    end
endmodule
