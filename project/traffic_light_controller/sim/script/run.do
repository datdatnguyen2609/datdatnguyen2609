vlib work
vlog ../../rtl/traffic_light_controller.sv
vlog ../../tb/traffic_light_controller_tb.sv
vsim -t 1ps -voptargs="+acc" work.traffic_light_controller_tb -wlf traffic_light_controller.wlf 
log -r *
run -all