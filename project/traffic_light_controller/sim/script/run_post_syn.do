vlib work
vmap work work
vlog /home/lib/cells.v
vlog ../../syn/outputs/traffic_light_controller_net.v
vlog ../../tb/traffic_light_controller_tb.sv
vsim -t 1ps -voptargs="+acc" -sdfmax /traffic_light_controller_inst=../../syn/outputs/traffic_light_controller_net.sdf -sdfnoerror -L work +no_neg_tcheck work.traffic_light_controller_tb -wlf traffic_light_controller_net.wlf 
log -r *
run -all