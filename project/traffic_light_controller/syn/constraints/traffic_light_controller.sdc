set sdc_version 1.4


# Set the current design
current_design traffic_light_controller

create_clock -name "clk" -add -period 10.0 -waveform {0.0 5.0} [get_ports clk]

set_false_path -from [list \
  [get_ports rst_n]]

# set_false_path -hold -through [get_pins PM_INST/clk_enable]

# set_input_delay -clock [get_clocks clk] -add_delay 0.3 [get_ports n]
# set_input_delay -clock [get_clocks clk] -add_delay 0.3 [get_ports d]
# set_input_delay -clock [get_clocks clk] -add_delay 0.3 [get_ports q]
# set_input_delay -clock [get_clocks clk] -add_delay 0.3 [get_ports rst_n]

# set_output_delay -clock [get_clocks clk] -add_delay 0.3 [get_ports dispense]
# set_output_delay -clock [get_clocks clk] -add_delay 0.3 [get_ports return_d]
# set_output_delay -clock [get_clocks clk] -add_delay 0.3 [get_ports return_n]
# set_output_delay -clock [get_clocks clk] -add_delay 0.3 [get_ports return_two_d]

set_max_fanout 15.000 [current_design]
set_max_transition 1.2 [current_design]
