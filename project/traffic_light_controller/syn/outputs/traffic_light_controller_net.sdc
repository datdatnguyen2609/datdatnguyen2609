# ####################################################################

#  Created by Genus(TM) Synthesis Solution 19.10-p002_1 on Sat Mar 15 04:47:47 EDT 2025

# ####################################################################

set sdc_version 2.0

set_units -capacitance 1fF
set_units -time 1000ps

# Set the current design
current_design traffic_light_controller

create_clock -name "clk" -period 10.0 -waveform {0.0 5.0} [get_ports clk]
set_clock_gating_check -setup 0.0 
set_max_fanout 15.000 [current_design]
set_max_transition 1.2 [current_design]
set_dont_use true [get_lib_cells NangateOpenCellLibrary/SDFF_X1]
set_dont_use true [get_lib_cells NangateOpenCellLibrary/SDFF_X2]
