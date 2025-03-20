design_name = traffic_light_controller
rtl_sim: clean_sim
	cd ./sim/work && vsim -c -do ../script/run.do
rtl_sim_wave:
	cd ./sim/work && vsim $(design_name).wlf -do ../script/add_wave.do
post_syn_sim: clean_sim
	cd ./sim/work && vsim -c -do ../script/run_post_syn.do
post_syn_sim_wave:
	cd ./sim/work && vsim $(design_name)_net.wlf -do ../script/add_wave.do
post_syn_sim: clean_sim
	cd ./sim/work && vsim -c -do ../script/run_post_syn.do
run_syn: clean_syn
	cd ./syn/work && genus -f ../script/syn.tcl
clean_sim:
	cd ./sim/work && rm -rf *
clean_syn:
	rm -rf ./syn/outputs/* ./syn/reports/* ./syn/work/*
clean: clean_sim clean_syn
.PHONY: clean_sim clean_syn 