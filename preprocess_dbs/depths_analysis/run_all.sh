nohup python depth_analysis.py xtc replica2 0 0 all > ./logs_replica2/logs_replica2_xtc_all.out 
wait
nohup python depth_analysis.py sgdepth replica 0 0 all > ./logs_replica/logs_replica_sgdepth_all.out 
wait
nohup python depth_analysis.py sgdepth replica2 0 0 all > ./logs_replica2/logs_replica2_sgdepth_all.out 
wait
nohup python depth_analysis.py xtc hypersim 0 0 all > ./logs_hypersim/logs_hypersim_xtc_all.out 
wait
nohup python depth_analysis.py sgdepth hypersim 0 0 all > ./logs_hypersim/logs_hypersim_sgdepth_all.out 
wait
nohup python depth_analysis.py xtc taskonomy 0 0 all > ./logs_taskonomy/logs_taskonomy_xtc_all.out 
wait
nohup python depth_analysis.py sgdepth taskonomy 0 0 all > ./logs_taskonomy/logs_taskonomy_sgdepth_all.out 

