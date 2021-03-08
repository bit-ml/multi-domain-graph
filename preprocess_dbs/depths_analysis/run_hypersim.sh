nohup python depth_analysis.py xtc hypersim 0 0 all > ./logs_replica/logs_hypersim_xtc_all.out 
wait
nohup python depth_analysis.py sgdepth hypersim 0 0 all > ./logs_replica/logs_hypersim_sgdepth_all.out 
