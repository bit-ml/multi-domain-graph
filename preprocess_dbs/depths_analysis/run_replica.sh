nohup python depth_analysis.py xtc replica 0 0 all > ./logs_replica/logs_replica_xtc_all.out 
wait
nohup python depth_analysis.py xtc replica2 0 0 all > ./logs_replica/logs_replica2_xtc_all.out 
wait
nohup python depth_analysis.py sgdepth replica 0 0 all > ./logs_replica/logs_replica_sgdepth_all.out 
wait
nohup python depth_analysis.py sgdepth replica2 0 0 all > ./logs_replica/logs_replica2_sgdepth_all.out 
