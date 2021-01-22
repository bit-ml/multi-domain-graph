nohup python main.py ./configs/config_ema_mean.ini > logs_ensemble_mean.out
wait
nohup python main.py ./configs/config_ema_median.ini > logs_ensemble_median.out
wait
nohup python main.py ./configs/config_ema_median10.ini > logs_ensemble_median10.out
wait
nohup python main.py ./configs/config_ema_histo.ini > logs_ensemble_histo.out
wait
nohup python main.py ./configs/config_ema_histo_median.ini > logs_ensemble_histo_median.out
wait
nohup python main.py ./configs/config_ema_ssim_twd_exp_mean.ini > logs_ensemble_ssim_twd_exp_mean.out
wait
nohup python main.py ./configs/config_ema_ssim_twd_exp_median.ini > logs_ensemble_ssim_twd_exp_median.out
wait
nohup python main.py ./configs/config_ema_ssim_twd_exp_median.ini > logs_ensemble_ssim_twd_exp_median_faster.out
wait
nohup python main.py ./configs/config_ema_ssim_twd_exp_median_w.ini > logs_ensemble_ssim_twd_exp_median_w.out
wait
nohup python main.py ./configs/config_ema_ssim_btw_tasks_mean.ini > logs_ensemble_ssim_btw_tasks_mean.out
wait
nohup python main.py ./configs/config_ema_ssim_btw_tasks_median.ini > logs_ensemble_ssim_btw_tasks_median.out
wait
nohup python main.py ./configs/config_ema_ssim_btw_tasks_median.ini > logs_ensemble_ssim_btw_tasks_median_faster.out
wait
nohup python main.py ./configs/config_ema_ssim_btw_tasks_median_w.ini > logs_ensemble_ssim_btw_tasks_median_w.out
