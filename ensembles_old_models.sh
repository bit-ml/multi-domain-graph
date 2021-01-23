nohup python main.py ./configs/config_ema_ssim_twd_exp_mean.ini > logs_ensemble_ssim_twd_exp_mean.out
wait
nohup python main.py ./configs/config_ema_ssim_twd_exp_mean_with_rgb.ini > logs_ensemble_ssim_twd_exp_mean_with_rgb.out
wait 
nohup python main.py ./configs/config_ema_ssim_twd_exp_median_faster.ini > logs_ensemble_ssim_twd_exp_median_faster.out
wait 
nohup python main.py ./configs/config_ema_ssim_twd_exp_median_faster_with_rgb.ini > logs_ensemble_ssim_twd_exp_median_faster_with_rgb.out
wait
nohup python main.py ./configs/config_ema_ssim_twd_exp_median.ini > logs_ensemble_ssim_twd_exp_median.out
wait
nohup python main.py ./configs/config_ema_ssim_twd_exp_median_with_rgb.ini > logs_ensemble_ssim_twd_exp_median_with_rgb.out
wait
