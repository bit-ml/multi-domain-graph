#!/bin/bash
gt_vs_exp=$1
exp_name=$2
CUDA_VISIBLE_DEVICES=0 python main_taskonomy.py  ${gt_vs_exp}  tiny-test ${exp_name} > logs/tasko_test.log & 
CUDA_VISIBLE_DEVICES=1 python main_taskonomy.py  ${gt_vs_exp}  tiny-val ${exp_name} > logs/tasko_val.log & 
CUDA_VISIBLE_DEVICES=2 python main_taskonomy.py  ${gt_vs_exp}  tiny-train-0.15-part1 ${exp_name} > logs/tasko_train_part1.log & 
CUDA_VISIBLE_DEVICES=3 python main_taskonomy.py  ${gt_vs_exp}  tiny-train-0.15-part2 ${exp_name} > logs/tasko_train_part2.log 
CUDA_VISIBLE_DEVICES=3 python main_taskonomy.py  ${gt_vs_exp}  tiny-train-0.15-part3 ${exp_name} > logs/tasko_train_part3.log 