#!/bin/bash
gt_vs_exp=$1
exp_name=$2
CUDA_VISIBLE_DEVICES=0 python main_replica.py ${gt_vs_exp} test ${exp_name} > logs/replica_test.log &
CUDA_VISIBLE_DEVICES=1 python main_replica.py ${gt_vs_exp}  val ${exp_name} > logs/replica_val.log & 
CUDA_VISIBLE_DEVICES=2 python main_replica.py ${gt_vs_exp}  train ${exp_name} > logs/replica_train.log & 
