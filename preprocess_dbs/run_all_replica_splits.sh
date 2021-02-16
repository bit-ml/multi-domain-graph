#!/bin/bash
exp_name=$1
python main_replica.py 1 test ${exp_name}
python main_replica.py 1 val ${exp_name}
python main_replica.py 1 train ${exp_name}