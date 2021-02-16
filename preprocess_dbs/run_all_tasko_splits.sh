#!/bin/bash
exp_name=$1
python main_taskonomy.py 1 tiny-test ${exp_name}
python main_taskonomy.py 1 tiny-val ${exp_name}
python main_taskonomy.py 1 tiny-train-0.15-part1 ${exp_name}
python main_taskonomy.py 1 tiny-train-0.15-part2 ${exp_name}
python main_taskonomy.py 1 tiny-train-0.15-part3 ${exp_name}