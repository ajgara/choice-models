#!/bin/bash

python estimate.py em lc examples/instances/30_periods_1_instance.json em_lc_estimation.json
python estimate.py em mkv examples/instances/30_periods_1_instance.json em_mkv_estimation.json
python estimate.py em rl examples/instances/30_periods_1_instance.json em_rl_estimation.json
python estimate.py fw lc examples/instances/30_periods_1_instance.json fw_lc_estimation.json
python estimate.py max exp examples/instances/30_periods_1_instance.json max_exp_estimation.json
python estimate.py max lc examples/instances/30_periods_1_instance.json max_lc_estimation.json
python estimate.py max mkv examples/instances/30_periods_1_instance.json max_mkv_estimation.json
python estimate.py max mkv2 examples/instances/30_periods_1_instance.json max_mkv2_estimation.json
python estimate.py max mnl examples/instances/30_periods_1_instance.json max_mnl_estimation.json
python estimate.py max mx examples/instances/30_periods_1_instance.json max_mx_estimation.json
python estimate.py max nl examples/instances/30_periods_1_instance.json max_nl_estimation.json
python estimate.py max rl examples/instances/30_periods_1_instance.json max_rl_estimation.json
python estimate.py max rnd examples/instances/30_periods_1_instance.json max_rnd_estimation.json
