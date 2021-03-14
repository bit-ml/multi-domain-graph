import configparser
import os
import sys
from datetime import datetime

import numpy as np

os.system("mkdir -p generated_configs/")

# dst domain
domain_id = sys.argv[1]
# base template
cfg_template = sys.argv[2]
# replica_template_eval_iter1.ini

# intro
cfg_out = "generated_configs/launch_ensembles_%s_%s_%s.ini" % (
    domain_id, str(datetime.now()), cfg_template[:-4])
config = configparser.ConfigParser()
config.read(cfg_template)

config.set("GraphStructure", "restricted_graph_type", '2')
config.set("GraphStructure", "restricted_graph_exp_identifier", domain_id)
orig_tensorboard_prefix = config.get("Logs", "tensorboard_prefix")

config.set('Ensemble', 'eval_top_edges_nr',
           '11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1')

random_edges = ['no', 'yes']
for random_edges_ in random_edges:

    config.set('Ensemble', 'random_edges', random_edges_)

    config.set('Ensemble', 'enable_simple_mean', 'yes')
    config.set(
        "Logs", "tensorboard_prefix", "%s_%s_simple_mean_random_%s" %
        (orig_tensorboard_prefix, domain_id, random_edges))
    with open(cfg_out, "w") as fd:
        config.write(fd)

    os.system('python main.py "%s"' % (cfg_out))

    config.set('Ensemble', 'enable_simple_mean', 'no')
    config.set('Ensemble', 'enable_simple_median', 'no')
    config.set('Ensemble', 'similarity_fct', 'lpips')
    config.set('Ensemble', 'fix_variance', 'no')
    config.set('Ensemble', 'kernel_fct', 'flat_weighted')
    th = 1
    config.set('Ensemble', 'meanshiftiter_thresholds', str(th))
    for comb_type in ['mean', 'median']:
        config.set('Ensemble', 'comb_type', comb_type)
        config.set(
            "Logs", "tensorboard_prefix",
            "%s_%s_%s__%s__%s" % (orig_tensorboard_prefix, domain_id,
                                  random_edges_, comb_type, str(th)))
        with open(cfg_out, "w") as fd:
            config.write(fd)

        os.system('python main.py "%s"' % (cfg_out))
