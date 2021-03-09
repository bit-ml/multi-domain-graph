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
type_of_run = np.int32(sys.argv[3])
# 0 - simple mean & simple median
# 1 - simple variance - no other metrics involved
# 2 - test metrics - without variance
# 3 - test metrics + variance logic
# 4 - test metrics with variance as dist
# 10 - all

# intro
cfg_out = "generated_configs/launch_ensembles_%s_%s_%s.ini" % (
    domain_id, str(datetime.now()), cfg_template[:-4])
config = configparser.ConfigParser()
config.read(cfg_template)

config.set("GraphStructure", "restricted_graph_type", '2')
config.set("GraphStructure", "restricted_graph_exp_identifier", domain_id)
tensorboard_prefix = config.get("Logs", "tensorboard_prefix")

if type_of_run == 0 or type_of_run == 10:
    ## test simple mean
    config.set('Ensemble', 'enable_simple_mean', 'yes')
    tensorboard_prefix = config.get("Logs", "tensorboard_prefix")
    config.set("Logs", "tensorboard_prefix",
               "%s_%s_simple_mean" % (tensorboard_prefix, domain_id))

    with open(cfg_out, "w") as fd:
        config.write(fd)

    os.system('python main.py "%s"' % (cfg_out))

    ## test simple median
    config.set('Ensemble', 'enable_simple_mean', 'no')
    config.set('Ensemble', 'enable_simple_median', 'yes')
    config.set("Logs", "tensorboard_prefix",
               "%s_%s_simple_median" % (tensorboard_prefix, domain_id))

    with open(cfg_out, "w") as fd:
        config.write(fd)

    os.system('python main.py "%s"' % (cfg_out))

if type_of_run == 1 or type_of_run == 10:
    ## test simple variance filter
    config.set('Ensemble', 'enable_simple_mean', 'no')
    config.set('Ensemble', 'enable_simple_median', 'no')
    config.set('Ensemble', 'similarity_fct', 'dist_mean')
    config.set('Ensemble', 'fix_variance', 'yes')

    var_dismiss_ths = np.array([0.11, 0.1, 0.09, 0.08, 0.07, 0.06])
    for th_variance in var_dismiss_ths:
        config.set('Ensemble', 'variance_dismiss_threshold', str(th_variance))
        for kernel_f in ['flat', 'flat_weighted', 'gauss']:
            config.set('Ensemble', 'kernel_fct', kernel_f)
            if kernel_f == 'flat' or kernel_f == 'flat_weighted':
                ths = np.array([0.25, 0.5, 0.75, 1])
            else:
                ths = np.array([0.25, 0.5, 1])
            for th in ths:
                config.set('Ensemble', 'meanshiftiter_thresholds', str(th))
                for comb_type in ['mean', 'median']:
                    config.set('Ensemble', 'comb_type', comb_type)
                    config.set(
                        "Logs", "tensorboard_prefix",
                        "%s_%s_simple_variance__%s__%s__%s__%s" %
                        (tensorboard_prefix, domain_id, kernel_f, comb_type,
                         str(th), comb_type))

                    with open(cfg_out, "w") as fd:
                        config.write(fd)

                    os.system('python main.py "%s"' % (cfg_out))

if type_of_run == 2 or type_of_run == 10:
    ## test metrics without variance
    config.set('Ensemble', 'enable_simple_mean', 'no')
    config.set('Ensemble', 'enable_simple_median', 'no')
    config.set('Ensemble', 'fix_variance', 'no')

    for sim_f in ['l1', 'l2', 'ssim', 'lpips', 'dist_mean']:
        config.set('Ensemble', 'similarity_fct', sim_f)
        for kernel_f in ['flat', 'flat_weighted', 'gauss']:
            config.set('Ensemble', 'kernel_fct', kernel_f)
            if kernel_f == 'flat' or kernel_f == 'flat_weighted':
                ths = np.array([0.25, 0.5, 0.75, 1])
            else:
                ths = np.array([0.25, 0.5, 1])
            for th in ths:
                config.set('Ensemble', 'meanshiftiter_thresholds', str(th))
                for comb_type in ['mean', 'median']:
                    config.set('Ensemble', 'comb_type', comb_type)
                    config.set(
                        "Logs", "tensorboard_prefix", "%s_%s__%s__%s__%s__%s" %
                        (tensorboard_prefix, domain_id, sim_f, kernel_f,
                         comb_type, str(th)))
                    with open(cfg_out, "w") as fd:
                        config.write(fd)

                    os.system('python main.py "%s"' % (cfg_out))

if type_of_run == 3 or type_of_run == 10:
    ## test metrics with variance
    config.set('Ensemble', 'enable_simple_mean', 'no')
    config.set('Ensemble', 'enable_simple_median', 'no')
    config.set('Ensemble', 'fix_variance', 'yes')

    var_dismiss_ths = np.array([0.11, 0.1, 0.09, 0.08, 0.07, 0.06])
    for th_variance in var_dismiss_ths:
        config.set('Ensemble', 'variance_dismiss_threshold', str(th_variance))
        for sim_f in ['l1', 'l2', 'ssim', 'lpips', 'dist_mean']:
            config.set('Ensemble', 'similarity_fct', sim_f)
            for kernel_f in ['flat', 'flat_weighted', 'gauss']:
                config.set('Ensemble', 'kernel_fct', kernel_f)
                if kernel_f == 'flat' or kernel_f == 'flat_weighted':
                    ths = np.array([0.25, 0.5, 0.75, 1])
                else:
                    ths = np.array([0.25, 0.5, 1])
                for th in ths:
                    config.set('Ensemble', 'meanshiftiter_thresholds', str(th))
                    for comb_type in ['mean', 'median']:
                        config.set('Ensemble', 'comb_type', comb_type)
                        config.set(
                            "Logs", "tensorboard_prefix",
                            "%s_%s__var__%s__%s__%s__%s__%04.2f" %
                            (tensorboard_prefix, domain_id, sim_f, kernel_f,
                             comb_type, str(th), th_variance))
                        print(config.get("Logs", "tensorboard_prefix"))
                        with open(cfg_out, "w") as fd:
                            config.write(fd)

                        os.system('python main.py "%s"' % (cfg_out))

if type_of_run == 4 or type_of_run == 10:
    ## test metrics with variance integrated as dist_fct
    config.set('Ensemble', 'enable_simple_mean', 'no')
    config.set('Ensemble', 'enable_simple_median', 'no')
    config.set('Ensemble', 'fix_variance', 'yes')

    var_dismiss_ths = np.array([0.11, 0.1, 0.09, 0.08, 0.07, 0.06])
    for th_variance in var_dismiss_ths:
        config.set('Ensemble', 'variance_dismiss_threshold', str(th_variance))
        for sim_f in ['l1', 'l2', 'ssim', 'lpips']:
            config.set('Ensemble', 'similarity_fct', sim_f + ', dist_mean')
            for kernel_f in ['flat', 'flat_weighted', 'gauss']:
                config.set('Ensemble', 'kernel_fct', kernel_f)
                if kernel_f == 'flat' or kernel_f == 'flat_weighted':
                    ths = np.array([0.25, 0.5, 0.75, 1])
                else:
                    ths = np.array([0.25, 0.5, 1])
                for th in ths:
                    config.set('Ensemble', 'meanshiftiter_thresholds', str(th))
                    for comb_type in ['mean', 'median']:
                        config.set('Ensemble', 'comb_type', comb_type)
                        config.set(
                            "Logs", "tensorboard_prefix",
                            "%s_%s__varmetric__%s__%s__%s__%s" %
                            (tensorboard_prefix, domain_id, sim_f, kernel_f,
                             comb_type, str(th)))
                        with open(cfg_out, "w") as fd:
                            config.write(fd)

                        os.system('python main.py "%s"' % (cfg_out))
