import configparser
import os
import sys
import numpy as np
from datetime import datetime

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

# intro
cfg_out = "generated_configs/launch_ensembles_%s_%s_%s.ini" % (
    domain_id, str(datetime.now()), cfg_template[:-4])
config = configparser.ConfigParser()
config.read(cfg_template)

config.set("GraphStructure", "restricted_graph_type", '2')
config.set("GraphStructure", "restricted_graph_exp_identifier", domain_id)

if type_of_run == 0:
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
    tensorboard_prefix = config.get("Logs", "tensorboard_prefix")
    config.set("Logs", "tensorboard_prefix",
               "%s_%s_simple_median" % (tensorboard_prefix, domain_id))

    with open(cfg_out, "w") as fd:
        config.write(fd)

    os.system('python main.py "%s"' % (cfg_out))

if type_of_run == 1:
    ## test simple variance filter
    config.set('Ensemble', 'enable_simple_mean', 'no')
    config.set('Ensemble', 'enable_simple_median', 'no')
    config.set('Ensemble', 'similarity_fct', 'dist_mean')
    config.set('Ensemble', 'fix_variance', 'yes')
    config.set('Ensemble', 'kernel_fct', 'flat')
    config.set('Ensemble', 'meanshiftiter_thresholds', '1')
    config.set('Ensemble', 'comb_type', 'mean')
    tensorboard_prefix = config.get("Logs", "tensorboard_prefix")
    config.set(
        "Logs", "tensorboard_prefix",
        "%s_%s_simple_variance_comb_mean" % (tensorboard_prefix, domain_id))
    with open(cfg_out, "w") as fd:
        config.write(fd)

    os.system('python main.py "%s"' % (cfg_out))

    config.set('Ensemble', 'comb_type', 'median')
    tensorboard_prefix = config.get("Logs", "tensorboard_prefix")
    config.set(
        "Logs", "tensorboard_prefix",
        "%s_%s_simple_variance_comb_median" % (tensorboard_prefix, domain_id))
    with open(cfg_out, "w") as fd:
        config.write(fd)

    os.system('python main.py "%s"' % (cfg_out))

if type_of_run == 2:
    ## test metrics without variance
    config.set('Ensemble', 'enable_simple_mean', 'no')
    config.set('Ensemble', 'enable_simple_median', 'no')
    config.set('Ensemble', 'fix_variance', 'no')

    for sim_f in ['l1', 'l2', 'ssim', 'lpips', 'dist_mean']:
        config.set('Ensemble', 'similarity_fct', sim_f)
        for kernel_f in ['flat', 'flat_weighted', 'gauss']:
            config.set('Ensemble', 'kernel_fct', kernel_f)
            if kernel_f == 'flat' or kernel_f == 'flat_weighted':
                ths = np.array([0, 0.25, 0.5, 0.75, 1])
            else:
                ths = np.array([0.25, 0.5, 1])
            for th in ths:
                config.set('Ensemble', 'meanshiftiter_thresholds', str(th))
                for comb_type in ['mean', 'median']:
                    config.set('Ensemble', 'comb_type', comb_type)

                    tensorboard_prefix = config.get("Logs",
                                                    "tensorboard_prefix")
                    config.set(
                        "Logs", "tensorboard_prefix",
                        "%s_%s__%s__%s__%s" % (tensorboard_prefix, domain_id,
                                               sim_f, kernel_f, str(th)))
                    with open(cfg_out, "w") as fd:
                        config.write(fd)

                    os.system('python main.py "%s"' % (cfg_out))

if type_of_run == 3:
    ## test metrics with variance
    config.set('Ensemble', 'enable_simple_mean', 'no')
    config.set('Ensemble', 'enable_simple_median', 'no')
    config.set('Ensemble', 'fix_variance', 'yes')

    var_dismiss_ths = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    for th_variance in var_dismiss_ths:
        config.set('Ensemble', 'variance_dismiss_thresholds', str(th_variance))
        for sim_f in ['l1', 'l2', 'ssim', 'lpips', 'dist_mean']:
            config.set('Ensemble', 'similarity_fct', sim_f)
            for kernel_f in ['flat', 'flat_weighted', 'gauss']:
                config.set('Ensemble', 'kernel_fct', kernel_f)
                if kernel_f == 'flat' or kernel_f == 'flat_weighted':
                    ths = np.array([0, 0.25, 0.5, 0.75, 1])
                else:
                    ths = np.array([0.25, 0.5, 1])
                for th in ths:
                    config.set('Ensemble', 'meanshiftiter_thresholds', str(th))
                    for comb_type in ['mean', 'median']:
                        config.set('Ensemble', 'comb_type', comb_type)

                        tensorboard_prefix = config.get(
                            "Logs", "tensorboard_prefix")
                        config.set(
                            "Logs", "tensorboard_prefix",
                            "%s_%s__%s__%s__%s__var_%04.2f" %
                            (tensorboard_prefix, domain_id, sim_f, kernel_f,
                             str(th), str(th_variance)))
                        with open(cfg_out, "w") as fd:
                            config.write(fd)

                        os.system('python main.py "%s"' % (cfg_out))

if type_of_run == 4:
    ## test metrics with variance integrated as dist_fct
    config.set('Ensemble', 'enable_simple_mean', 'no')
    config.set('Ensemble', 'enable_simple_median', 'no')
    config.set('Ensemble', 'fix_variance', 'yes')

    config.set('Ensemble', 'variance_dismiss_thresholds', str(th_variance))
    for sim_f in ['l1', 'l2', 'ssim', 'lpips']:
        config.set('Ensemble', 'similarity_fct', sim_f + ', dist_mean')
        for kernel_f in ['flat', 'flat_weighted', 'gauss']:
            config.set('Ensemble', 'kernel_fct', kernel_f)
            if kernel_f == 'flat' or kernel_f == 'flat_weighted':
                ths = np.array([0, 0.25, 0.5, 0.75, 1])
            else:
                ths = np.array([0.25, 0.5, 1])
            for th in ths:
                config.set('Ensemble', 'meanshiftiter_thresholds', str(th))
                for comb_type in ['mean', 'median']:
                    config.set('Ensemble', 'comb_type', comb_type)

                    tensorboard_prefix = config.get("Logs",
                                                    "tensorboard_prefix")
                    config.set(
                        "Logs", "tensorboard_prefix",
                        "%s_%s__%s__%s__%s__varmetric" %
                        (tensorboard_prefix, domain_id, sim_f, kernel_f,
                         str(th)))
                    with open(cfg_out, "w") as fd:
                        config.write(fd)

                    os.system('python main.py "%s"' % (cfg_out))