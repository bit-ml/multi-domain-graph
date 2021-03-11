import configparser
import os
import sys
from datetime import datetime

os.system("mkdir -p generated_configs/")

domain_id = sys.argv[1]
cfg_template = sys.argv[2]
# replica_template_iter1.ini
# replica_template_train_iter2.ini
# hypersim_template_train_iter1.ini

# intro
cfg_out = "generated_configs/launch_%s_%s.ini" % (domain_id, str(
    datetime.now()))
config = configparser.ConfigParser()
config.read(cfg_template)
'''
if cfg_template == 'hypersim_template_iter1.ini':
    if domain_id == 'depth_n_1_xtc':
        config.set('Experts', 'selector_map',
                   '1, 4, 5, 6, 7, 8, 9, 10, 11, 12')
    if domain_id == 'normals_xtc':
        config.set('Experts', 'selector_map', '2, 8, 9, 10, 11, 12')
    if domain_id == 'sobel_small':
        config.set('Experts', 'selector_map', '6, 7, 8, 9, 10, 11, 12')
    if domain_id == 'edges_dexined':
        config.set('Experts', 'selector_map',
                   '3, 4, 5, 6, 7, 8, 9, 10, 11, 12')
    if domain_id == 'halftone_gray':
        config.set('Experts', 'selector_map', '4, 8, 9, 10, 11, 12')
    if domain_id == 'grayscale':
        config.set('Experts', 'selector_map',
                   '3, 4, 5, 6, 7, 8, 9, 10, 11, 12')
    if domain_id == 'rgb':
        config.set('Experts', 'selector_map',
                   '0, 4, 5, 6, 7, 8, 9, 10, 11, 12')
'''
config.set("GraphStructure", "restricted_graph_exp_identifier", domain_id)

# SET model type
if domain_id in [
        "edges_dexined", "sobel_large", "sobel_small", "sobel_medium", "hsv"
]:
    config.set("Edge Models", "regression_losses", "l2")
    config.set("Edge Models", "regression_losses_weights", "1")

config.set("Edge Models", "model_type", "1")
tensorboard_prefix = config.get("Logs", "tensorboard_prefix")
config.set("Logs", "tensorboard_prefix",
           "%s_%s" % (tensorboard_prefix, domain_id))

with open(cfg_out, "w") as fd:
    config.write(fd)

os.system('python main.py "%s"' % (cfg_out))
