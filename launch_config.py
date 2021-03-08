import configparser
import os
import sys
from datetime import datetime

os.system("mkdir -p generated_configs/")

domain_id = sys.argv[1]
cfg_template = sys.argv[2]
# replica_template_iter1.ini
# hypersim_template_train_iter1.ini
# cfg_template = "replica_template_iter1.ini"

# intro
cfg_out = "generated_configs/launch_%s_%s.ini" % (domain_id, str(
    datetime.now()))
config = configparser.ConfigParser()
config.read(cfg_template)

config.set("GraphStructure", "restricted_graph_exp_identifier", domain_id)

# SET model type
if domain_id in [
        "edges_dexined", "sobel_large", "sobel_small", "sobel_medium", "hsv"
]:
    config.set("Edge Models", "regression_losses", "l2")
    config.set("Edge Models", "regression_losses_weights", "1")

config.set("Edge Models", "model_type", "1")
config.set("Logs", "tensorboard_prefix", "rep_it1_%s" % domain_id)

with open(cfg_out, "w") as fd:
    config.write(fd)

os.system('python main.py "%s"' % (cfg_out))
