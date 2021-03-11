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
# replica_template_save_iter1.ini

# intro
cfg_out = "generated_configs/launch_save_%s_%s_%s.ini" % (
    domain_id, str(datetime.now()), cfg_template[:-4])
config = configparser.ConfigParser()
config.read(cfg_template)

config.set("GraphStructure", "restricted_graph_type", '2')
config.set("GraphStructure", "restricted_graph_exp_identifier", domain_id)
tensorboard_prefix = config.get("Logs", "tensorboard_prefix")
config.set("Logs", "tensorboard_prefix",
           "%s_%s" % (tensorboard_prefix, domain_id))
print(config.get("Logs", "tensorboard_prefix"))
with open(cfg_out, "w") as fd:
    config.write(fd)

os.system('python main.py "%s"' % (cfg_out))
