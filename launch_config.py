import configparser
import os
import sys
from datetime import datetime

os.system("mkdir -p generated_configs/")

domain_id = sys.argv[1]
cfg_template = sys.argv[2]

# intro
cfg_out = "generated_configs/launch_%s_%s.ini" % (domain_id, str(
    datetime.now()))
config = configparser.ConfigParser()
config.read(cfg_template)

config.set("GraphStructure", "restricted_graph_exp_identifier", domain_id)

# SET model type
if domain_id in ["rgb", "normals_xtc", "sem_seg_hrnet", "cartoon_wb"]:
    config.set("Edge Models", "model_type", "1")
else:
    config.set("Edge Models", "model_type", "0")

with open(cfg_out, "w") as fd:
    config.write(fd)

os.system('python main.py "%s"' % (cfg_out))
