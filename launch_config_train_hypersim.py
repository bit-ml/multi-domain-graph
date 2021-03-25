import configparser
import os
import sys
import numpy as np
from datetime import datetime

os.system("mkdir -p generated_configs/")

domain_id = sys.argv[1]
cfg_template = sys.argv[2]
# replica_template_iter1.ini
# replica_template_train_iter2.ini
# hypersim_template_train_iter1.ini
# hypersim_v2_template_iter1.ini
# 0. RGBModel(full_experts),
# 1. DepthModelXTC(full_experts),
# 2. SurfaceNormalsXTC(full_experts),
# 3. EdgesModel(full_experts),
# 4. HalftoneModel(full_experts, 0),
# 5. SSegHRNet
# 6. Grayscale(full_experts),
# 7. HSVExpert(full_experts),
# 8. CartoonWB(full_experts),
# 9. SobelEdgesExpertSigmaLarge(full_experts),
# 10. SobelEdgesExpertSigmaMedium(full_experts),
# 11. SobelEdgesExpertSigmaSmall(full_experts),
# 12. SuperPixel(full_experts),
#selector_map = 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12
#selector_map = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12

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

tensorboard_prefix = config.get("Logs", "tensorboard_prefix")
config.set("Logs", "tensorboard_prefix",
           "%s_%s" % (tensorboard_prefix, domain_id))

with open(cfg_out, "w") as fd:
    config.write(fd)

os.system('python main.py "%s"' % (cfg_out))
