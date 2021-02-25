import configparser
import os
from datetime import datetime

cfg_template = "template_eval.ini"

top5 = "2, 0, 1, 3, 4, 6, 9, 10, 11, 12"
top10 = "2, 0, 1, 3, 4, 6, 8, 9, 10, 11, 12"
all12 = "2, 0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12"

for start_epoch in [300, 260, 299, 160]:
    for selector_map in [all12]:  #, all12, top10, top5]:
        for ens_fcn in ["lpips"]:  #, "lpips", "mssim", "ssim", "equal"]:
            # intro
            cfg_out = "replica_gen_configs/replica_eval_normals_%s.ini" % str(
                datetime.now())
            config = configparser.ConfigParser()
            config.read(cfg_template)

            # SET models
            config.set("Edge Models", "start_epoch", "%d" % start_epoch)
            config.set(
                "Edge Models", "load_path",
                '/data/multi-domain-graph/models/replica_iter1_300epochs_2chan_unetmedium'
            )

            # SET metric
            config.set("Ensemble", "similarity_fct", ens_fcn)

            # SET domains
            config['Experts'] = {'selector_map': selector_map}

            with open(cfg_out, "w") as fd:
                config.write(fd)

            os.system('python main.py "%s"' % (cfg_out))
        # break
