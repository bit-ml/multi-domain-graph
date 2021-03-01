import configparser
import os
from datetime import datetime

os.system("mkdir -p generated_configs/")

cfg_template = "replica_template_eval.ini"

# sorted pt replica unet medium
sorted_by_score = [2, 0, 6, 10, 11, 1, 3, 9, 8, 7, 5, 4, 12]

top5 = str(sorted_by_score[:6])[1:-1]
top10 = str(sorted_by_score[:11])[1:-1]
all12 = str(sorted_by_score)[1:-1]

top3 = str(sorted_by_score[:4])[1:-1]
top7 = str(sorted_by_score[:8])[1:-1]
# replica: 300, 260, 200, 160
for start_epoch in [200]:
    # all12, top10, top7, top5, top3
    for selector_map in [all12]:
        #"lpips", "mssim", "ssim", "equal"
        for ens_fcn in ["equal"]:
            # intro
            cfg_out = "generated_configs/eval_normals_%s.ini" % str(
                datetime.now())
            config = configparser.ConfigParser()
            config.read(cfg_template)

            config.set("General", "DATASET_NAME", "replica")

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
