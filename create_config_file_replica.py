import configparser
import os
from datetime import datetime

os.system("mkdir -p generated_configs/")

cfg_template = "replica_template_eval.ini"

# sorted pt replica unet medium
sorted_by_score = [2, 11, 10, 7, 6, 0, 8, 12, 9, 3, 1, 5, 4]

top5 = str(sorted_by_score[:6])[1:-1]
top10 = str(sorted_by_score[:11])[1:-1]
all12 = str(sorted_by_score)[1:-1]

top3 = str(sorted_by_score[:4])[1:-1]
top7 = str(sorted_by_score[:8])[1:-1]
# replica: 300, 260, 200, 160
for start_epoch in [100]:
    for selector_map in [all12]:
        # "ssim", "lpips", "l1", "l2", "mssim", "equal", "equal_mean"
        for ens_fcn in [
                # ("ssim"),
            ("lpips"),
                # ("l1, lpips"),
                # ("l1, ssim"),
                # ("lpips, ssim"),
                # ("l1, lpips, ssim"),
                # ("l2"),
                # ("mssim"),
                # "equal_mean",
        ]:
            # intro
            cfg_out = "generated_configs/eval_normals_%s.ini" % str(
                datetime.now())
            config = configparser.ConfigParser()
            config.read(cfg_template)

            config.set("General", "DATASET_NAME", "replica")

            # SET models
            config.set("Edge Models", "model_type", "1")
            config.set("Edge Models", "start_epoch", "%d" % start_epoch)
            config.set(
                "Edge Models", "load_path",
                '/data/multi-domain-graph/models/replica_iter1_unetmedium_ssim'
            )

            # SET metric
            config.set("Ensemble", "similarity_fct", ens_fcn)
            config.set("Ensemble", "thresholds", "0")

            # SET domains
            config['Experts'] = {'selector_map': selector_map}

            # Logs
            config.set("Logs", "tensorboard_prefix", "testare_%s" % ens_fcn)

            with open(cfg_out, "w") as fd:
                config.write(fd)

            os.system('python main.py "%s"' % (cfg_out))
        # break
