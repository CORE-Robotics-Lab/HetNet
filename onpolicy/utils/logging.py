from collections import defaultdict
import json
import logging
import numpy as np
# import torch as th
import os
class Logger:
    def __init__(self, log_dir):
        # self.console_logger = console_logger
        self.log_dir = log_dir
        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False
        self.sacred_info = {}

    def log_stat(self, key, value, t, time):
        '''logs values over sample step T'''

        if key in self.sacred_info:
            self.sacred_info["{}_T".format(key)].append(t)
            self.sacred_info["{}_time".format(key)].append(time)
            self.sacred_info[key].append(value)
        else:
            self.sacred_info["{}_T".format(key)] = [t]
            self.sacred_info["{}_time".format(key)] = [time]
            self.sacred_info[key] = [value]

    def save_config(self, config):
        path = os.path.join(self.log_dir, 'config.json')
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)
    
    def save_stats(self):
        path = os.path.join(self.log_dir, 'run_stats.json')
        with open(path, 'w') as f:
            json.dump(self.sacred_info, f, indent=4)

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(*self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            # item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
            item = "{:.4f}".format(th.mean(th.tensor([x[1] for x in self.stats[k][-window:]])))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)


# set up a custom logger
def get_logger():
    logger = logging.getLogger()
    # logger = logging.getLogger("imported_module").setLevel(logging.WARNING)
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # logger.setLevel('DEBUG')
    logger.setLevel(logging.INFO)
    # logging.getLogger("imported_module").setLevel(logging.WARNING)
    return logger

