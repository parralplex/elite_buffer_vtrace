import os

import torch.multiprocessing as mp
import numpy as np
import random
import datetime as dt
import torch.backends.cudnn

from option_flags import flags, change_args
from agent.learner_d.learner import Learner
from utils.logger import create_logger, logger, change_logger_file_handler


os.environ["OMP_NUM_THREADS"] = "1"
if flags.debug:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def get_dict(**kwargs):
    return kwargs


if __name__ == '__main__':
    mp.set_start_method('spawn')

    for i in range(1):
        batch_size = 10
        r_f_steps = 50
        replay_out_cache_size = 1
        backend = "python_native"
        reproducible = False

        feature_out_layer_size = 512
        lr_scheduler_steps = 70000
        p = 2
        for j in range(1):
            run_id = int(dt.datetime.timestamp(dt.datetime.now()))
            save_url = "results/" + flags.env + "_" + str(run_id)
            os.makedirs(save_url)
            if j == 0 and i == 0:
                create_logger(save_url)
            else:
                change_logger_file_handler(save_url)
            logger.info("Starting execution " + str(run_id) + " with order number " + str(j))

            mini_batch_multiplier = 9

            replay_parameters = '[{"type": "queue", "capacity": 1, "sample_ratio": 0.5}, {"type": "custom", "capacity": 10000, "sample_ratio": 0.5, "dist_function":"ln_norm", "sample_strategy":"elite_sampling", "lambda_batch_multiplier":6, "alfa_annealing_factor":2.0, "elite_sampling_strategy":"strategy3"}]'

            additional_args = get_dict(batch_size=batch_size, r_f_steps=r_f_steps, multiprocessing_backend=backend,
                                reproducible=reproducible,replay_out_cache_size=replay_out_cache_size, p=p,
                                feature_out_layer_size=feature_out_layer_size, lr_scheduler_steps=lr_scheduler_steps,
                                mini_batch_multiplier=mini_batch_multiplier, replay_parameters=replay_parameters)
            flags = change_args(**additional_args)

            if flags.reproducible:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                if float(torch.__version__[0: 3]) >= 1.8:
                    torch.use_deterministic_algorithms(True)
                else:
                    torch.set_deterministic(True)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.cuda.manual_seed(flags.seed)
                torch.cuda.manual_seed_all(flags.seed)
                torch.manual_seed(flags.seed)
                np.random.seed(flags.seed)
                random.seed(flags.seed)

            try:
                Learner(flags, run_id, additional_args).start()
            except Exception as e:
                logger.exception("Learner execution " + str(run_id) + "  interrupted by exception " + str(e))
                raise e


