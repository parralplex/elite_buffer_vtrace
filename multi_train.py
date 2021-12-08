import os

import torch.multiprocessing as mp
import numpy as np
import random
import datetime as dt
import torch.backends.cudnn

from option_flags import flags, change_args
from agent.learner_d.learner import Learner
from utils import create_logger, logger, change_logger_file_handler


os.environ["OMP_NUM_THREADS"] = "1"
if flags.debug:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


if __name__ == '__main__':
    mp.set_start_method('spawn')

    for i in range(1):
        batch_size = 20
        r_f_steps = 15
        replay_out_cache_size = 1
        backend = "ray"
        reproducible = True

        replay_buffer_size = 600
        elite_set_data_ratio = 0.3
        elite_set_size = 30
        replay_data_ratio = 1
        elite_pop_strategy = "lim_zero"
        feature_out_layer_size = 512
        strategy1_trajectory_life = 15
        scheduler_steps = 10000
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

            # if j >= 3:
            #     p = 2
            # if j < 3:
            #     replay_buffer_size = 950
            #     elite_set_data_ratio = 0.06
            #     elite_set_size = 50
            #     replay_data_ratio = 0.94
            # elif 3 <= j < 6:
            #     replay_buffer_size = 900
            #     elite_set_data_ratio = 0.1
            #     elite_set_size = 100
            #     replay_data_ratio = 0.9
            # elif 6 <= j < 9:
            #     replay_buffer_size = 800
            #     elite_set_data_ratio = 0.2
            #     elite_set_size = 200
            #     replay_data_ratio = 0.8

            # replay_buffer_size = 1000
            # elite_set_size = 50
            # replay_data_ratio = 0.5
            # elite_set_data_ratio = 0.5

            flags = change_args(batch_size=batch_size, r_f_steps=r_f_steps, multiprocessing_backend=backend,
                                replay_buffer_size=replay_buffer_size, reproducible=reproducible,
                                replay_out_cache_size=replay_out_cache_size,
                                elite_set_size=elite_set_size, replay_data_ratio=replay_data_ratio,
                                elite_set_data_ratio=elite_set_data_ratio, elite_pop_strategy=elite_pop_strategy, p=p,
                                feature_out_layer_size=feature_out_layer_size,
                                strategy1_trajectory_life=strategy1_trajectory_life, scheduler_steps=scheduler_steps)

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
                Learner(flags, run_id).start()
            except Exception as e:
                logger.exception("Learner execution " + str(run_id) + "  interrupted by exception")
                raise e
