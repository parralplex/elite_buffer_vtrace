import os

import torch.multiprocessing as mp
import numpy as np
import random
import datetime as dt
import torch.backends.cudnn

from option_flags import flags, change_args
from agent.learner import Learner
from agent.tester import Tester
from utils import create_logger, logger, change_logger_file_handler


os.environ["OMP_NUM_THREADS"] = "1"
if flags.debug:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    mp.set_start_method('spawn')

    if flags.op_mode == "train" or flags.op_mode == "train_w_load":

        for i in range(1):
            batch_size = 20
            r_f_steps = 20
            replay_buffer_size = 1000
            backend = "ray"
            reproducible = True
            replay_out_cache_size = 20
            elite_set_data_ratio = 1
            elite_set_size = 1000
            replay_data_ratio = 1
            elite_pop_strategy = "lim_inf"
            for j in range(1):
                run_id = int(dt.datetime.timestamp(dt.datetime.now()))
                save_url = "results/" + flags.env + "_" + str(run_id)
                os.makedirs(save_url)
                if j == 0 and i == 0:
                    create_logger(save_url)
                else:
                    change_logger_file_handler(save_url)
                logger.info("Starting execution " + str(run_id) + " with order number " + str(j))

                if j == 0:
                    replay_buffer_size = 700
                    elite_set_size = 300
                    replay_data_ratio = 0.7
                    elite_set_data_ratio = 0.3
                elif j == 1:
                    replay_buffer_size = 850
                    elite_set_size = 150
                    replay_data_ratio = 0.85
                    elite_set_data_ratio = 0.15
                elif j == 2:
                    replay_buffer_size = 950
                    elite_set_size = 50
                    replay_data_ratio = 0.95
                    elite_set_data_ratio = 0.05

                flags = change_args(batch_size=batch_size, r_f_steps=r_f_steps, multiprocessing_backend=backend, replay_buffer_size=replay_buffer_size,reproducible=reproducible, replay_out_cache_size=replay_out_cache_size,
                                    elite_set_size=elite_set_size, replay_data_ratio=replay_data_ratio, elite_set_data_ratio=elite_set_data_ratio, elite_pop_strategy=elite_pop_strategy)

                if flags.reproducible:
                    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

                    torch.use_deterministic_algorithms(True)
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                    torch.cuda.manual_seed(flags.seed)
                    torch.cuda.manual_seed_all(flags.seed)
                    torch.manual_seed(flags.seed)
                    np.random.seed(flags.seed) # Beware that numpy.random. IS NOT THREAD-SAFE !! Avoid usage if possible!
                    random.seed(flags.seed)

                try:
                    Learner(flags, run_id).start()
                except Exception as e:
                    logger.exception("Learner execution " + str(run_id) + "  interrupted by exception")
    elif flags.op_mode == "test":
        try:
            Tester(flags.test_episode_count, flags.load_model_uri, flags).test(flags.render)
        except Exception as e:
            logger.exception("Tester execution interrupted by exception")
    else:
        raise NameError(
            "Unknown operation mode selected - please check if the wording of the argument is correct.")



