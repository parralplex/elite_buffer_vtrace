import os

import torch.multiprocessing as mp
import torch
import numpy as np
import random
import datetime as dt

from option_flags import flags, change_args
from agent.learner import Learner
from agent.tester import Tester
from utils import _set_up_logger, logger


os.environ["OMP_NUM_THREADS"] = "1"
if flags.debug:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    if flags.multiprocessing_backend == "python_native":
        mp.set_start_method('spawn')

    if flags.reproducible:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        torch.set_deterministic(True)
        torch.cuda.manual_seed(flags.seed)
        torch.cuda.manual_seed_all(flags.seed)
        torch.manual_seed(flags.seed)
        np.random.seed(flags.seed)
        random.seed(flags.seed)

    if flags.op_mode == "train" or flags.op_mode == "train_w_load":
        batch_size = 100
        r_f_steps = 20
        backend = "ray"
        for j in range(6):
            run_id = int(dt.datetime.timestamp(dt.datetime.now()))
            save_url = "results/" + flags.env + "_" + str(run_id)
            os.makedirs(save_url)
            _set_up_logger(save_url)

            if j % 2 == 1:
                backend = "python_native"
            else:
                if j > 0:
                    batch_size -= 40
                    r_f_steps += 20
                backend = "ray"
            flags = change_args(batch_size=batch_size, r_f_steps=r_f_steps, multiprocessing_backend=backend)
            try:
                if flags.reproducible:
                    Learner(flags, run_id).start_sync()
                else:
                    Learner(flags, run_id).start_async()
            except Exception as e:
                logger.exception("Learner execution interrupted by exception")
    elif flags.op_mode == "test":
        try:
            Tester(flags.test_episode_count, flags.load_model_uri, flags).test(flags.render)
        except Exception as e:
            logger.exception("Tester execution interrupted by exception")
    else:
        raise NameError(
            "Unknown operation mode selected - please check if the wording of the argument is correct.")



