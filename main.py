import os

import torch.multiprocessing as mp
import numpy as np
import random
import datetime as dt
import torch.backends.cudnn

from option_flags import flags
from agent.learner_d.learner import Learner
from agent.tester import Tester
from utils import create_logger, logger


os.environ["OMP_NUM_THREADS"] = "1"
if flags.debug:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


if __name__ == '__main__':
    mp.set_start_method('spawn')

    if flags.op_mode == "train" or flags.op_mode == "train_w_load":

        run_id = int(dt.datetime.timestamp(dt.datetime.now()))
        save_url = "results/" + flags.env + "_" + str(run_id)
        os.makedirs(save_url)
        create_logger(save_url)
        logger.info("Starting execution " + str(run_id) + " with order number " + str(j))

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
    elif flags.op_mode == "test":
        try:
            Tester(flags.test_episode_count, flags.load_model_uri, flags).test(flags.render)
        except Exception as e:
            logger.exception("Tester execution interrupted by exception")
    else:
        raise NameError(
            "Unknown operation mode selected - please check if the wording of the argument is correct.")



