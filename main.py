import os
import torch.multiprocessing as mp
import ray
import torch
import numpy as np
import random

from option_flags import flags
from agent.learner import Learner
from agent.tester import Tester


os.environ["OMP_NUM_THREADS"] = "1"


if __name__ == '__main__':
    if flags.multiprocessing_backend == "ray":
        ray.init()
    elif flags.multiprocessing_backend == "python_native":
        mp.set_start_method('spawn')
    else:
        raise NameError("Unknown multiprocessing backend selected - please check if the wording of the argument is correct.")

    if flags.reproducible:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        torch.set_deterministic(True)
        torch.cuda.manual_seed(flags.seed)
        torch.cuda.manual_seed_all(flags.seed)
        torch.manual_seed(flags.seed)
        np.random.seed(flags.seed)
        random.seed(flags.seed)

    if flags.op_mode == "train" or flags.op_mode == "train_w_load":
        for j in range(1):
            try:
                if flags.reproducible:
                    Learner().start_sync()
                else:
                    Learner().start_async()
            except Exception as e:
                # TODO log this error into the file
                print("main program crashed, info: " + str(e.args))
    elif flags.op_mode == "test":
        try:
            Tester(flags.test_episode_count, flags.load_model_uri).test(flags.render)
        except Exception as e:
            print("main program crashed, info: " + str(e.args))
            pass
    else:
        raise NameError(
            "Unknown operation mode selected - please check if the wording of the argument is correct.")

