import unittest
import os
import torch
import torch.backends.cudnn
import random
import numpy as np
import torch.multiprocessing as mp

from agent.learner import Learner
from option_flags import flags, change_args
from wrappers.atari_wrappers import make_atari, wrap_pytorch, wrap_deepmind


class ReproducibilityTest(unittest.TestCase):

    def setUp(self):
        self.env = "PongNoFrameskip-v4"
        placeholder_env = wrap_pytorch(
            wrap_deepmind(make_atari(self.env, 0), episode_life=True, clip_rewards=False, frame_stack=True,
                          scale=False))
        actions_count = placeholder_env.action_space.n
        observation_shape = placeholder_env.observation_space.shape
        placeholder_env.close()

        self.seed = 123456

        flags = change_args(r_f_steps=20, reproducible=True, env=self.env, seed=self.seed, envs_per_worker=2, actions_count=actions_count, observation_shape=observation_shape,
                                 batch_size=5, learner_thread_count=3, worker_count=5, training_max_steps=500, max_avg_reward=20, max_episodes=1000000, replay_out_cache_size=5,
                                 background_save=True, use_elite_set=False, replay_buffer_size=100, replay_data_ratio=1)

        self.flags = flags

    def test_reproducibility_ray(self):
        total_rew_run = [[], []]
        for i in range(2):
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            os.environ["OMP_NUM_THREADS"] = "1"
            # reseeding has to happen here before calling .start() on Learner to secure reproducibility (having it in setUp() is not enough)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            id = i
            save_url = "results/" + self.env + "_" + str(id)
            os.makedirs(save_url)
            Learner(self.flags, id).start()
            with open(save_url + "/Scores.txt") as file:
                for line in file:
                    total_rew_run[i].append(float(line.rstrip('\n')))

            os.system('rm -rf ' + save_url)
        length = min(len(total_rew_run[0]), len(total_rew_run[1]))
        if length < 10:
            raise EOFError("scores file captures less then 10 values - which not reliable enough to determine reproducibility - please raise ending condition")
        sum_rew_run_1 = 0
        sum_rew_run_2 = 0
        for i in range(length):
            sum_rew_run_1 = total_rew_run[0][i]
            sum_rew_run_2 = total_rew_run[1][i]

        self.assertEqual(sum_rew_run_1, sum_rew_run_2, 'total sum of rewards is not the same')

    def test_reproducibility_native(self):
        mp.set_start_method('spawn')
        flags = change_args(multiprocessing_backend="python_native")

        self.flags = flags
        total_rew_run = [[], []]
        for i in range(2):
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            os.environ["OMP_NUM_THREADS"] = "1"
            # reseeding has to happen here before calling .start() on Learner to secure reproducibility (having it in setUp() is not enough)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)
            id = i
            save_url = "results/" + self.env + "_" + str(id)
            os.makedirs(save_url)
            Learner(self.flags, id).start()
            with open(save_url + "/Scores.txt") as file:
                for line in file:
                    total_rew_run[i].append(float(line.rstrip('\n')))

            os.system('rm -rf ' + save_url)
        length = min(len(total_rew_run[0]), len(total_rew_run[1]))
        if length < 10:
            raise EOFError("scores file captures less then 10 values - which not reliable enough to determine reproducibility - please raise ending condition")
        sum_rew_run_1 = 0
        sum_rew_run_2 = 0
        for i in range(length):
            sum_rew_run_1 = total_rew_run[0][i]
            sum_rew_run_2 = total_rew_run[1][i]

        self.assertEqual(sum_rew_run_1, sum_rew_run_2, 'total sum of rewards is not the same')


if __name__ == '__main__':
    unittest.main()
