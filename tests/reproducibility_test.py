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

        flags = change_args(op_mode="train", debug=False, lr=0.0004, batch_size=5, r_f_steps=20, baseline_loss_coef=0.5,
                            entropy_loss_coef=0.0005, max_grad_norm=40, gamma=0.99, c_const=1, rho_const=1, seed=self.seed, reproducible=True,
                            replay_out_cache_size=5, max_cache_pos_pointer=1000000, env=self.env, actions_count=actions_count, observation_shape=observation_shape,
                            learner_thread_count=1, worker_count=5,  envs_per_worker=2, multiprocessing_backend="ray", shared_queue_size=4,
                            max_episodes=1000000, training_max_steps=500, max_avg_reward=20,  max_avg_reward_deviation=0.5, max_avg_rew_time_accept_deviation=3600,
                            use_replay_buffer=True, replay_writer_queue_size=10, discarding_strategy="keep_latest", use_state_compression=True,
                            replay_buffer_size=70, elite_set_size=30, replay_data_ratio=0.7, elite_set_data_ratio=0.3, avg_buff_size=100, verbose_worker_out_int=50,
                            verbose_learner_out_int=250, use_elite_set=True, elite_reset_period=1000000, random_search=True, add_rew_feature=True, p=2,
                            elite_pop_strategy="lim_zero", background_save=True)

        self.flags = flags

    def test_reproducibility_ray_single_thread_learner(self):
        flags = change_args(multiprocessing_backend="ray", learner_thread_count=1)
        self.flags = flags
        self.make_2_runs_and_compare()

    def test_reproducibility_ray_multi_thread_learner(self):
        flags = change_args(multiprocessing_backend="ray", learner_thread_count=3)
        self.flags = flags
        self.make_2_runs_and_compare()

    def test_reproducibility_native_single_thread_learner(self):
        mp.set_start_method('spawn')
        flags = change_args(multiprocessing_backend="python_native", learner_thread_count=1)

        self.flags = flags
        self.make_2_runs_and_compare()

    def test_reproducibility_native_multi_thread_learner(self):
        mp.set_start_method('spawn')
        flags = change_args(multiprocessing_backend="python_native", learner_thread_count=3)

        self.flags = flags
        self.make_2_runs_and_compare()

    def make_2_runs_and_compare(self):
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
            raise EOFError(
                "scores file captures less then 10 values - which not reliable enough to determine reproducibility - please raise ending condition")
        sum_rew_run_1 = 0
        sum_rew_run_2 = 0
        for i in range(length):
            sum_rew_run_1 = total_rew_run[0][i]
            sum_rew_run_2 = total_rew_run[1][i]

        self.assertEqual(sum_rew_run_1, sum_rew_run_2, 'total sum of rewards is not the same')


if __name__ == '__main__':
    unittest.main()
