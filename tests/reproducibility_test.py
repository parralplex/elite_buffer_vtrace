import unittest
import os
import torch
import torch.backends.cudnn
import random
import numpy as np
import torch.multiprocessing as mp
import datetime

from agent.learner_d.learner import Learner
from multi_train import get_dict
from option_flags import change_args
from wrappers.atari_wrappers import make_stock_atari
import timeout_decorator


LOCAL_TIMEOUT = 3000


def get_args(multiprocessing_backend, learner_thread_count, env):
    placeholder_env = make_stock_atari(env)
    actions_count = placeholder_env.action_space.n
    observation_shape = placeholder_env.observation_space.shape
    placeholder_env.close()

    seed = 123456

    additional_args = get_dict(op_mode="train", debug=False, save_model_period=1000, lr=0.0004, batch_size=4, r_f_steps=20,
                        gradient_clip_by_norm_threshold=40, gamma=0.99, c_const=1, rho_const=1,
                        seed=seed, reproducible=True, env=env, actions_count=actions_count,
                        observation_shape=observation_shape,
                        clip_rewards=True, reward_clipping_method="abs_one_sign", skipped_frames=4, frames_stacked=4,
                        noop_threshold=30, grayscaling_frames=True, frame_scale_resolution="84,84",
                        learner_thread_count=learner_thread_count, worker_count=5, envs_per_worker=2,
                        multiprocessing_backend=multiprocessing_backend,
                        shared_queue_size=4, environment_max_steps=30000, replay_writer_queue_size=1,
                        discarding_strategy="keep_latest",
                        use_replay_compression=True, lz4_compression_level=0, caching=True, caching_threads=1,
                        cache_sample_size=1, cache_output_buffer_size=3,
                        avg_buff_size=100, verbose_worker_out_int=50, verbose_learner_out_int=250, basic_verbose=True,
                        worker_verbose=False, background_save=False,
                        feature_out_layer_size=512, use_additional_scaling_FC_layer=True, lr_scheduler_steps=3000,
                        optimizer="rmsprop", rmsprop_eps=0.01, lr_end_value=0.00001,
                        replay_parameters='[{"type": "queue", "capacity": 1, "sample_ratio": 0.5}, {"type": "standard", "capacity": 500, "sample_ratio": 0.5}]',
                        training_fill_in_factor=0.2,
                        policy_gradient_loss_weight=1, value_loss_weight=0.5, entropy_loss_weight=0.01,
                        use_kl_mask=False)
    flags = change_args(**additional_args)
    return flags, additional_args


class ReproducibilityTest(unittest.TestCase):

    @timeout_decorator.timeout(LOCAL_TIMEOUT)
    def test_reproducibility_ray_single_thread_learner(self):
        env = "PongNoFrameskip-v4"
        self.flags, self.additional_args = get_args(multiprocessing_backend="ray", learner_thread_count=1, env=env)
        self.make_2_runs_and_compare()

    @timeout_decorator.timeout(LOCAL_TIMEOUT)
    def test_reproducibility_ray_multi_thread_learner(self):
        env = "PongNoFrameskip-v4"
        self.flags, self.additional_args = get_args(multiprocessing_backend="ray", learner_thread_count=3, env=env)
        self.make_2_runs_and_compare()

    @timeout_decorator.timeout(LOCAL_TIMEOUT)
    def test_reproducibility_native_single_thread_learner(self):
        env = "PongNoFrameskip-v4"
        try:
            mp.set_start_method('spawn')
        except RuntimeError as exp:
            pass
        self.flags, self.additional_args = get_args(multiprocessing_backend="python_native", learner_thread_count=1, env=env)
        self.make_2_runs_and_compare()

    @timeout_decorator.timeout(LOCAL_TIMEOUT)
    def test_reproducibility_native_multi_thread_learner(self):
        env = "PongNoFrameskip-v4"
        try:
            mp.set_start_method('spawn')
        except RuntimeError as exp:
            pass
        self.flags, self.additional_args = get_args(multiprocessing_backend="python_native", learner_thread_count=3, env=env)
        self.make_2_runs_and_compare()

    def make_2_runs_and_compare(self):
        env = "PongNoFrameskip-v4"
        seed = 123456
        total_rew_run = [[], []]
        for i in range(2):
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            os.environ["OMP_NUM_THREADS"] = "1"
            # reseeding has to happen here before calling .start() on Learner to secure reproducibility (having it in setUp() is not enough)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            run_id = int(datetime.datetime.timestamp(datetime.datetime.now()))
            save_url = "results/" + env + "_" + str(run_id)
            os.makedirs(save_url)
            Learner(self.flags, run_id, self.additional_args).start()
            with open(save_url + "/Scores.txt") as file:
                for line in file:
                    total_rew_run[i].append(float(line.split(',')[0]))

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

