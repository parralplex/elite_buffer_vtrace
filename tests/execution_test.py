import unittest
import os
import datetime
import timeout_decorator
import torch.multiprocessing as mp

from agent.learner_d.learner import Learner
from agent.tester import Tester
from multi_train import get_dict
from option_flags import change_args

LOCAL_TIMEOUT = 3000


class TestExecution(unittest.TestCase):
    @timeout_decorator.timeout(LOCAL_TIMEOUT)
    def test_caching_with_threads_ray(self):
        try:
            env = "BreakoutNoFrameskip-v4"
            additional_args = get_dict(op_mode="train",  lr=0.0004, batch_size=10, r_f_steps=20,
                                env=env, learner_thread_count=2, worker_count=5, envs_per_worker=3,multiprocessing_backend="ray",
                                environment_max_steps=500000,  caching=True, caching_threads=2,cache_sample_size=2,
                                replay_parameters='[{"type": "queue", "capacity": 1, "sample_ratio": 0.5}, {"type": "standard", "capacity": 500, "sample_ratio": 0.5}]')
            flags = change_args(**additional_args)
            run_id = int(datetime.datetime.timestamp(datetime.datetime.now()))
            save_url = "results/" + env + "_" + str(run_id)
            os.makedirs(save_url)
            Learner(flags, run_id, additional_args).start()
            os.system('rm -rf ' + save_url)
        except Exception as exp:
            self.fail("Unexpected  exception during execution: " + str(exp))

    @timeout_decorator.timeout(LOCAL_TIMEOUT)
    def test_caching_with_threads_python_native(self):
        try:
            try:
                mp.set_start_method('spawn')
            except RuntimeError as exp:
                pass
            env = "BreakoutNoFrameskip-v4"
            additional_args = get_dict(op_mode="train",  lr=0.0004, batch_size=10, r_f_steps=20,
                                env=env, learner_thread_count=2, worker_count=5, envs_per_worker=3,multiprocessing_backend="python_native",
                                environment_max_steps=500000,  caching=True, caching_threads=2,cache_sample_size=2,
                                replay_parameters='[{"type": "queue", "capacity": 1, "sample_ratio": 0.5}, {"type": "standard", "capacity": 500, "sample_ratio": 0.5}]')
            flags = change_args(**additional_args)
            run_id = int(datetime.datetime.timestamp(datetime.datetime.now()))
            save_url = "results/" + env + "_" + str(run_id)
            os.makedirs(save_url)
            Learner(flags, run_id, additional_args).start()
            os.system('rm -rf ' + save_url)
        except Exception as exp:
            self.fail("Unexpected  exception during execution: " + str(exp))

    @timeout_decorator.timeout(LOCAL_TIMEOUT)
    def test_elite_sampling_strategy1(self):
        try:
            try:
                mp.set_start_method('spawn')
            except RuntimeError as exp:
                pass
            env = "BreakoutNoFrameskip-v4"
            additional_args = get_dict(op_mode="train",  lr=0.0004, batch_size=10, r_f_steps=20,
                                env=env, learner_thread_count=2, worker_count=5, envs_per_worker=3, multiprocessing_backend="python_native",
                                environment_max_steps=500000,  caching=True, caching_threads=2,cache_sample_size=2,
                                replay_parameters='[{"type": "queue", "capacity": 1, "sample_ratio": 0.5}, {"type": "custom", "capacity": 1000, "sample_ratio": 0.5, "dist_function":"ln_norm", "sample_strategy":"elite_sampling", "lambda_batch_multiplier":6, "alfa_annealing_factor":2.0, "elite_sampling_strategy":"strategy1"}]')
            flags = change_args(**additional_args)
            run_id = int(datetime.datetime.timestamp(datetime.datetime.now()))
            save_url = "results/" + env + "_" + str(run_id)
            os.makedirs(save_url)
            Learner(flags, run_id, additional_args).start()
            os.system('rm -rf ' + save_url)
        except Exception as exp:
            self.fail("Unexpected  exception during execution: " + str(exp))

    @timeout_decorator.timeout(LOCAL_TIMEOUT)
    def test_elite_sampling_strategy2(self):
        try:
            try:
                mp.set_start_method('spawn')
            except RuntimeError as exp:
                pass
            env = "BreakoutNoFrameskip-v4"
            additional_args = get_dict(op_mode="train",  lr=0.0004, batch_size=10, r_f_steps=20,
                                env=env, learner_thread_count=2, worker_count=5, envs_per_worker=3, multiprocessing_backend="python_native",
                                environment_max_steps=500000,  caching=True, caching_threads=2,cache_sample_size=2,
                                replay_parameters='[{"type": "queue", "capacity": 1, "sample_ratio": 0.5}, {"type": "custom", "capacity": 1000, "sample_ratio": 0.5, "dist_function":"ln_norm", "sample_strategy":"elite_sampling", "lambda_batch_multiplier":6, "alfa_annealing_factor":2.0, "elite_sampling_strategy":"strategy2"}]')
            flags = change_args(**additional_args)
            run_id = int(datetime.datetime.timestamp(datetime.datetime.now()))
            save_url = "results/" + env + "_" + str(run_id)
            os.makedirs(save_url)
            Learner(flags, run_id, additional_args).start()
            os.system('rm -rf ' + save_url)
        except Exception as exp:
            self.fail("Unexpected  exception during execution: " + str(exp))

    @timeout_decorator.timeout(LOCAL_TIMEOUT)
    def test_elite_sampling_strategy3(self):
        try:
            try:
                mp.set_start_method('spawn')
            except RuntimeError as exp:
                pass
            env = "BreakoutNoFrameskip-v4"
            additional_args = get_dict(op_mode="train", lr=0.0004, batch_size=10, r_f_steps=20,
                                env=env, learner_thread_count=2, worker_count=5, envs_per_worker=3,
                                multiprocessing_backend="python_native",
                                environment_max_steps=500000, caching=True, caching_threads=2, cache_sample_size=2,
                                replay_parameters='[{"type": "queue", "capacity": 1, "sample_ratio": 0.5}, {"type": "custom", "capacity": 1000, "sample_ratio": 0.5, "dist_function":"ln_norm", "sample_strategy":"elite_sampling", "lambda_batch_multiplier":6, "alfa_annealing_factor":2.0, "elite_sampling_strategy":"strategy3"}]')
            flags = change_args(**additional_args)
            run_id = int(datetime.datetime.timestamp(datetime.datetime.now()))
            save_url = "results/" + env + "_" + str(run_id)
            os.makedirs(save_url)
            Learner(flags, run_id, additional_args).start()
            os.system('rm -rf ' + save_url)
        except Exception as exp:
            self.fail("Unexpected  exception during execution: " + str(exp))

    @timeout_decorator.timeout(LOCAL_TIMEOUT)
    def test_elite_sampling_strategy4(self):
        try:
            try:
                mp.set_start_method('spawn')
            except RuntimeError as exp:
                pass
            env = "BreakoutNoFrameskip-v4"
            additional_args = get_dict(op_mode="train", lr=0.0004, batch_size=10, r_f_steps=20,
                                env=env, learner_thread_count=2, worker_count=5, envs_per_worker=3,
                                multiprocessing_backend="python_native",
                                environment_max_steps=500000, caching=True, caching_threads=2, cache_sample_size=2,
                                replay_parameters='[{"type": "queue", "capacity": 1, "sample_ratio": 0.5}, {"type": "custom", "capacity": 1000, "sample_ratio": 0.5, "dist_function":"ln_norm", "sample_strategy":"elite_sampling", "lambda_batch_multiplier":6, "alfa_annealing_factor":2.0, "elite_sampling_strategy":"strategy4"}]')
            flags = change_args(**additional_args)
            run_id = int(datetime.datetime.timestamp(datetime.datetime.now()))
            save_url = "results/" + env + "_" + str(run_id)
            os.makedirs(save_url)
            Learner(flags, run_id, additional_args).start()
            os.system('rm -rf ' + save_url)
        except Exception as exp:
            self.fail("Unexpected  exception during execution: " + str(exp))

    @timeout_decorator.timeout(LOCAL_TIMEOUT)
    def test_attentive_sampling(self):
        try:
            try:
                mp.set_start_method('spawn')
            except RuntimeError as exp:
                pass
            env = "BreakoutNoFrameskip-v4"
            additional_args = get_dict(op_mode="train", lr=0.0004, batch_size=10, r_f_steps=20,
                                env=env, learner_thread_count=2, worker_count=5, envs_per_worker=3,
                                multiprocessing_backend="python_native",
                                environment_max_steps=500000, caching=True, caching_threads=2, cache_sample_size=2,
                                replay_parameters='[{"type": "queue", "capacity": 1, "sample_ratio": 0.5}, {"type": "custom", "capacity": 1000, "sample_ratio": 0.5, "dist_function":"ln_norm", "sample_strategy":"attentive_sampling", "lambda_batch_multiplier":6, "alfa_annealing_factor":2.0}]')
            flags = change_args(**additional_args)
            run_id = int(datetime.datetime.timestamp(datetime.datetime.now()))
            save_url = "results/" + env + "_" + str(run_id)
            os.makedirs(save_url)
            Learner(flags, run_id, additional_args).start()
            os.system('rm -rf ' + save_url)
        except Exception as exp:
            self.fail("Unexpected  exception during execution: " + str(exp))

    @timeout_decorator.timeout(LOCAL_TIMEOUT)
    def test_elite_insertion(self):
        try:
            try:
                mp.set_start_method('spawn')
            except RuntimeError as exp:
                pass
            env = "BreakoutNoFrameskip-v4"
            additional_args = get_dict(op_mode="train", lr=0.0004, batch_size=10, r_f_steps=20,
                                env=env, learner_thread_count=2, worker_count=5, envs_per_worker=3,
                                multiprocessing_backend="ray",
                                environment_max_steps=500000, caching=True, caching_threads=2, cache_sample_size=2,
                                replay_parameters='[{"type": "queue", "capacity": 1, "sample_ratio": 0.5}, {"type": "custom", "capacity": 1000, "sample_ratio": 0.5, "dist_function":"ln_norm", "insert_strategy":"elite_insertion", "elite_batch_size":20}]')
            flags = change_args(**additional_args)
            run_id = int(datetime.datetime.timestamp(datetime.datetime.now()))
            save_url = "results/" + env + "_" + str(run_id)
            os.makedirs(save_url)
            Learner(flags, run_id, additional_args).start()
            os.system('rm -rf ' + save_url)
        except Exception as exp:
            self.fail("Unexpected  exception during execution: " + str(exp))

    @timeout_decorator.timeout(LOCAL_TIMEOUT)
    def test_elite_insertion_and_sampling(self):
        try:
            try:
                mp.set_start_method('spawn')
            except RuntimeError as exp:
                pass
            env = "BreakoutNoFrameskip-v4"
            additional_args = get_dict(op_mode="train", lr=0.0004, batch_size=10, r_f_steps=20,
                                env=env, learner_thread_count=2, worker_count=5, envs_per_worker=3,
                                multiprocessing_backend="ray",
                                environment_max_steps=500000, caching=True, caching_threads=2, cache_sample_size=2,
                                replay_parameters='[{"type": "queue", "capacity": 1, "sample_ratio": 0.5}, {"type": "custom", "capacity": 1000, "sample_ratio": 0.5, "dist_function":"ln_norm", "insert_strategy":"elite_insertion", "elite_batch_size":5, "sample_strategy":"elite_sampling", "lambda_batch_multiplier":3, "alfa_annealing_factor":2.0, "elite_sampling_strategy":"strategy4"}]')
            flags = change_args(**additional_args)
            run_id = int(datetime.datetime.timestamp(datetime.datetime.now()))
            save_url = "results/" + env + "_" + str(run_id)
            os.makedirs(save_url)
            Learner(flags, run_id, additional_args).start()
            os.system('rm -rf ' + save_url)
        except Exception as exp:
            self.fail("Unexpected  exception during execution: " + str(exp))

    @timeout_decorator.timeout(LOCAL_TIMEOUT)
    def test_training_from_checkpoint(self):
        try:

            env = "BreakoutNoFrameskip-v4"
            run_id = int(datetime.datetime.timestamp(datetime.datetime.now()))
            save_url = "results/" + env + "_" + str(run_id)
            os.makedirs(save_url)
            for i in range(2):
                if i == 0:
                    op_mode = "train"
                else:
                    op_mode = "train_w_load"
                additional_args = get_dict(op_mode=op_mode, load_model_url=save_url, lr=0.0004, batch_size=10, r_f_steps=20,
                                    env=env, learner_thread_count=2, worker_count=5, envs_per_worker=3,
                                    multiprocessing_backend="ray",
                                    environment_max_steps=50000, caching=True, caching_threads=2, cache_sample_size=2,
                                    replay_parameters='[{"type": "queue", "capacity": 1, "sample_ratio": 0.5}, {"type": "custom", "capacity": 1000, "sample_ratio": 0.5, "dist_function":"ln_norm", "insert_strategy":"elite_insertion", "elite_batch_size":20, "sample_strategy":"elite_sampling", "lambda_batch_multiplier":6, "alfa_annealing_factor":2.0, "elite_sampling_strategy":"strategy4"}]')

                flags = change_args(**additional_args)
                Learner(flags, run_id, additional_args).start()
            os.system('rm -rf ' + save_url)
        except Exception as exp:
            self.fail("Unexpected  exception during execution: " + str(exp))

    @timeout_decorator.timeout(LOCAL_TIMEOUT)
    def test_agent_evaluation(self):
        try:
            env = "PongNoFrameskip-v4"
            run_id = int(datetime.datetime.timestamp(datetime.datetime.now()))
            save_url = "results/" + env + "_" + str(run_id)
            os.makedirs(save_url)
            for i in range(2):
                if i == 0:
                    op_mode = "train"
                else:
                    op_mode = "test"
                additional_args = get_dict(op_mode=op_mode, load_model_url=save_url, lr=0.0004, batch_size=10, r_f_steps=20,
                                    env=env, learner_thread_count=2, worker_count=5, envs_per_worker=3,
                                    multiprocessing_backend="ray", render=False, test_episode_count=20,
                                    environment_max_steps=50000, caching=True, caching_threads=2, cache_sample_size=2,
                                    replay_parameters='[{"type": "queue", "capacity": 1, "sample_ratio": 0.5}, {"type": "custom", "capacity": 1000, "sample_ratio": 0.5, "dist_function":"ln_norm", "insert_strategy":"elite_insertion", "elite_batch_size":20, "sample_strategy":"elite_sampling", "lambda_batch_multiplier":6, "alfa_annealing_factor":2.0, "elite_sampling_strategy":"strategy4"}]')
                flags = change_args(**additional_args)
                if i == 0:
                    Learner(flags, run_id, additional_args).start()
                else:
                    avg_score, awg_steps = Tester(flags.test_episode_count, flags.load_model_url, flags).test()
            os.system('rm -rf ' + save_url)
        except Exception as exp:
            self.fail("Unexpected  exception during execution: " + str(exp))


