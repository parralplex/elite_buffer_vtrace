import torch
import threading
import os
import random
import numpy as np
import torch.backends.cudnn
import datetime as dt

from agent.algorithms.v_trace import v_trace
from agent.manager.native_worker_manager import NativeWorkerManager
from rollout_storage.elite_set.buf_population_strategy.lim_inf_strategy import LimInfStrategy
from rollout_storage.elite_set.elite_set_replay import EliteSetReplay
from rollout_storage.experience_replay import ExperienceReplayTorch
from rollout_storage.writer_queue.keep_latest_strategy import KeepLatestStrategy

from rollout_storage.writer_queue.replay_buffer_writer import ReplayWriterQueue
from torch.optim import Adam
from model.network import ModelNetwork
from stats.stats import Statistics


class Learner(object):
    def __init__(self, observation_shape, actions_count, options_flags):
        if options_flags.reproducible:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

            torch.set_deterministic(True)
            torch.cuda.manual_seed(options_flags.seed)
            torch.cuda.manual_seed_all(options_flags.seed)
            torch.manual_seed(options_flags.seed)
            np.random.seed(options_flags.seed)
            random.seed(options_flags.seed)

        self.run_ID = int(dt.datetime.timestamp(dt.datetime.now()))
        self.file_save_dir_ulr = "results/" + self.options_flags.env + "_" + str(self.run_ID)
        os.makedirs(self.file_save_dir_ulr)

        self.stats = Statistics(self.file_save_dir_ulr, True)

        self.options_flags = options_flags
        self.observation_shape = observation_shape
        self.actions_count = actions_count
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Learner is using CUDA")
        else:
            print("CUDA IS NOT AVAILABLE")
        self.model = ModelNetwork(self.actions_count).to(self.device)
        self.feature_reset_model = ModelNetwork(self.actions_count).eval()

        # self.model.load_state_dict(torch.load('results/BreakoutNoFrameskip-v4_1630692566/regular_model_save_.pt')["model_state_dict"])

        self.optimizer = Adam(self.model.parameters(), lr=self.options_flags.lr)

        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=2000, T_mult=1)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.9999)

        self.learning_lock = threading.Lock()
        self.batch_lock = threading.Lock()

        self.replay_buffers = [
            ExperienceReplayTorch(self.options_flags.buffer_size, self.options_flags.replay_data_ratio,
                                  self.options_flags, self.actions_count, self.observation_shape, self.batch_lock),
            EliteSetReplay(self.options_flags.elite_set_size, self.options_flags.elite_set_data_ratio,
                           self.options_flags, self.actions_count, self.observation_shape,
                           True, (self.model.get_flatten_layer_output_size(),),
                           LimInfStrategy(self.options_flags.elite_set_size))]

        self.model_warmed_up_event = threading.Event()

        self.replay_writer = ReplayWriterQueue(self.replay_buffers, queue_size=10,
                                               fill_in_strategy=KeepLatestStrategy())
        self.replay_writer.start()

        self.stop_event = threading.Event()

        self.training_iteration = 0

        self.training_event = threading.Event()
        self.worker_manager = NativeWorkerManager(True, self.stop_event, self.replay_writer, self.replay_buffers,
                                                  self.options_flags,
                                                  self.observation_shape, self.actions_count, self.model, self.stats)

    def start_async(self):
        threads = []
        thread = threading.Thread(target=self.worker_manager.manage_workers, name="manage_workers")
        thread.start()
        threads.append(thread)

        if not self.options_flags.reproducible:
            for i in range(3):
                thread = threading.Thread(target=self.learning, name="learning_thread-%d" % i)
                thread.start()
                threads.append(thread)

        for thread in threads:
            thread.join()

        self.stats.close()

    def start_sync(self):
        self.worker_manager.manage_workers(self._learning_iteration, True)
        self.stats.close()

    def learning(self):
        for p in range(len(self.replay_buffers)):
            self.replay_buffers[p].replay_filled_event.wait()
        # self.model_warmed_up_event.wait()  TODO WARM_UP PERIOD EVENT WHEN CONTINUING LEARNING PROCESS FROM MODEL SAVED IN FILE

        self.training_event.set()
        self.stats.mark_warm_up_period()
        while self.training_iteration < self.options_flags.training_max_steps:
            if self.stop_event.is_set():
                break
            self._learning_iteration()
        self.stop_event.set()

    def _learning_iteration(self, check_events=False):
        if check_events:
            for p in range(len(self.replay_buffers)):
                if not self.replay_buffers[p].replay_filled_event.is_set():
                    return
            if not self.model_warmed_up_event.is_set():
                return
            if not self.training_event.is_set():
                self.training_event.set()
            if self.training_iteration >= self.options_flags.training_max_steps:
                self.stop_event.set()
                return

        actions, beh_logits, not_done, rewards, states = self._prepare_batch()

        with self.learning_lock:
            self.training_iteration += 1
            bootstrap_value, current_logits, current_values = self._foward_pass(states)

            baseline_loss, entropy_loss, policy_loss = v_trace(actions, beh_logits, bootstrap_value,
                                                               current_logits, current_values, not_done, rewards,
                                                               self.options_flags, self.device)

            self._backprop(policy_loss, baseline_loss, entropy_loss)

        self.worker_manager.update_model_data(self.model)
        if self.options_flags.use_elite_set:
            self._update_elite_features()
        if self.training_iteration % 1000 == 0:
            self._save_current_model_state()

    def _foward_pass(self, states):
        current_logits, current_values = self.model(states.detach(), no_feature_vec=True)

        bootstrap_value = current_values[-1].squeeze(-1)
        current_values = current_values.squeeze(-1)

        return bootstrap_value, current_logits, current_values

    def _backprop(self, policy_loss, baseline_loss, entropy_loss):
        loss = policy_loss + baseline_loss + entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.options_flags.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.stats.process_learning_iter(policy_loss.item(), baseline_loss.item(), entropy_loss.item(), self.lr_scheduler.get_last_lr()[0])

    def _prepare_batch(self):
        states, actions, rewards, beh_logits, not_done = self.replay_buffers[0].random_sample(
            self.options_flags.batch_size)

        for i in range(1, len(self.replay_buffers)):
            states_n, actions_n, rewards_n, beh_logits_n, not_done_n = self.replay_buffers[i].random_sample(
                self.options_flags.batch_size)
            states = torch.cat((states, states_n), 0)
            actions = torch.cat((actions, actions_n), 0)
            rewards = torch.cat((rewards, rewards_n), 0)
            beh_logits = torch.cat((beh_logits, beh_logits_n), 0)
            not_done = torch.cat((not_done, not_done_n), 0)

        states, actions, rewards, beh_logits, not_done = states.to(self.device).transpose(1, 0), actions.to(
            self.device).transpose(1, 0), rewards.to(self.device).transpose(1, 0), beh_logits.to(self.device).transpose(
            1, 0), not_done.to(self.device).transpose(1, 0)

        return actions, beh_logits, not_done, rewards, states

    def _update_elite_features(self):
        if self.training_iteration % 200 == 0:
            with self.batch_lock:
                print("recalculating features")
                prior_states = self.replay_buffers[1].get_prior_buf_states()

                self.feature_reset_model.load_state_dict({k: v.cpu() for k, v in self.model.state_dict().items()})

                with torch.no_grad():
                    _, _, feature_vecs_prior = self.feature_reset_model(prior_states, True)
                self.replay_buffers[1].set_feature_vecs_prior(feature_vecs_prior)

                # feature_vecoefs = None
                # for j in range(6):
                #     with torch.no_grad():
                #         _, _, feature_vecoefs_prior = self.model(prior_states[j*100:(j+1)*100].cuda(), True)
                #     if feature_vecoefs is None:
                #         feature_vecoefs = feature_vecoefs_prior.cpu()
                #     else:
                #         feature_vecoefs = torch.cat((feature_vecoefs, feature_vecoefs_prior.cpu()), 0)
                # self.replay_buffer.set_feature_vecoefs_prior(feature_vecoefs)

                print("recalculating features DONE")

    def _save_current_model_state(self):
        torch.save({"model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.lr_scheduler.state_dict(),
                    "flags": vars(self.options_flags),
                    },
                   self.file_save_dir_ulr + '/regular_model_save_.pt')

    def _load_model_state(self, url):
        # TODO check if url is valid
        state_dict = torch.load(url)
        self.model.load_state_dict(state_dict["model_state_dict"])
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        self.options_flags = state_dict["flags"]
