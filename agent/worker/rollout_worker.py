import sys

import torch
import numpy as np
import torch.nn.functional as F
import random
import torch.backends.cudnn
from wrappers import atari_wrappers

from rollout_storage.worker_buf.torch_worker_buffer import TorchWorkerBuffer
from queue import Queue
from model.network import ModelNetwork, StateTransformationNetwork


class RolloutWorker(object):
    def __init__(self, worker_id, flags, state_transf_model, file_save_url, verbose=False):
        self.device = torch.device("cpu")
        self.verbose = verbose
        self.flags = flags
        self.seed = flags.seed + worker_id * flags.envs_per_worker
        if flags.reproducible:
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            if float(torch.__version__[0: 3]) >= 1.8:
                torch.use_deterministic_algorithms(True)
            else:
                torch.set_deterministic(True)

        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)  # Beware random generation is NOT THREAD-SAFE !!
        random.seed(self.seed)

        self.model = ModelNetwork(flags.actions_count, flags).eval()

        if self.flags.op_mode == "train_w_load":
            state_dict = torch.load(file_save_url, map_location=self.device)
            self.model.load_state_dict(state_dict["model_state_dict"])

        self.feature_vec_dim = self.flags.feature_out_layer_size
        self.worker_id = worker_id

        self.episode_rewards = Queue(maxsize=flags.avg_buff_size)
        self.max_avg_reward = -sys.maxsize
        self.observations = []
        self.workers_buffers = []
        self.workers_envs = []

        self.iteration_counter = 0

        self.state_transf_network = StateTransformationNetwork(self.flags)
        self.state_transf_network.load_state_dict(state_transf_model)
        self.all_states = torch.zeros(flags.envs_per_worker, self.flags.r_f_steps, *self.flags.observation_shape)

        for i in range(flags.envs_per_worker):
            env = atari_wrappers.make_atari(flags.env, self.seed + i, clip_rewards=self.flags.clip_rewards, frames_skipped=self.flags.skipped_frames)

            self.workers_envs.append(env)
            self.workers_buffers.append(TorchWorkerBuffer((self.feature_vec_dim,), flags))
            init_state = torch.from_numpy(self.workers_envs[i].reset()).float()
            self.observations.append(init_state)
        self.observations = torch.stack(self.observations)

    def load_model(self, new_weights):
        self.model.load_state_dict(new_weights)
        self.model.eval()

    def exec_and_eval_rollout(self):
        iteration_rewards = []
        iteration_ep_steps = []
        self.iteration_counter += 1

        for i in range(len(self.workers_envs)):
            self.workers_buffers[i].reset()
            self.workers_buffers[i].states[0] = self.observations[i]
            self.all_states[i][0] = self.observations[i]

        for step in range(self.flags.r_f_steps):
            with torch.no_grad():
                logits, _ = self.model(self.observations)

                probs = F.softmax(logits, dim=-1)
                actions = probs.multinomial(num_samples=1)

            self.observations = []
            for i in range(len(self.workers_envs)):
                new_state, reward, terminal, _ = self.workers_envs[i].step(actions[i].item())

                if self.workers_envs[i].was_real_done:
                    ep_rewards, ep_steps = self.workers_envs[i].get_episode_metrics()

                    iteration_rewards.append(ep_rewards)
                    iteration_ep_steps.append(ep_steps)

                    if self.verbose:
                        if self.episode_rewards.full():
                            self.episode_rewards.get()
                        self.episode_rewards.put(ep_rewards)

                if terminal:
                    new_state = self.workers_envs[i].reset()

                new_state = torch.from_numpy(new_state).float()
                self.workers_buffers[i].insert(new_state, torch.from_numpy(np.array([[actions[i].item()]])),
                                               torch.from_numpy(np.array([[reward]])).float(),
                                               logits[i],
                                               not terminal)
                if self.flags.use_elite_set:
                    if step < (self.flags.r_f_steps - 1):
                        self.all_states[i][step + 1] = new_state
                self.observations.append(new_state)
            self.observations = torch.stack(self.observations)

        if self.flags.use_elite_set:
            with torch.no_grad():
                feature_vecs = self.state_transf_network(self.all_states)

            for i in range(len(self.workers_envs)):
                self.workers_buffers[i].feature_vec = feature_vecs[i]

        if self.verbose:
            avg_rew = np.average(list(self.episode_rewards.queue))
            if avg_rew > self.max_avg_reward:
                self.max_avg_reward = avg_rew
                print("Worker: " + str(self.worker_id) + " New local MAX avg(100)rew: ", "{:.2f}".format(self.max_avg_reward))

            if self.iteration_counter % self.flags.verbose_worker_out_int == 0:
                print('Worker: ' + str(self.worker_id) + '  LocalWorkerIteration: ', self.iteration_counter, " Avg(100)rew: ", "{:.2f}".format(avg_rew))

        return self.workers_buffers, self.worker_id, iteration_rewards, iteration_ep_steps
