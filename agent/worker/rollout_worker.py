import sys

import torch
import numpy as np
import torch.nn.functional as F
import random
import torch.backends.cudnn
from wrappers import atari_wrappers

from rollout_storage.worker_buf.torch_worker_buffer import TorchWorkerBuffer
from queue import Queue
from model.network import ModelNetwork


class RolloutWorker(object):
    def __init__(self, id, flags, verbose=False):
        self.device = torch.device("cpu")
        self.verbose = verbose
        self.flags = flags
        if flags.reproducible:
            torch.cuda.manual_seed(flags.seed)
            torch.cuda.manual_seed_all(flags.seed)
            torch.manual_seed(flags.seed)
            np.random.seed(flags.seed)
            random.seed(flags.seed)

        self.model = ModelNetwork(flags.actions_count).eval()

        self.feature_vec_dim = self.model.get_flatten_layer_output_size()
        self.worker_id = id

        self.episode_rewards = Queue(maxsize=flags.avg_buff_size)
        self.max_avg_reward = -sys.maxsize
        self.env_steps = []
        self.rewards = []
        self.observations = []
        self.workers_buffers = []
        self.workers_envs = []

        for i in range(flags.envs_per_worker):
            self.rewards.append(0.0)
            self.env_steps.append(0)
            env = atari_wrappers.wrap_pytorch(
                atari_wrappers.wrap_deepmind(
                    atari_wrappers.make_atari(flags.env, flags.seed),
                    episode_life=True,
                    clip_rewards=False,
                    frame_stack=True,
                    scale=False,
                )
            )

            self.workers_envs.append(env)
            self.workers_buffers.append(
                TorchWorkerBuffer((self.feature_vec_dim,), flags))
            init_state = torch.from_numpy(self.workers_envs[i].reset()).float()
            self.observations.append(init_state)
        self.observations = torch.stack(self.observations)

        self.iteration_counter = 0

    def performing(self, model_state_dict=None, update=False):
        with torch.no_grad():
            if update:
                self.model.load_state_dict(model_state_dict)
                self.model.eval()

            iteration_rewards = []
            iteration_ep_steps = []
            self.iteration_counter += 1

            for i in range(len(self.workers_envs)):
                self.workers_buffers[i].reset()
                self.workers_buffers[i].states[0] = self.observations[i]

            for step in range(self.flags.r_f_steps):
                with torch.no_grad():
                    logits, _, feature_vecs = self.model(self.observations)

                    probs = F.softmax(logits, dim=-1)
                    actions = probs.multinomial(num_samples=1).detach()

                    self.observations = []
                    for i in range(len(self.workers_envs)):
                        new_state, reward, terminal, ter_real = self.workers_envs[i].step(actions[i].item())

                        self.rewards[i] += reward
                        self.env_steps[i] += 1
                        if terminal:
                            new_state = self.workers_envs[i].reset()

                            if self.verbose:
                                if self.episode_rewards.full():
                                    self.episode_rewards.get()
                                self.episode_rewards.put(self.rewards[i])

                            iteration_rewards.append(self.rewards[i])
                            iteration_ep_steps.append(self.env_steps[i])
                            self.rewards[i] = 0.0
                            self.env_steps[i] = 0
                        new_state = torch.from_numpy(new_state).float()
                        self.workers_buffers[i].insert(new_state, torch.from_numpy(np.array([[actions[i].item()]])),
                                                       torch.from_numpy(np.array([[reward]])).float(),
                                                       logits[i],
                                                       not terminal,
                                                       feature_vecs[i])
                        self.observations.append(new_state)
                    self.observations = torch.stack(self.observations)

            if self.verbose:
                avg_rew = np.average(list(self.episode_rewards.queue))
                if avg_rew > self.max_avg_reward:
                    self.max_avg_reward = avg_rew
                    print("Worker: " + str(self.worker_id) + " New MAX avg(100)rew: ", "{:.2f}".format(self.max_avg_reward))

                if self.iteration_counter % self.flags.verbose_worker_out_int == 0:
                    print('Worker: ' + str(self.worker_id) + '  WorkerIteration: ', self.iteration_counter, " Avg(100)rew: ",
                              "{:.2f}".format(avg_rew))

            return self.workers_buffers, self.worker_id, iteration_rewards, iteration_ep_steps
