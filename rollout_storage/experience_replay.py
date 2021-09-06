import numpy as np
import torch

from utils import decompress
from threading import Event


class ExperienceReplayTorch(object):
    def __init__(self, size, sample_ratio, options_flags, action_count, env_state_dim, batch_lock, state_compression=True):
        self.options_flags = options_flags
        self.size = size
        self.sample_ratio = sample_ratio
        self.state_compression = state_compression
        self.not_used = True
        self.filled = False
        self.pos_pointer = 0
        self.replay_filled_event = Event()
        self.env_state_dim = env_state_dim
        self.batch_lock = batch_lock

        if state_compression:
            self.states = []
        else:
            self.states = torch.zeros(self.size, options_flags.r_f_steps, *env_state_dim)
        self.actions = torch.zeros(self.size, options_flags.r_f_steps)
        self.rewards = torch.zeros(self.size, options_flags.r_f_steps)
        self.logits = torch.zeros(self.size, options_flags.r_f_steps, action_count)
        self.not_done = torch.zeros(self.size, options_flags.r_f_steps)

    def _store(self, index, **kwargs):
        if self.state_compression:
            if len(self.states) > index:
                self.states[index] = kwargs['state']
            else:
                self.states.append(kwargs['state'])
        else:
            self.states[index] = kwargs['state']
        self.actions[index] = kwargs['action']
        self.rewards[index] = kwargs['reward']
        self.logits[index] = kwargs['logits']
        self.not_done[index] = kwargs['not_done']

    def store_next(self, **kwargs):
        with self.batch_lock:
            index = self.calc_index(**kwargs)
            if index == -1:
                return
        self._store(index, **kwargs)

    def calc_index(self, **kwargs):
        if not self.filled:
            if not self.not_used and (self.pos_pointer % self.size) == 0:
                self.filled = True
                self.replay_filled_event.set()

        index = self.pos_pointer % self.size

        if self.not_used:
            self.not_used = False

        self.pos_pointer += 1
        return index

    def random_sample(self, batch_size):
        indices = np.random.choice(self.size, int(batch_size * self.sample_ratio))
        return self._get_batch(indices)

    def _get_batch(self, indices):
        if self.state_compression:
            states = []
            for k in indices:
                states.append(decompress(self.states[k]))
            states = torch.stack(states)
        else:
            states = self.states[indices]
        actions = self.actions[indices].long()
        rewards = self.rewards[indices]
        logits = self.logits[indices]
        not_done = self.not_done[indices]

        return states, actions, rewards, logits, not_done

    def on_policy_sample(self, batch_size):
        indices = []
        for i in range(self.pos_pointer, self.pos_pointer - batch_size, -1):
            if i < 0:
                indices.append(i + self.size)
            else:
                indices.append(i % self.size)
        return self._get_batch(indices)

