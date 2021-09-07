import numpy as np
import torch

from utils import decompress
from threading import Event
from option_flags import flags


class ExperienceReplayTorch(object):
    def __init__(self, batch_lock):
        self.not_used = True
        self.filled = False
        self.pos_pointer = 0
        self.replay_filled_event = Event()
        self.batch_lock = batch_lock

        if flags.use_state_compression:
            self.states = []
        else:
            self.states = torch.zeros(flags.replay_buffer_size, flags.r_f_steps, *flags.observation_shape)
        self.actions = torch.zeros(flags.replay_buffer_size, flags.r_f_steps)
        self.rewards = torch.zeros(flags.replay_buffer_size, flags.r_f_steps)
        self.logits = torch.zeros(flags.replay_buffer_size, flags.r_f_steps, flags.actions_count)
        self.not_done = torch.zeros(flags.replay_buffer_size, flags.r_f_steps)

    def _store(self, index, **kwargs):
        if flags.use_state_compression:
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
            if not self.not_used and (self.pos_pointer % flags.replay_buffer_size) == 0:
                self.filled = True
                self.replay_filled_event.set()

        index = self.pos_pointer % flags.replay_buffer_size

        if self.not_used:
            self.not_used = False

        self.pos_pointer += 1
        return index

    def random_sample(self, batch_size):
        indices = np.random.choice(flags.replay_buffer_size, int(batch_size * flags.replay_data_ratio))
        return self._get_batch(indices)

    def _get_batch(self, indices):
        if flags.use_state_compression:
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
                indices.append(i + flags.replay_buffer_size)
            else:
                indices.append(i % flags.replay_buffer_size)
        return self._get_batch(indices)

