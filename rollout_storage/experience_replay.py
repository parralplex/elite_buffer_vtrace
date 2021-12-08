import numpy as np

from rollout_storage.intefaces.replay_base import ReplayBase


class ExperienceReplayTorch(ReplayBase):
    def __init__(self, flags, training_event):
        super().__init__()
        self.flags = flags
        self.not_used = True
        self.filled = False
        self.pos_pointer = 0

        self.buffer = [None for i in range(self.flags.replay_buffer_size)]
        self.replay_data_ratio = self.flags.replay_data_ratio
        self.finished = False
        self.training_event = training_event

    def _store(self, index, **kwargs):
        self.buffer[index] = kwargs['data']

    def store_next(self, **kwargs):
        index = self.calc_index(**kwargs)
        if index == -1:
            return False
        self._store(index, **kwargs)
        return True

    def calc_index(self, **kwargs):
        buf_size = len(self.buffer)
        if not self.filled:
            if not self.not_used and (self.pos_pointer % buf_size) == 0:
                self.filled = True
                self.replay_filled_event.set()

        index = self.pos_pointer % buf_size

        if self.not_used:
            self.not_used = False

        self.pos_pointer += 1
        return index

    def sample(self, batch_size, local_random=None):
        if local_random is None:
            indices = np.random.choice(self.flags.replay_buffer_size, int(batch_size * self.replay_data_ratio))
        else:
            indices = local_random.choice(self.flags.replay_buffer_size, int(batch_size * self.replay_data_ratio))

        batch_data = [self.buffer[k] for k in indices]
        return batch_data

    def close(self):
        self.finished = True
        if not self.replay_filled_event.is_set():
            self.replay_filled_event.set()

    def full(self):
        return False

    def reset(self):
        self.not_used = True
        self.filled = False
        self.pos_pointer = 0

