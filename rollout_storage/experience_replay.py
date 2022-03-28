
import numpy as np
import signal
from utils.logger import logger
from rollout_storage.intefaces.replay_base import ReplayBase


class ExperienceReplayTorch(ReplayBase):
    def __init__(self, flags, training_event, replay_dict):
        super().__init__()
        self.flags = flags
        self.not_used = True
        self.filled = False
        self.position_pointer = 0
        self.replay_dict = replay_dict
        self.replay_capacity = self.replay_dict["capacity"]

        self.buffer = [None for i in range(self.replay_capacity)]
        self.replay_sample_ratio = self.replay_dict["sample_ratio"]
        self.finished = False
        self.training_event = training_event

        self.training_started = False
        self.fill_in_threshold = self.flags.training_fill_in_factor * self.replay_capacity
        if self.fill_in_threshold < 10 * self.flags.batch_size:
            self.fill_in_threshold = 10 * self.flags.batch_size

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
            if not self.not_used and (self.position_pointer % self.fill_in_threshold) == 0:
                self.replay_filled_event.set()
            if not self.not_used and (self.position_pointer % buf_size) == 0:
                self.filled = True
                self.replay_filled_event.set()

        index = self.position_pointer % buf_size

        if self.not_used:
            self.not_used = False

        self.position_pointer += 1
        return index

    def sample(self, batch_size, local_random=None):
        try:
            if not self.training_started:
                self.training_event.wait()
                self.training_started = True

            if self.filled:
                allowed_values = self.replay_capacity
            else:
                allowed_values = self.position_pointer
            if local_random is None:
                indices = np.random.choice(allowed_values, int(batch_size * self.replay_sample_ratio))
            else:
                indices = local_random.choice(allowed_values, int(batch_size * self.replay_sample_ratio))

            batch_data = [self.buffer[k] for k in indices]
        except Exception as exp:
            logger.exception("Replay buffer sampling has raised an exception:" + str(exp))
            signal.raise_signal(signal.SIGINT)
            return []

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
        self.position_pointer = 0

