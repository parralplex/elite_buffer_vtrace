import sys

from rollout_storage.elite_set.buf_population_strategy.lim_zero_strategy import LimZeroStrategy


class LimInfStrategy(LimZeroStrategy):
    def __init__(self, buf_size):
        super().__init__(buf_size)

    def process_dist(self, distance, index) -> bool:
        if distance == 0:
            self.update_entry = False
            return False
        elif distance > self.entry_distance:
            self.entry_idx = index
            self.entry_distance = distance

    def reset_idx(self):
        self.entry_idx = 0
        self.entry_distance = -sys.maxsize
