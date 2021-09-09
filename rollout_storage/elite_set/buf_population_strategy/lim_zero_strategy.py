import sys
import torch

from rollout_storage.intefaces.elite_set_insert_strategy import EliteSetInsertStrategy
from rollout_storage.elite_set.decoratos.rand_ouput import rand_output
from option_flags import flags


class LimZeroStrategy(EliteSetInsertStrategy):

    def __init__(self, flags):
        super().__init__(flags)
        self.min_sum_reward = -sys.maxsize
        self.rew_sum = torch.zeros(self.flags.elite_set_size)
        self.entry_idx = 0
        self.entry_distance = sys.maxsize
        self.update_entry = True

    @rand_output(chance=flags.rnd_index_chance)
    def calculate_best_index_pos(self, feature_vecs, new_feature_vec, new_reward, **kwargs):
        self.update_entry = True
        entry_rew = torch.sum(new_reward)
        if entry_rew < self.min_sum_reward:
            return -1

        self.reset_idx()

        if kwargs['add_rew_feature']:
            feature_vec_rewarded = new_feature_vec + entry_rew
            super(LimZeroStrategy, self).calculate_best_index_pos(feature_vecs, feature_vec_rewarded, new_reward, **kwargs)
        else:
            super(LimZeroStrategy, self).calculate_best_index_pos(feature_vecs, new_feature_vec,
                                                                  new_reward, **kwargs)
        
        if not self.update_entry or entry_rew < self.rew_sum[self.entry_idx]:
            return -1

        if self.rew_sum[self.entry_idx] == self.min_sum_reward:
            self.rew_sum[self.entry_idx] = entry_rew
            self.min_sum_reward = min(self.rew_sum)
        else:
            self.rew_sum[self.entry_idx] = entry_rew
            
        return self.entry_idx

    def process_dist(self, distance, index) -> bool:
        if distance == 0:
            self.update_entry = False
            return False
        elif distance < self.entry_distance:
            self.entry_idx = index
            self.entry_distance = distance

    def reset_idx(self):
        self.entry_idx = 0
        self.entry_distance = sys.maxsize

    def on_insert_before_filled(self, index, **kwargs):
        entry_rew = torch.sum(kwargs['reward'])
        if entry_rew < self.min_sum_reward:
            self.min_sum_reward = entry_rew
        self.rew_sum[index] = entry_rew

    def recalculate_dist(self, feature_vecs):
        pass
            