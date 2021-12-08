import torch
import sys

from rollout_storage.elite_set.elite_set_replay import EliteSetReplay
from rollout_storage.intefaces.elite_set_insert_strategy import EliteSetInputStrategy


class DistFilteringInputStrategy(EliteSetInputStrategy):
    def __init__(self, flags):
        super().__init__(flags)
        self.min_sum_reward = sys.maxsize
        self.rew_sum = torch.zeros(self.flags.elite_set_size)
        self.index_time_array = [0 for i in range(self.flags.elite_set_size)]
        self.empty_indices = []

    def calc_index(self, index, filled, feature_vecs, **kwargs):
        entry_rew = torch.sum(abs(kwargs['reward']))

        if not filled:
            # index = elite_ref(**kwargs)
            self.rew_sum[index] = entry_rew
            if entry_rew.item() < self.min_sum_reward:
                self.min_sum_reward = entry_rew
        else:
            if entry_rew.item() < self.min_sum_reward:
                return -1
            if len(self.empty_indices) == 0:
                if self.flags.dist_direction == "lim_inf":
                    min_dist = 0
                elif self.flags.dist_direction == "lim_zero":
                    min_dist = sys.maxsize
                else:
                    raise Exception("Unknown distance calculation strategy selected")
                index = -1
                for i in range(self.flags.elite_set_size):
                    distance = torch.dist(torch.flatten(feature_vecs[i]),
                                          torch.flatten(kwargs['feature_vec']), p=self.flags.p)
                    if self.flags.dist_direction == "lim_inf":
                        if distance.item() > min_dist:
                            min_dist = distance.item()
                            index = i
                    elif self.flags.dist_direction == "lim_zero":
                        if distance.item() < min_dist:
                            min_dist = distance.item()
                            index = i
                    else:
                        raise Exception("Unknown distance calculation strategy selected")

            else:
                index = self.empty_indices[len(self.empty_indices) - 1]

            if entry_rew < self.rew_sum[index]:
                return -1

            if self.rew_sum[index] == self.min_sum_reward:
                self.rew_sum[index] = entry_rew
                self.min_sum_reward = min(self.rew_sum)
            else:
                self.rew_sum[index] = entry_rew

            if len(self.empty_indices) > 0:
                self.empty_indices.pop(len(self.empty_indices) - 1)

        self.index_time_array[index] = self.flags.sample_life
        return index

    def before_sampling(self):
        if self.flags.drop_old_samples:
            for i in range(self.flags.elite_set_size):
                self.index_time_array[i] -= 1
                if self.index_time_array[i] == 0:
                    self.empty_indices.append(i)

