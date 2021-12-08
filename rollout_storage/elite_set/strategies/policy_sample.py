import torch
from rollout_storage.intefaces.elite_set_sample_strategy import EliteSetSampleStrategy


class PolicySampleStrategy(EliteSetSampleStrategy):
    def __init__(self, flags):
        super().__init__(flags)
        self.cache_last_vec_index = -1
        self.cache_indices = []

    def sample(self, batch_size, replay_ratio, last_vec_index, feature_vecs):
        vecs = {}
        if self.cache_last_vec_index == last_vec_index:
            indices = self.cache_indices
        else:
            for i in range(self.flags.elite_set_size):
                if i != last_vec_index:
                    distance = torch.dist(torch.flatten(feature_vecs[i]),
                                          torch.flatten(feature_vecs[last_vec_index]), p=self.flags.p)
                    vecs[distance.item()] = i

            sorted_keys = sorted(vecs.keys())
            if self.flags.sample_dist_direction == "lim_zero":
                indices = [vecs[sorted_keys[i]] for i in range(int(batch_size * replay_ratio))]
            elif self.flags.sample_dist_direction == "lim_inf":
                indices = [vecs[sorted_keys[(len(sorted_keys) - i - 1)]] for i in
                           range(int(batch_size * replay_ratio))]
            else:
                raise Exception("Unknown distance calculation strategy selected")
            self.cache_indices = indices
            self.cache_last_vec_index = last_vec_index
        return indices
