import random
import torch

from rollout_storage.intefaces.elite_set_insert_strategy import EliteSetInsertStrategy
from rollout_storage.elite_set.decoratos.rand_ouput import rand_output
from option_flags import flags


class BruteForceStrategy(EliteSetInsertStrategy):
    def __init__(self):
        super().__init__()
        self.buf_distances_sum = 0
        self.buf_distances = []

        self.new_tr_dis_buf = []
        self.new_tr_dis_sum = 0
        self.swap_index = -1

    @rand_output(chance=flags.rnd_index_chance)
    def calculate_best_index_pos(self, feature_vecs, new_feature_vec, new_reward, **kwargs):
        entry_rew = torch.sum(new_reward)

        self.new_tr_dis_buf = []
        self.new_tr_dis_sum = 0
        self.swap_index = -1

        if kwargs["add_rew_feature"]:
            feature_vec_rewarded = new_feature_vec + entry_rew
            super(BruteForceStrategy, self).calculate_best_index_pos(feature_vecs, feature_vec_rewarded, new_reward,
                                                                  **kwargs)
        else:
            super(BruteForceStrategy, self).calculate_best_index_pos(feature_vecs, new_feature_vec,
                                                                  new_reward, **kwargs)

        offset = 0
        if kwargs["random_search"]:
            offset = random.randint(0, flags.elite_set_size - 1)
        for i in range(flags.elite_set_size):
            if kwargs["random_search"]:
                index = (i + offset) % flags.elite_set_size
            else:
                index = i
            new_buff_dis_sum = (self.buf_distances_sum - self.buf_distances[index]) + self.new_tr_dis_sum - \
                               self.new_tr_dis_buf[index]
            if self.buf_distances_sum < new_buff_dis_sum:
                self.swap_index = index
                self.buf_distances_sum = new_buff_dis_sum
                self.buf_distances[index] = self.new_tr_dis_sum - self.new_tr_dis_buf[index]
                break

        return self.swap_index

    def process_dist(self, distance, index) -> bool:
        self.new_tr_dis_buf.append(distance)
        self.new_tr_dis_sum += distance
        return True

    def on_insert_before_filled(self, index, **kwargs):
        self.buf_distances.append(0)
        for i in range(index):
            distance = torch.dist(kwargs['feature_vecs'][index], kwargs['feature_vec'], p=kwargs['p'])

            self.buf_distances_sum += distance
            self.buf_distances[i] += distance
            self.buf_distances[index] += distance
