import os
import torch
import random
from rollout_storage.experience_replay import ExperienceReplayTorch
from rollout_storage.intefaces.elite_set_insert_strategy import EliteSetInsertStrategy
from stats.prof_timer import Timer
from utils import decompress

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


class EliteSetReplay(ExperienceReplayTorch):
    def __init__(self, batch_lock, feature_vec_dim, insert_strategy : EliteSetInsertStrategy, flags, file_save_dir_url):
        super().__init__(batch_lock, flags)

        if self.flags.use_state_compression:
            self.states = []
        else:
            self.states = torch.zeros(self.flags.elite_set_size, self.flags.r_f_steps, *self.flags.observation_shape)
        self.actions = torch.zeros(self.flags.elite_set_size, self.flags.r_f_steps)
        self.rewards = torch.zeros(self.flags.elite_set_size, self.flags.r_f_steps)
        self.logits = torch.zeros(self.flags.elite_set_size, self.flags.r_f_steps, self.flags.actions_count)
        self.not_done = torch.zeros(self.flags.elite_set_size, self.flags.r_f_steps)

        self.feature_vecs = torch.zeros(self.flags.elite_set_size, self.flags.r_f_steps, *feature_vec_dim)
        self.insert_strategy = insert_strategy
        self.file_save_dir_url = file_save_dir_url
        self.calc_index_timer = Timer("Insert strategy avg. index calculation time:")
        self.index_array = [i for i in range(self.flags.elite_set_size)]

    def calc_index(self, **kwargs):
        if not self.filled:
            index = super(EliteSetReplay, self).calc_index(elite=True, **kwargs)
            self.insert_strategy.on_insert_before_filled(index, feature_vecs=self.feature_vecs, **kwargs)
            return index
        with self.calc_index_timer:
            index = self.insert_strategy.calculate_best_index_pos(feature_vecs=self.feature_vecs, new_feature_vec=kwargs['feature_vec'],
                                                                  new_reward=kwargs['reward'], **kwargs)
        return index

    def _store(self, index, **kwargs):
        super(EliteSetReplay, self)._store(index, **kwargs)
        if kwargs["add_rew_feature"]:
            entry_rew = torch.sum(kwargs['reward'])
            self.feature_vecs[index] = kwargs['feature_vec'] + entry_rew
        else:
            self.feature_vecs[index] = kwargs['feature_vec']

    def get_prior_buf_states(self):
        self.feature_vecs = None
        if self.flags.use_state_compression:
            prior_states = torch.zeros(self.flags.elite_set_size, self.flags.r_f_steps, *self.flags.observation_shape)
            for i in range(len(self.states)):
                prior_states[i] = decompress(self.states[i])[:-1]
            return prior_states
        else:
            return self.states

    def set_feature_vecs_prior(self, feature_vecs):
        self.feature_vecs = feature_vecs
        self.insert_strategy.recalculate_dist(self.feature_vecs)

    def random_sample(self, batch_size, local_random=None):
        if local_random is None:
            indices = random.choices(self.index_array, k=int(batch_size * self.flags.elite_set_data_ratio))
        else:
            indices = local_random.choices(self.index_array, k=int(batch_size * self.flags.elite_set_data_ratio))
        return self._get_batch(indices)

    def close(self):
        pca_features = PCA(n_components=50).fit_transform(torch.flatten(self.feature_vecs, start_dim=1))
        X_embedded = TSNE(n_components=2).fit_transform(pca_features)
        plt.clf()
        plt.plot(X_embedded)
        if not os.path.exists(self.file_save_dir_url + "/Charts"):
            os.mkdir(self.file_save_dir_url + "/Charts")
        plt.savefig(self.file_save_dir_url + "/Charts/elite_set_feature_dist.png")
        torch.save(self.feature_vecs, self.file_save_dir_url + '/elite_features_end.pt')
        self.calc_index_timer.save_stats(self.file_save_dir_url)
        super(EliteSetReplay, self).close()

