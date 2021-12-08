import os

import torch
import numpy as np
from rollout_storage.experience_replay import ExperienceReplayTorch
from rollout_storage.intefaces.elite_set_insert_strategy import EliteSetInputStrategy
from rollout_storage.intefaces.elite_set_sample_strategy import EliteSetSampleStrategy
from stats.prof_timer import Timer


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# import matplotlib.pyplot as plt
import torch.multiprocessing as mp


class EliteSetReplay(ExperienceReplayTorch):
    def __init__(self, feature_vec_dim, insert_strategy: EliteSetInputStrategy, sample_strategy: EliteSetSampleStrategy, flags, file_save_dir_url, training_event):
        super().__init__(flags, training_event)

        self.feature_vecs = torch.zeros(self.flags.elite_set_size, self.flags.r_f_steps, *feature_vec_dim)

        self.insert_strategy = insert_strategy
        self.sample_strategy = sample_strategy
        self.file_save_dir_url = file_save_dir_url
        self.calc_index_timer = Timer("Insert strategy avg. index calculation time:")

        self.feature_vec_dim = feature_vec_dim
        self.last_vec_index = 0
        self.replay_data_ratio = self.flags.elite_set_data_ratio

        self.buffer = [None for i in range(self.flags.elite_set_size)]

    def calc_index(self, **kwargs):
        if self.insert_strategy is not None:
            filled = self.filled
            index = -1
            if not self.filled:
                index = super(EliteSetReplay, self).calc_index(**kwargs)
            index = self.insert_strategy.calc_index(index, filled, self.feature_vecs, **kwargs)
        else:
            index = super(EliteSetReplay, self).calc_index(**kwargs)
        self.last_vec_index = index

        return index

    def _store(self, index, **kwargs):
        super(EliteSetReplay, self)._store(index, **kwargs)
        self.feature_vecs[index] = kwargs['feature_vec']

    def sample(self, batch_size, local_random=None):
        if self.insert_strategy is not None:
            self.insert_strategy.before_sampling()

        if self.sample_strategy is not None:
            indices = self.sample_strategy.sample(batch_size, self.replay_data_ratio, self.last_vec_index, self.feature_vecs)
        else:
            if local_random is None:
                indices = np.random.choice(self.flags.elite_set_size, int(batch_size * self.replay_data_ratio))
            else:
                indices = local_random.choice(self.flags.elite_set_size, int(batch_size * self.replay_data_ratio))

        batch_data = [self.buffer[k] for k in indices]
        return batch_data

    def close(self):
        # pca_features = PCA(n_components=50).fit_transform(torch.flatten(self.feature_vecs, start_dim=1))
        # _2d_projection_data = TSNE(n_components=2).fit_transform(pca_features)
        # process = mp.Process(target=safe_buff_plot, args=(self.file_save_dir_url, _2d_projection_data))
        # process.start()
        # process.join()
        torch.save(self.feature_vecs, self.file_save_dir_url + '/elite_features_end.pt')
        self.calc_index_timer.save_stats(self.file_save_dir_url)
        super(EliteSetReplay, self).close()

    def reset(self):
        super(EliteSetReplay, self).reset()


def safe_buff_plot(file_save_dir_url, buffer_2d_projection):
    # plt.clf()
    # plt.scatter(buffer_2d_projection[:, 0], buffer_2d_projection[:, 1])
    # if not os.path.exists(file_save_dir_url + "/Charts"):
    #     os.mkdir(file_save_dir_url + "/Charts")
    # plt.savefig(file_save_dir_url + "/Charts/elite_set_feature_dist.png")
    pass

