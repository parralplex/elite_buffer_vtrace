import torch
from rollout_storage.experience_replay import ExperienceReplayTorch
from rollout_storage.intefaces.elite_set_insert_strategy import EliteSetInsertStrategy
from utils import decompress


class EliteSetReplay(ExperienceReplayTorch):
    def __init__(self, batch_lock, feature_vec_dim, insert_strategy : EliteSetInsertStrategy, flags):
        super().__init__(batch_lock, flags)

        self.feature_vecs = torch.zeros(self.flags.elite_set_size, self.flags.r_f_steps, *feature_vec_dim)
        self.insert_strategy = insert_strategy

    def calc_index(self, **kwargs):
        if not self.filled:
            index = super(EliteSetReplay, self).calc_index(**kwargs)
            self.insert_strategy.on_insert_before_filled(index, feature_vecs=self.feature_vecs, **kwargs)
            return index
        index = self.insert_strategy.calculate_best_index_pos(feature_vecs=self.feature_vecs, new_feature_vec=kwargs['feature_vec'],
                                                              new_reward=kwargs['reward'], **kwargs)
        if index == -1:
            return index
        return index

    def _store(self, index, **kwargs):
        super(EliteSetReplay, self)._store(index, **kwargs)
        if kwargs["add_rew_feature"]:
            entry_rew = torch.sum(kwargs['reward'])
            self.feature_vecs[index] = kwargs['feature_vec'] + entry_rew
        else:
            self.feature_vecs[index] = kwargs['feature_vec']

    def get_prior_buf_states(self):
        if self.flags.use_state_compression:
            prior_states = torch.zeros(self.flags.elite_set_size, self.flags.r_f_steps, *self.flags.observation_shape)
            for i in range(len(self.states)):
                prior_states[i] = decompress(self.states[i])[:-1]
            return prior_states
        else:
            return self.states

    def set_feature_vecs_prior(self, feature_vecs):
        self.feature_vecs = feature_vecs

