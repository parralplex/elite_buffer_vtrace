import queue

import torch
import numpy as np
import signal
from utils.logger import logger

from model.network import ModelNetwork
from rollout_storage.experience_replay import ExperienceReplayTorch
from rollout_storage.intefaces.custom_insert_strategy import CustomInputStrategy
from rollout_storage.intefaces.custom_sample_strategy import CustomSampleStrategy
from stats.prof_timer import Timer


class CustomReplay(ExperienceReplayTorch):
    def __init__(self, insert_strategy: CustomInputStrategy, sample_strategy: CustomSampleStrategy, flags, file_save_dir_url, training_event, train_model, device, replay_dict):
        super().__init__(flags, training_event, replay_dict)

        self.feature_vecs = torch.zeros(self.replay_capacity, self.flags.r_f_steps, *(self.flags.feature_out_layer_size,))

        self.insert_strategy = insert_strategy
        self.sample_strategy = sample_strategy
        self.file_save_dir_url = file_save_dir_url
        self.calc_index_timer = Timer("Insert strategy avg. index calculation time:")
        self.feature_vec_dim = (self.flags.feature_out_layer_size,)
        self.last_vec_index = 0
        self.device = device

        self.train_model = ModelNetwork(self.flags.actions_count, self.flags.frames_stacked, self.flags.feature_out_layer_size, self.flags.use_additional_scaling_FC_layer).to(device)
        self.train_model.load_state_dict(train_model.state_dict())
        self.local_random = np.random.RandomState(self.flags.seed)
        self.sampling_counter = [1 for i in range(self.replay_capacity)]

    def calc_index(self, **kwargs):
        if self.insert_strategy is not None:
            if not self.filled:
                index = super(CustomReplay, self).calc_index(**kwargs)
            else:
                allowed_values = self.replay_capacity
                index = self.insert_strategy.calc_index(self.train_model, self.buffer,  allowed_values, self.device, self.feature_vecs, self.local_random, **kwargs)
        else:
            index = super(CustomReplay, self).calc_index(**kwargs)

        self.last_vec_index = index
        return index

    def _store(self, index, **kwargs):
        super(CustomReplay, self)._store(index, **kwargs)
        self.feature_vecs[index] = kwargs['feature_vec']
        self.sampling_counter[index] = 1

    def sample(self, batch_size, local_random=None):
        try:
            if not self.training_started:
                self.training_event.wait()
                self.training_started = True

            if self.sample_strategy is not None:
                if self.filled:
                    allowed_values = self.replay_capacity
                else:
                    allowed_values = self.position_pointer
                indices = self.sample_strategy.sample(self.feature_vecs, batch_size, self.replay_sample_ratio, self.train_model, self.buffer, allowed_values, local_random, self.device, self.last_vec_index, self.sampling_counter)
            else:

                if self.filled:
                    allowed_values = self.replay_capacity
                else:
                    allowed_values = self.position_pointer

                if local_random is None:
                    indices = np.random.choice(allowed_values, int(batch_size * self.replay_sample_ratio))
                else:
                    indices = local_random.choice(allowed_values, int(batch_size * self.replay_sample_ratio))

            batch_data = [self.buffer[k] for k in indices]

            for k in indices:
                self.sampling_counter[k] += 1

        except Exception as exp:
            logger.exception("Replay buffer sampling has raised an exception:" + str(exp))
            signal.raise_signal(signal.SIGINT)
            return []

        return batch_data

    def close(self):
        #torch.save(self.feature_vecs, self.file_save_dir_url + '/elite_features_end.pt')
        self.calc_index_timer.save_stats(self.file_save_dir_url)
        super(CustomReplay, self).close()

    def reset(self):
        super(CustomReplay, self).reset()

