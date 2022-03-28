import torch
import numpy as np

from utils.compression import decompress
from rollout_storage.intefaces.custom_insert_strategy import CustomInputStrategy


class EliteInsertStrategy(CustomInputStrategy):

    def __init__(self, flags, elite_batch_size, dist_function, p):
        super().__init__(flags)
        self.elite_batch_size = elite_batch_size
        self.p = p
        self.dist_function = dist_function
        if dist_function == "cos_dist":
            self.cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    def calc_index(self, model, buffer, sample_indices_threshold, device, feature_vecs, local_random, **kwargs):
        with torch.no_grad():

            if local_random is None:
                rand_bucket = np.random.choice(sample_indices_threshold, self.elite_batch_size, replace=False)
            else:
                rand_bucket = local_random.choice(sample_indices_threshold, self.elite_batch_size, replace=False)

            if self.flags.use_replay_compression:
                mini_buffer = [decompress(buffer[index]) for index in rand_bucket]
            else:
                mini_buffer = [buffer[index] for index in rand_bucket]

            state = torch.stack([sample.states for sample in mini_buffer])

            _, _, new_feature_vecs = model(state.to(device), features=True)
            new_feature_vecs = new_feature_vecs[:, :-1].cpu()

            elite_batch = {}
            i = 0
            for index in rand_bucket:
                if self.dist_function == "ln_norm":
                    distance = torch.dist(feature_vecs[index], new_feature_vecs[i], p=self.p)
                elif self.dist_function == "cos_dist":
                    distance = self.cos(torch.flatten(feature_vecs[index]),
                                        torch.flatten(new_feature_vecs[i]))
                elif self.dist_function == "kl_div":
                    beh_dist = torch.distributions.Categorical(logits=feature_vecs[index])
                    tar_dist = torch.distributions.Categorical(logits=new_feature_vecs[i])
                    distance = torch.distributions.kl.kl_divergence(tar_dist, beh_dist).sum()
                else:
                    from agent.learner_d.builder.learner_builder import ForbiddenSetting
                    raise ForbiddenSetting("Unknown dist function used")

                elite_batch[index] = [distance.item()]
                i += 1

            mini_batch = sorted(elite_batch.items(), key=lambda item: item[1], reverse=True)
            for index, item in mini_batch:
                return index

