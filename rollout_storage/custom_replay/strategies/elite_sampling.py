import torch, sys
import numpy as np

from utils.compression import decompress
from rollout_storage.intefaces.custom_sample_strategy import CustomSampleStrategy


class EliteSampleStrategy(CustomSampleStrategy):
    def __init__(self, flags, alfa, lambda_batch_multiplier, strategy, dist_function, p):
        super().__init__(flags)

        self.lambda_batch_multiplier = lambda_batch_multiplier
        self.batch_annealing_factor = (lambda_batch_multiplier - 1) / (self.flags.lr_scheduler_steps * alfa)

        self.p = p
        self.strategy = strategy
        self.dist_function = dist_function
        if dist_function == "cos_dist":
            self.cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    def sample(self, feature_vecs, batch_size, sampling_ratio, model, replay_buffer, sample_indices_threshold, local_random, device, last_vec_index, sampling_counter):
        with torch.no_grad():

            self.lambda_batch_multiplier -= self.batch_annealing_factor

            number_of_samples = int(batch_size * sampling_ratio)

            if self.lambda_batch_multiplier < 1:
                self.lambda_batch_multiplier = 1

            if local_random is None:
                rand_bucket = np.random.choice(sample_indices_threshold, int(self.lambda_batch_multiplier * number_of_samples))
            else:
                rand_bucket = local_random.choice(sample_indices_threshold, int(self.lambda_batch_multiplier * number_of_samples))

            if self.lambda_batch_multiplier == 1:
                return rand_bucket

            if self.flags.use_replay_compression:
                mini_replay_buffer = [decompress(replay_buffer[index]) for index in rand_bucket]
            else:
                mini_replay_buffer = [replay_buffer[index] for index in rand_bucket]

            state = torch.stack([sample.states for sample in mini_replay_buffer])

            mini_batch = self._recalculate_priorities(state, rand_bucket, feature_vecs=feature_vecs, model=model, buffer=replay_buffer, device=device, last_vec_index=last_vec_index, sampling_counter=sampling_counter)

            mini_batch = sorted(mini_batch.items(), key=lambda item:item[1][0][0])

            sample_counter = 0
            samples_per_mini_batch = int(len(mini_batch) / number_of_samples)
            indices = []

            if self.strategy == "strategy1":
                for index, item in mini_batch:
                    for i in range(len(item)):
                        indices.append(index)
                        sample_counter += 1
                        if sample_counter == number_of_samples:
                            break
                    if sample_counter == number_of_samples:
                        break
            elif self.strategy == "strategy2":
                i = 0
                for index, item in mini_batch:
                    if i % samples_per_mini_batch == 0:
                        indices.append(index)
                        sample_counter += 1
                        if sample_counter == number_of_samples:
                            break
                    i += 1
            elif self.strategy == "strategy3":
                j = 1
                while sample_counter < number_of_samples:
                    for index, item in mini_batch:
                        if item[0][1] == j:
                            indices.append(index)
                            sample_counter += 1
                            if sample_counter == number_of_samples:
                                break
                    j += 1
            elif self.strategy == "strategy4":
                i = 0
                lowest_value = sys.maxsize
                lowest_idx = -1
                for index, item in mini_batch:
                    if lowest_value > item[0][1]:
                        lowest_value = item[0][1]
                        lowest_idx = index
                    if i % samples_per_mini_batch == (samples_per_mini_batch - 1):
                        indices.append(lowest_idx)
                        sample_counter += 1
                        lowest_value = sys.maxsize
                        lowest_idx = -1
                        if sample_counter == number_of_samples:
                            break
                    i += 1
            else:
                from agent.learner_d.builder.learner_builder import ForbiddenSetting
                raise ForbiddenSetting("Unknown elite sampling strategy used")

        return indices

    def _recalculate_priorities(self, states, rand_bucket, **kwargs):
        tg_logit, _, new_feature_vecs = kwargs['model'](states.to(kwargs['device']), features=True)
        new_feature_vecs = new_feature_vecs[:, :-1].cpu()

        mini_batch = {}
        i = 0
        for index in rand_bucket:
            if self.dist_function == "ln_norm":
                distance = torch.dist(new_feature_vecs[i], kwargs['feature_vecs'][index], p=self.p)
            elif self.dist_function == "cos_dist":
                distance = self.cos(torch.flatten(new_feature_vecs[i]), torch.flatten(kwargs['feature_vecs'][index]))
            elif self.dist_function == "kl_div":
                beh_dist = torch.distributions.Categorical(logits=kwargs['feature_vecs'][index])
                tar_dist = torch.distributions.Categorical(logits=new_feature_vecs[i])
                distance = torch.distributions.kl.kl_divergence(tar_dist, beh_dist).sum()
            else:
                from agent.learner_d.builder.learner_builder import ForbiddenSetting
                raise ForbiddenSetting("Unknown dist function used")
            
            if index in mini_batch.keys():
                mini_batch[index].append([distance, kwargs['sampling_counter'][index]])
            else:
                mini_batch[index] = [[distance, kwargs['sampling_counter'][index]]]
            i += 1

        return mini_batch
