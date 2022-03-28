'''
Modified attentive experience replay - it doesnt work with single step transitions but with n-steps. So it cannot simply compare
the last state visited by workers. To calculate n-steps transition priority is calculates cross distance between states.
https://ojs.aaai.org/index.php/AAAI/article/view/6049
'''

import torch

from utils.compression import decompress

from rollout_storage.custom_replay.strategies.elite_sampling import EliteSampleStrategy


class AttentiveSampleStrategy(EliteSampleStrategy):
    def __init__(self, flags, alfa, lambda_batch_multiplier, strategy, dist_function, p):
        super().__init__(flags, alfa, lambda_batch_multiplier, strategy, dist_function, p)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _recalculate_priorities(self, states, rand_bucket, **kwargs):
        _, _, new_feature_vecs = kwargs['model'](states.to(kwargs['device']), features=True)
        new_feature_vecs = new_feature_vecs[:, :-1].cpu()

        if self.flags.use_replay_compression:
            last_vec_state = decompress(kwargs['buffer'][kwargs['last_vec_index']]).states
        else:
            last_vec_state = kwargs['buffer'][kwargs['last_vec_index']].states

        _, _, last_vec_features = kwargs['model'](last_vec_state.to(kwargs['device']), features=True)
        last_vec_features = last_vec_features[:-1].cpu()

        last_state_feature_vec = []
        for i in range(self.flags.r_f_steps):
            last_state_feature_vec.append(last_vec_features[-1].unsqueeze(0).repeat(self.flags.r_f_steps, 1))
        last_state_feature_vec = torch.stack(last_state_feature_vec)

        mini_batch = {}

        for i in range(len(rand_bucket)):
            if self.dist_function == "ln_norm":
                distance = torch.dist(new_feature_vecs[i], last_state_feature_vec, p=self.p)
            elif self.dist_function == "cos_dist":
                distance = self.cos(torch.flatten(new_feature_vecs[i]), torch.flatten(last_state_feature_vec))
            elif self.dist_function == "kl_div":
                beh_dist = torch.distributions.Categorical(logits=last_state_feature_vec)
                tar_dist = torch.distributions.Categorical(logits=new_feature_vecs[i])
                distance = torch.distributions.kl.kl_divergence(tar_dist, beh_dist).sum()
            else:
                from agent.learner_d.builder.learner_builder import ForbiddenSetting
                raise ForbiddenSetting("Unknown dist function used")
            index = rand_bucket[i]
            if index in mini_batch.keys():
                mini_batch[index].append([distance, kwargs['sampling_counter'][index]])
            else:
                mini_batch[index] = [[distance, kwargs['sampling_counter'][index]]]

        return mini_batch

