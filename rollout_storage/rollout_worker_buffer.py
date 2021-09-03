import torch


class RolloutWorkerBuffer(object):
    def __init__(self, r_f_steps, env_state_dim, action_count, feature_vec_dim):
        self.states = torch.zeros(r_f_steps + 1, *env_state_dim)
        self.rewards = torch.zeros(r_f_steps)
        self.actions = torch.zeros(r_f_steps)
        self.logits = torch.zeros(r_f_steps, action_count)
        self.values = torch.zeros(r_f_steps)
        self.not_done = torch.zeros(r_f_steps)
        self.feature_vec = torch.zeros(r_f_steps, *feature_vec_dim)

        self.step = 0
        self.r_f_steps = r_f_steps

    def insert(self, state, action, reward, logits, done, feature_vec):
        self.states[self.step + 1] = state
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.logits[self.step] = logits
        self.not_done[self.step] = not done
        self.feature_vec[self.step] = feature_vec

        self.step = (self.step + 1) % self.r_f_steps

    def reset(self):
        self.step = 0


