import numpy as np
import torch
import sys
from utils import decompress


class ExperienceReplay:
    def __init__(self, action_count, feature_vec_dim, option_flags, state_dim):
        self.index = 0
        self.size = option_flags.buffer_size
        self.r_f_steps = option_flags.r_f_steps
        self.prior_buf_size = option_flags.elite_set_size
        self.action_count = action_count
        self.feature_vec_dim = feature_vec_dim
        self.option_flags = option_flags
        self.state_dim = state_dim

        self.not_used = True
        self.not_used_prior = True
        self.filled = False
        self.prior_buf_filled = False
        self.index = 0
        self.min_sum_reward = sys.maxsize
        self.index_prior = 0
        self.states = []
        self.actions = torch.zeros(self.size, self.r_f_steps)
        self.rewards = torch.zeros(self.size, self.r_f_steps)
        self.logits = torch.zeros(self.size, self.r_f_steps, action_count)
        self.not_done = torch.zeros(self.size, self.r_f_steps)

        self.states_prior = []
        self.actions_prior = torch.zeros(self.prior_buf_size, self.r_f_steps)
        self.rewards_prior = torch.zeros(self.prior_buf_size, self.r_f_steps)
        self.logits_prior = torch.zeros(self.prior_buf_size, self.r_f_steps, action_count)
        self.not_done_prior = torch.zeros(self.prior_buf_size, self.r_f_steps)

        self.rew_sum = torch.zeros(self.prior_buf_size)
        self.feature_vecs_prior = torch.zeros(self.prior_buf_size, self.r_f_steps, *feature_vec_dim)
        self.prior_start_states = []
        self.start_states = []

        self.replay_filled_event = None
        self.replay_prior_filled_event = None

    def _store_replay(self, state, action, reward, logits, not_done):
        if not self.not_used and (self.index % self.size) == 0:
            self.filled = True
            self.replay_filled_event.set()

        self.index = self.index % self.size

        if self.not_used:
            self.not_used = False

        if len(self.states) > self.index:
            self.states[self.index] = state
        else:
            self.states.append(state)
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.logits[self.index] = logits
        self.not_done[self.index] = not_done

        self.index += 1

    def _store_replay_prior(self, state, action, reward, logits, not_done, feature_vec):
        if not self.not_used_prior and (self.index_prior % self.prior_buf_size) == 0:
            self.prior_buf_filled = True
            self.replay_prior_filled_event.set()

        self.index_prior = self.index_prior % self.prior_buf_size

        if self.not_used_prior:
            self.not_used_prior = False

        if self.prior_buf_filled:
            entry_rew = torch.sum(reward)
            if entry_rew > self.min_sum_reward:
                closest_entry_idx = 0
                closest_entry_distance = sys.maxsize
                update_entry = True
                for i in range(self.prior_buf_size):
                    distance = abs(torch.sum(self.feature_vecs_prior[i] - feature_vec))
                    if distance == 0:
                        update_entry = False
                        break
                    elif distance < closest_entry_distance:
                        closest_entry_idx = i
                        closest_entry_distance = distance
                if entry_rew < torch.sum(self.rewards_prior[closest_entry_idx]):
                    update_entry = False
                if update_entry:
                    update_min_rew = False
                    if self.rew_sum[closest_entry_idx] == self.min_sum_reward:
                        update_min_rew = True
                    if len(self.states_prior) > closest_entry_idx:
                        self.states_prior[closest_entry_idx] = state
                    else:
                        self.states_prior.append(state)
                    self.actions_prior[closest_entry_idx] = action
                    self.rewards_prior[closest_entry_idx] = reward
                    self.logits_prior[closest_entry_idx] = logits
                    self.not_done_prior[closest_entry_idx] = not_done
                    self.feature_vecs_prior[closest_entry_idx] = feature_vec
                    self.rew_sum[closest_entry_idx] = entry_rew
                    if update_min_rew:
                        self.min_sum_reward = sys.maxsize
                        for i in range(self.prior_buf_size):
                            rew = self.rew_sum[i]
                            if rew < self.min_sum_reward:
                                self.min_sum_reward = rew
        else:
            entry_rew = torch.sum(reward)
            if entry_rew < self.min_sum_reward:
                self.min_sum_reward = entry_rew

            if len(self.states_prior) > self.index_prior:
                self.states_prior[self.index_prior] = state
            else:
                self.states_prior.append(state)
            self.actions_prior[self.index_prior] = action
            self.rewards_prior[self.index_prior] = reward
            self.logits_prior[self.index_prior] = logits
            self.not_done_prior[self.index_prior] = not_done
            self.feature_vecs_prior[self.index_prior] = feature_vec
            self.rew_sum[self.index_prior] = torch.sum(reward)

            self.index_prior += 1

    def calculate_replay_prior_pos(self, reward, feature_vec):
        entry_rew = torch.sum(reward)
        update_min_rew = False
        closest_entry_idx = -1
        update_entry = False
        if entry_rew > self.min_sum_reward:
            closest_entry_idx = 0
            closest_entry_distance = sys.maxsize
            update_entry = True
            for i in range(self.prior_buf_size):
                distance = abs(torch.sum(self.feature_vecs_prior[i] - feature_vec))
                if distance == 0:
                    update_entry = False
                    break
                elif distance < closest_entry_distance:
                    closest_entry_idx = i
                    closest_entry_distance = distance
            if entry_rew < torch.sum(self.rewards_prior[closest_entry_idx]):
                update_entry = False
            if update_entry:
                update_min_rew = False
                if self.rew_sum[closest_entry_idx] == self.min_sum_reward:
                    update_min_rew = True

        if not update_entry:
            closest_entry_idx = -1
        return [closest_entry_idx, update_min_rew, entry_rew]

    def store(self, state, action, reward, logits, not_done, feature_vec, batch_lock):
        with batch_lock:
            self._store_replay(state, action, reward, logits, not_done)
            # self._store_replay_prior(state, action, reward, logits, not_done, feature_vec)

    def sample(self, batch_size):
        with torch.no_grad():
            batch = np.random.choice(self.size, int(batch_size * self.option_flags.replay_data_ratio))

            states = []
            for k in batch:
                states.append(decompress(self.states[k]))
            states = torch.stack(states)
            actions = self.actions[batch].long()
            rewards = self.rewards[batch]
            logits = self.logits[batch]
            not_done = self.not_done[batch]


            # if self.prior_buf_filled:
            #     length = self.prior_buf_size
            # else:
            #     length = min(self.prior_buf_size, self.index_prior)
            #
            # batch = np.random.choice(length, int(batch_size * self.option_flags.elite_set_data_ratio))
            #
            #
            # states_prior = []
            # for k in batch:
            #     states_prior.append(decompress(self.states_prior[k]))
            # states_prior = torch.stack(states_prior)
            # actions_prior = self.actions_prior[batch].long()
            # rewards_prior = self.rewards_prior[batch]
            # logits_prior = self.logits_prior[batch]
            # not_done_prior = self.not_done_prior[batch]
            #
            # states = torch.cat((states, states_prior), 0)
            # actions = torch.cat((actions, actions_prior), 0)
            # rewards = torch.cat((rewards, rewards_prior), 0)
            # logits = torch.cat((logits, logits_prior), 0)
            # not_done = torch.cat((not_done, not_done_prior), 0)

            return states.transpose(1, 0), actions.transpose(1, 0), rewards.transpose(1, 0), logits.transpose(1, 0), not_done.transpose(1, 0)

    def on_policy_sample(self, batch_size):
        with torch.no_grad():

            indices = []
            for i in range(self.index, self.index - batch_size, -1):
                if i < 0:
                    indices.append(i + self.size)
                else:
                    indices.append(i % self.size)

            states = []
            for k in indices:
                states.append(decompress(self.states[k]))
            states = torch.stack(states)
            actions = self.actions[indices].long()
            rewards = self.rewards[indices]
            logits = self.logits[indices]
            not_done = self.not_done[indices]

            return states.transpose(1, 0), actions.transpose(1, 0), rewards.transpose(1, 0), logits.transpose(1, 0), not_done.transpose(1, 0)

    def sample_base(self, batch_size):
        if self.filled:
            length = self.size
        else:
            length = min(self.size, self.index)

        batch = np.random.choice(length, int(batch_size * self.option_flags.replay_data_ratio))

        states = []
        for k in batch:
            states.append(decompress(self.states[k]))
        states = torch.stack(states)
        actions = self.actions[batch].long()
        rewards = self.rewards[batch]
        logits = self.logits[batch]
        not_done = self.not_done[batch]

        return states.transpose(1, 0).detach(), actions.transpose(1, 0).detach(), rewards.transpose(1, 0).detach(), logits.transpose(1, 0).detach(), not_done.transpose(1, 0).detach()

    def get_prior_buf_states(self):
        prior_states = torch.zeros(self.prior_buf_size, self.r_f_steps, *self.state_dim)
        for i in range(len(self.states_prior)):
            prior_states[i] = decompress(self.states_prior[i])[:-1]
        return prior_states

    def set_feature_vecs_prior(self, feature_vecs_prior):
        self.feature_vecs_prior = feature_vecs_prior

    def set_filled_events(self, replay_filled_event, replay_prior_filled_event):
        self.replay_filled_event = replay_filled_event
        self.replay_prior_filled_event = replay_prior_filled_event





