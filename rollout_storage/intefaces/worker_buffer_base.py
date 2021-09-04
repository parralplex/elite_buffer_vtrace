
class WorkerBufferBase(object):
    def __init__(self, r_f_steps):
        self.pos_pointer = 0
        self.r_f_steps = r_f_steps
        self.states = None
        self.rewards = None
        self.actions = None
        self.logits = None
        self.values = None
        self.not_done = None
        self.feature_vec = None

    def insert(self, state, action, reward, logits, not_done, feature_vec):
        self.states[self.pos_pointer + 1] = state
        self.actions[self.pos_pointer] = action
        self.rewards[self.pos_pointer] = reward
        self.logits[self.pos_pointer] = logits
        self.not_done[self.pos_pointer] = not_done
        self.feature_vec[self.pos_pointer] = feature_vec

        self.pos_pointer = (self.pos_pointer + 1) % self.r_f_steps

    def reset(self):
        self.pos_pointer = 0
