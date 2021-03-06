
class WorkerBufferBase(object):
    def __init__(self, flags):
        self.flags = flags
        self.pos_pointer = 0
        self.states = None
        self.rewards = None
        self.actions = None
        self.logits = None
        self.not_done = None
        self.feature_vec = None
        self.values = None

    def insert(self, state, action, reward, logits, not_done, value, feature_vec):
        self.states[self.pos_pointer + 1] = state
        self.actions[self.pos_pointer] = action
        self.rewards[self.pos_pointer] = reward
        self.logits[self.pos_pointer] = logits
        self.not_done[self.pos_pointer] = not_done
        self.values[self.pos_pointer] = value

        self.feature_vec[self.pos_pointer] = feature_vec

        self.pos_pointer = (self.pos_pointer + 1) % self.flags.r_f_steps

    def reset(self):
        self.pos_pointer = 0
