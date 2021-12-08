from rollout_storage.intefaces.worker_buffer_base import WorkerBufferBase


class NativeWorkerBuffer(WorkerBufferBase):
    def __init__(self, flags):
        super().__init__(flags)
        self.states = [None] * self.flags.r_f_steps + 1
        self.rewards = [None] * self.flags.r_f_steps
        self.actions = [None] * self.flags.r_f_steps
        self.logits = [None] * self.flags.r_f_steps
        self.values = [None] * self.flags.r_f_steps
        self.not_done = [bool] * self.flags.r_f_steps
        self.feature_vec = [None] * self.flags.r_f_steps

