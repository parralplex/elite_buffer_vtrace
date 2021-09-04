from rollout_storage.intefaces.worker_buffer_base import WorkerBufferBase


class NativeWorkerBuffer(WorkerBufferBase):
    def __init__(self, r_f_steps):
        super().__init__(r_f_steps)
        self.states = [None] * r_f_steps + 1
        self.rewards = [None] * r_f_steps
        self.actions = [None] * r_f_steps
        self.logits = [None] * r_f_steps
        self.values = [None] * r_f_steps
        self.not_done = [bool] * r_f_steps
        self.feature_vec = [None] * r_f_steps
