from rollout_storage.intefaces.worker_buffer_base import WorkerBufferBase
from option_flags import flags


class NativeWorkerBuffer(WorkerBufferBase):
    def __init__(self):
        super().__init__()
        self.states = [None] * flags.r_f_steps + 1
        self.rewards = [None] * flags.r_f_steps
        self.actions = [None] * flags.r_f_steps
        self.logits = [None] * flags.r_f_steps
        self.values = [None] * flags.r_f_steps
        self.not_done = [bool] * flags.r_f_steps
        self.feature_vec = [None] * flags.r_f_steps
