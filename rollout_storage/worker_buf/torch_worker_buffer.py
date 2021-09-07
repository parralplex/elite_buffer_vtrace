import torch

from rollout_storage.intefaces.worker_buffer_base import WorkerBufferBase
from option_flags import flags


class TorchWorkerBuffer(WorkerBufferBase):
    def __init__(self, feature_vec_dim):
        super().__init__(flags.r_f_steps)
        self.states = torch.zeros(flags.r_f_steps + 1, *flags.observation_shape)
        self.rewards = torch.zeros(flags.r_f_steps)
        self.actions = torch.zeros(flags.r_f_steps)
        self.logits = torch.zeros(flags.r_f_steps, flags.actions_count)
        self.values = torch.zeros(flags.r_f_steps)
        self.not_done = torch.zeros(flags.r_f_steps)
        self.feature_vec = torch.zeros(flags.r_f_steps, *feature_vec_dim)


