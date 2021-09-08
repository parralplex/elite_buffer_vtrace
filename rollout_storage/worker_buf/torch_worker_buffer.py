import torch

from rollout_storage.intefaces.worker_buffer_base import WorkerBufferBase


class TorchWorkerBuffer(WorkerBufferBase):
    def __init__(self, feature_vec_dim, flags):
        super().__init__(flags)
        self.states = torch.zeros(self.flags.r_f_steps + 1, *self.flags.observation_shape)
        self.rewards = torch.zeros(self.flags.r_f_steps)
        self.actions = torch.zeros(self.flags.r_f_steps)
        self.logits = torch.zeros(self.flags.r_f_steps, self.flags.actions_count)
        self.values = torch.zeros(self.flags.r_f_steps)
        self.not_done = torch.zeros(self.flags.r_f_steps)
        self.feature_vec = torch.zeros(self.flags.r_f_steps, *feature_vec_dim)


