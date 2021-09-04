import torch

from rollout_storage.intefaces.worker_buffer_base import WorkerBufferBase


class TorchWorkerBuffer(WorkerBufferBase):
    def __init__(self, r_f_steps, env_state_dim, action_count, feature_vec_dim):
        super().__init__(r_f_steps)
        self.states = torch.zeros(r_f_steps + 1, *env_state_dim)
        self.rewards = torch.zeros(r_f_steps)
        self.actions = torch.zeros(r_f_steps)
        self.logits = torch.zeros(r_f_steps, action_count)
        self.values = torch.zeros(r_f_steps)
        self.not_done = torch.zeros(r_f_steps)
        self.feature_vec = torch.zeros(r_f_steps, *feature_vec_dim)


