import torch

from rollout_storage.intefaces.worker_buffer_base import WorkerBufferBase


class TorchWorkerBuffer(WorkerBufferBase):
    def __init__(self, feature_vec_dim, flags):
        super(TorchWorkerBuffer, self).__init__(flags)
        self.states = torch.zeros(self.flags.r_f_steps + 1, *self.flags.observation_shape)
        self.rewards = torch.zeros(self.flags.r_f_steps)
        self.actions = torch.zeros(self.flags.r_f_steps)
        self.logits = torch.zeros(self.flags.r_f_steps, self.flags.actions_count)
        self.not_done = torch.zeros(self.flags.r_f_steps)
        self.feature_vec = torch.zeros(self.flags.r_f_steps, *feature_vec_dim)
        self.values = torch.zeros(self.flags.r_f_steps)

    def main_data_copy(self, worker_buffer):
        self.states = worker_buffer.states
        self.rewards = worker_buffer.rewards
        self.actions = worker_buffer.actions
        self.logits = worker_buffer.logits
        self.not_done = worker_buffer.not_done
        self.values = worker_buffer.values
        self.feature_vec = worker_buffer.feature_vec


