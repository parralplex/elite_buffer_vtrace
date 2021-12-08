import unittest
import os
import torch
import numpy as np
import random
import torch.backends.cudnn
from option_flags import flags, change_args

from agent.algorithms.v_trace import v_trace


class VTraceTest(unittest.TestCase):

    def setUp(self):
        self.init_data_pack = torch.load("v_trace_loss_result.pt")

        self.actions = self.init_data_pack["actions"]
        self.logits = self.init_data_pack["logits"]
        self.target_logits = self.init_data_pack["target_logits"]
        self.target_values = self.init_data_pack["target_values"]
        self.not_done = self.init_data_pack["not_done"]
        self.rewards = self.init_data_pack["rewards"]
        self.bootstrap_value = self.init_data_pack["bootstrap_value"]

        self.flags = change_args(r_f_steps=10, c_const=1, rho_const=1, baseline_loss_coef=0.5, entropy_loss_coef=0.01,
                                 gamma=0.99)

    def test_loss_computation(self):
        seed = 1

        # SEEDING cannot be done in setup method - IT does not work properly
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if float(torch.__version__[0: 3]) >= 1.8:
            torch.use_deterministic_algorithms(True)
        else:
            torch.set_deterministic(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        baseline_loss, entropy_loss, policy_loss = v_trace(self.actions, self.logits, self.bootstrap_value, self.target_logits,
                                                           self.target_values, self.not_done, self.rewards, self.flags)
        total_loss = baseline_loss + entropy_loss + policy_loss
        self.assertEqual(total_loss.item(), self.init_data_pack["total_loss"].item(), 'total loss should be the same')
