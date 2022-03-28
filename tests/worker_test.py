import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import flatten
from model.utils import  num_flat_features, normalized_columns_initializer, weights_init
from option_flags import flags, change_args
from wrappers.atari_wrappers import make_stock_atari
from agent.worker.rollout_worker import RolloutWorker


class ModelNetwork(nn.Module):
    def __init__(self, actions_count, flags):
        super(ModelNetwork, self).__init__()
        self.actions_count = actions_count
        self.flags = flags

        self.conv1 = nn.Conv2d(4, 64, stride=(3, 3), kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(64, 128, stride=(2, 2), kernel_size=(3, 3))

        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, actions_count)

        self.fc3 = nn.Linear(512, 1)

        self.apply(weights_init)
        self.fc2.weight.data = normalized_columns_initializer(
            self.fc2.weight.data, 0.001)
        self.fc2.bias.data.fill_(0)
        self.fc3.weight.data = normalized_columns_initializer(
            self.fc3.weight.data, 1.0)
        self.fc3.bias.data.fill_(0)

        self.train()

    def forward(self, inputs, features=False):
        time_dim = False
        t_dim, batch_dim = 1, 1
        if inputs.ndim == 5:
            t_dim, batch_dim, *_ = inputs.shape
            x = flatten(inputs, 0, 1)
            time_dim = True
        else:
            x = inputs
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, num_flat_features(x))
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        value = self.fc3(x)

        if features:
            feature_vec = x

        if time_dim:
            logits = logits.view(t_dim, batch_dim, self.actions_count)
            value = value.view(t_dim, batch_dim)
            if features:
                feature_vec = feature_vec.view(t_dim, batch_dim, feature_vec.shape[1])

        if features:
            return logits, value, feature_vec

        return logits, value

class WorkerTest(unittest.TestCase):

    def setUp(self):
        env = "PongNoFrameskip-v4"
        placeholder_env = make_stock_atari(env)
        actions_count = placeholder_env.action_space.n
        observation_shape = placeholder_env.observation_space.shape
        placeholder_env.close()

        self.flags = change_args(r_f_steps=20, reproducible=True, env=env, seed=123456, envs_per_worker=1, actions_count=actions_count, observation_shape=observation_shape)
        self.workers = []
        self.model_state_dict = torch.load("pong_model_save.pt")["model_state_dict"]

    def test_reproducibility(self):
        total_rewards_1 = 0.0
        total_rewards_2 = 0.0
        total_ep_steps_1 = 0.0
        total_ep_steps_2 = 0.0
        for j in range(2):
            worker = RolloutWorker(0, self.flags,  "")   # worker has to have sme id in order to be seeded the same way
            worker.load_model(self.model_state_dict)

            self.workers.append(worker)
            for i in range(120):
                _, _, iteration_rewards, iteration_ep_steps = self.workers[j].exec_and_eval_rollout()
                if j == 0:
                    total_rewards_1 += sum(iteration_rewards)
                    total_ep_steps_1 += sum(iteration_ep_steps)
                else:
                    total_rewards_2 += sum(iteration_rewards)
                    total_ep_steps_2 += sum(iteration_ep_steps)
        if total_rewards_1 == 0:
            raise ValueError('Total reward has to non-zero for test to work')
        self.assertEqual(total_rewards_1, total_rewards_2, 'total sum of rewards is not the same')
        self.assertEqual(total_ep_steps_1, total_ep_steps_2, 'total sum of episode_steps is not the same')

    def test_updated_model_output(self):
        flags = change_args(reproducible=False, r_f_steps=3000, envs_per_worker=2)
        self.flags = flags
        worker = RolloutWorker(0, self.flags, "")
        worker.load_model(self.model_state_dict)
        worker_data, worker_id, iteration_rewards, iteration_ep_steps = worker.exec_and_eval_rollout()
        self.assertGreater(sum(iteration_rewards), 30)
        self.assertGreater(sum(iteration_ep_steps), 12000)

        self.assertEqual(worker_id, 0)
        self.assertIsNotNone(worker_data)

