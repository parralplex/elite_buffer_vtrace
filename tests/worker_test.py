import unittest
import torch

from model.network import StateTransformationNetwork
from option_flags import flags, change_args
from wrappers.atari_wrappers import make_atari
from agent.worker.rollout_worker import RolloutWorker


class WorkerTest(unittest.TestCase):

    def setUp(self):
        env = "PongNoFrameskip-v4"
        placeholder_env = make_atari(env, 0)
        actions_count = placeholder_env.action_space.n
        observation_shape = placeholder_env.observation_space.shape
        placeholder_env.close()

        self.flags = change_args(r_f_steps=20, reproducible=True, env=env, seed=123456, envs_per_worker=1, actions_count=actions_count, observation_shape=observation_shape)
        self.workers = []
        self.state_transf_network = StateTransformationNetwork(self.flags)
        self.model_state_dict = torch.load("trained_model.pt")["model_state_dict"]

    def test_reproducibility(self):
        total_rewards_1 = 0.0
        total_rewards_2 = 0.0
        total_ep_steps_1 = 0.0
        total_ep_steps_2 = 0.0
        for j in range(2):
            worker = RolloutWorker(0, self.flags, self.state_transf_network.state_dict(), "")   # worker has to have sme id in order to be seeded the same way
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
        flags = change_args(reproducible=False, r_f_steps=2300, envs_per_worker=2)
        self.flags = flags
        worker = RolloutWorker(0, self.flags, self.state_transf_network.state_dict(), "")
        worker.load_model(self.model_state_dict)
        worker_data, worker_id, iteration_rewards, iteration_ep_steps = worker.exec_and_eval_rollout()
        self.assertGreater(sum(iteration_rewards), 37)
        self.assertGreater(sum(iteration_ep_steps), 12000)

        self.assertEqual(worker_id, 0)
        self.assertIsNotNone(worker_data)

if __name__ == '__main__':
    unittest.main()
