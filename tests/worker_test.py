import unittest
import torch

from option_flags import flags, change_args
from wrappers.atari_wrappers import make_atari, wrap_pytorch, wrap_deepmind
from agent.worker.rollout_worker import RolloutWorker


class WorkerTest(unittest.TestCase):

    def setUp(self):
        env = "PongNoFrameskip-v4"
        placeholder_env = wrap_pytorch(
            wrap_deepmind(make_atari(env, 0), episode_life=True, clip_rewards=False, frame_stack=True,
                          scale=False))
        actions_count = placeholder_env.action_space.n
        observation_shape = placeholder_env.observation_space.shape
        placeholder_env.close()

        self.flags = change_args(r_f_steps=20, reproducible=True, env=env, seed=123456, envs_per_worker=2, actions_count=actions_count, observation_shape=observation_shape)
        self.workers = []

    def test_reproducibility(self):
        total_rewards_1 = 0.0
        total_rewards_2 = 0.0
        total_ep_steps_1 = 0.0
        total_ep_steps_2 = 0.0
        for j in range(4):
            self.workers.append(RolloutWorker(j % 2, self.flags))
            for i in range(40):
                _, _, iteration_rewards, iteration_ep_steps = self.workers[j].performing()
                if j <= 1:
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
        flags = change_args(reproducible=False, r_f_steps=100, envs_per_worker=10)
        self.flags = flags
        worker = RolloutWorker(0, self.flags)
        worker_data, worker_id, iteration_rewards, iteration_ep_steps = worker.performing(torch.load("trained_model.pt"), update=True)
        self.assertGreater(sum(iteration_rewards), 200)
        self.assertGreater(sum(iteration_ep_steps), 14000)
        self.assertEqual(worker_id, 0)
        self.assertIsNotNone(worker_data)

if __name__ == '__main__':
    unittest.main()
