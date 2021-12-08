import shutil
import unittest
import os

from option_flags import change_args
from stats.stats import Statistics
from threading import Event


class TestStats(unittest.TestCase):

    def setUp(self):
        if not os.path.exists("temp"):
            os.makedirs("temp")
        self.flags = change_args(max_avg_reward=10000000, max_episodes=10000000)
        self.stats = Statistics(Event(), "temp", self.flags, "", "", False, True)

    def tearDown(self) -> None:
        os.system('rm -rf temp')

    def test_rollout_data_processing(self):
        start_rew_1 = [10.1, 5, 7, 6.3, 7, 6, 6, 8]
        start_rew_2 = [15, 5, 4, 7, 34, 6, 2, 0.5, 3, 4]
        start_ep_steps_1 = [0, 5, 4, 8, 21, 33, 47, 53]
        start_ep_steps_2 = [5, 5, 4, 7, 34, 60, 28, 71, 38, 29]
        self.stats.process_worker_rollout(start_rew_1, start_ep_steps_1)
        self.stats.process_worker_rollout(start_rew_2, start_ep_steps_2)
        total_start_rew = sum(start_rew_1) + sum(start_rew_2)
        total_start_steps = sum(start_ep_steps_1) + sum(start_ep_steps_2)
        total_end_rew = 0
        total_end_steps = 0
        self.stats.file_writer.close()
        with open("temp/Scores.txt") as file:
            for line in file:
                total_end_rew += float(line.split(',')[0])
        with open("temp/Episode_steps.txt") as file:
            for line in file:
                total_end_steps += int(line.split(',')[0])

        self.assertEqual(total_start_rew, total_end_rew, 'total sum of rewards is not the same')
        self.assertEqual(total_start_steps, total_end_steps, 'total sum of episode_steps is not the same')

    def test_processing_unequal_data_length(self):
        start_rew = [10.1, 5, 7, 6.3, 7, 6, 6, 8]
        start_ep_steps_good = [4, 5, 6, 6.3, 7, 7, 6, 11]
        start_ep_steps = [0, 5, 4.5]

        self.stats.process_worker_rollout(start_rew, start_ep_steps_good)

        with self.assertRaises(ValueError):
            self.stats.process_worker_rollout(start_rew, start_ep_steps)
        self.stats.file_writer.close()


