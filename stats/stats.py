import os
import sys
import numpy as np
import datetime as dt
from queue import Queue

from stats.data_plotter import set_global_chart_settings, create_chart
from stats.safe_file_writer import SafeOrderedMultiFileWriter
from option_flags import flags


stat_file_names = ["Scores.txt", "Train_time.txt", "Episode_steps.txt", "Loss_file.txt",
                   "lr_file.txt", "max_reward_file.txt"]


class Statistics(object):
    def __init__(self, stop_event, file_save_dir_url, verbose=False):
        self.stop_event = stop_event
        self.warm_up_period = 0
        self.max_reward = -sys.maxsize
        self.max_avg_reward = -sys.maxsize
        self.file_writer = SafeOrderedMultiFileWriter(self._generate_file_urls(stat_file_names, file_save_dir_url))
        self.file_writer.start()
        self.verbose = verbose
        self.file_save_dir_url = file_save_dir_url

        self.episodes = 0
        self.START_TIME = dt.datetime.now()
        self.WARM_UP_TIME = dt.datetime.now()
        self.worker_rollout_counter = 0
        self.train_iter_counter = 0
        self.score_queue = Queue(maxsize=flags.avg_buff_size)
        self.last_lr = 0

    @staticmethod
    def _generate_file_urls(names, path):
        urls = names
        for i in range(len(stat_file_names)):
            urls[i] = path + "/" + names[i]
        return urls

    def mark_warm_up_period(self):
        self.warm_up_period = self.episodes
        self.WARM_UP_TIME = dt.datetime.now()

    def process_worker_rollout(self, rewards, ep_steps):
        self.worker_rollout_counter += 1
        self.episodes += len(rewards)

        self.file_writer.write(rewards, 0)  # score_file
        self.file_writer.write(ep_steps, 2) # ep_step_file

        if len(rewards) > 0:
            local_max_reward = np.max(rewards)
            if local_max_reward > self.max_reward:
                self.max_reward = local_max_reward

            self.file_writer.write([self.max_reward for _ in range(len(rewards))], 5)

        new_max_rew = False
        rew_avg = -sys.maxsize
        if not self.score_queue.empty():
            rew_avg = np.average(list(self.score_queue.queue))
            if rew_avg > self.max_avg_reward:
                self.max_avg_reward = rew_avg
                new_max_rew = True
                if self.max_avg_reward >= flags.max_avg_reward:
                    self.stop_event.set()

        if len(rewards) > 0:
            self.file_writer.write([str(len(rewards)) + ',' + str(dt.datetime.now() - self.START_TIME) + ',' + str((dt.datetime.now() - self.START_TIME).total_seconds()) + "," + str(self.train_iter_counter)], 1)
        if self.verbose:
            self._verbose_process_rollout(rewards, new_max_rew, rew_avg)
        if self.episodes >= flags.max_episodes:
            self.stop_event.set()

    def _verbose_process_rollout(self, rewards, new_max_rew, rew_avg):
        for k in range(len(rewards)):
            if self.score_queue.full():
                self.score_queue.get()
            self.score_queue.put(rewards[k])

        if new_max_rew:
            print("New MAX average reward per 100/ep: ", self.max_avg_reward)

        if self.worker_rollout_counter % flags.verbose_output_interval == 0:
            print('Episode ', self.episodes, '  Iteration: ', self.worker_rollout_counter, "  Avg. reward 100/ep: ",
                  rew_avg, " Training iterations: ", self.train_iter_counter)

    def process_learning_iter(self, policy_loss, baseline_loss, entropy_loss, lr):
        self.train_iter_counter += 1
        self.last_lr = lr
        self.file_writer.write([str(policy_loss) + ',' + str(baseline_loss) + ',' + str(entropy_loss)], 3)
        self.file_writer.write([str(lr)])

        if self.verbose and self.train_iter_counter % flags.verbose_output_interval == 0:
            print("Training iterations: ", self.train_iter_counter, " Lr:", lr, " Total_loss:", policy_loss+baseline_loss+entropy_loss,
                  " Policy_loss:", policy_loss, " Baseline_loss:", baseline_loss, " Entropy_loss:", entropy_loss)

    def close(self):
        self.file_writer.close()
        stats_file_desc = open(self.file_save_dir_url, "w", 1)
        stats_file_desc.write("Warm_up_period: " + str(self.warm_up_period) + '\n')
        stats_file_desc.write("Max_reach_reward: " + str(self.max_reward) + '\n')
        stats_file_desc.write("Max_avg(100)_reward: " + str(self.max_avg_reward) + '\n')
        stats_file_desc.write("Total_episodes: " + str(self.episodes) + '\n')
        stats_file_desc.write("Total_worker_rollout_iter: " + str(self.worker_rollout_counter) + '\n')
        stats_file_desc.write("Total_learning_iter: " + str(self.process_learning_iter) + '\n')
        stats_file_desc.write("Last_lr: " + str(self.last_lr) + '\n')
        current_time = dt.datetime.now()
        stats_file_desc.write("Total_execution_time: " + str(current_time - self.START_TIME) + '\n')
        if self.warm_up_period > 0:
            stats_file_desc.write("Total_learning_time: " + str(current_time - self.WARM_UP_TIME) + '\n')
            stats_file_desc.write("Total_warm_up_time: " + str(self.WARM_UP_TIME - self.START_TIME) + '\n')
        stats_file_desc.flush()
        stats_file_desc.close()

    def _create_charts(self):
        avg_buf_size = self.episodes * 0.001
        ignore_period = avg_buf_size / 2
        os.mkdir(self.file_save_dir_url + "/Charts")
        set_global_chart_settings()
        create_chart(self.file_save_dir_url, stat_file_names[0], 'Episodes', 'Reward per episode', ["Avg(" + str(avg_buf_size) + ")"], "reward_chart.png", avg_buf_size)
        create_chart(self.file_save_dir_url, stat_file_names[2], 'Episodes', 'Steps per episode', ["Avg(" + str(avg_buf_size) + ")"],
                     "episode_steps_chart.png", avg_buf_size)
        create_chart(self.file_save_dir_url, stat_file_names[5], 'Episodes', 'Avg Max reward', ["Avg(" + str(avg_buf_size) + ")max reward"],
                     "max_avg_reward_chart.png", avg_buf_size)
        create_chart(self.file_save_dir_url, stat_file_names[5], 'Episodes', 'Max reward', ["max reward"],
                     "max_reward_chart.png", avg_buf_size, False)
        create_chart(self.file_save_dir_url, stat_file_names[3], 'Training iteration', 'Loss',
                     ["policy", "baseline", "entropy", "total"],
                     "loss_chart.png", avg_buf_size, ignore_period)
        create_chart(self.file_save_dir_url, stat_file_names[4], 'Training iteration', 'Learning rate decay',
                     ["Avg(" + str(avg_buf_size) + ")"],
                     "lr_decay_chart.png", avg_buf_size)
