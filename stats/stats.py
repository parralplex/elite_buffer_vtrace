import signal
import sys
import numpy as np
import datetime as dt
from queue import Queue

from stats.safe_file_writer import SafeOrderedMultiFileWriter
from utils.logger import logger


stat_file_names = ["Scores.txt", "Train_time.txt", "Episode_steps.txt", "Loss_file.txt",
                   "lr_file.txt", "max_reward_file.txt"]


class Statistics(object):
    def __init__(self, stop_event, file_save_dir_url, flags, optimizer_str_desc, scheduler_str_desc, verbose=False, background_file_save=True):
        self.flags = flags
        self.stop_event = stop_event
        self.max_reward = -sys.maxsize
        self.max_avg_reward = -sys.maxsize
        self.file_writer = SafeOrderedMultiFileWriter(self._generate_file_urls(stat_file_names, file_save_dir_url))
        if background_file_save:
            self.file_writer.start()
        self.verbose = verbose
        self.file_save_dir_url = file_save_dir_url

        self.episodes = 0
        self.START_TIME = dt.datetime.now()
        self.WARM_UP_TIME = dt.datetime.now()
        self.worker_rollout_counter = 0
        self.train_iter_counter = 0
        self.score_queue = Queue(maxsize=self.flags.avg_buff_size)
        self.last_lr = 0
        self.dyn_batch_size = self.flags.batch_size
        self.background_file_save = background_file_save
        self.optimizer_str_desc = optimizer_str_desc
        self.scheduler_str_desc = scheduler_str_desc
        self.max_avg_rew_with_dev_time = None
        signal.signal(signal.SIGTERM, self.on_critical_state_save)
        signal.signal(signal.SIGINT, self.on_critical_state_save)
        signal.signal(signal.SIGABRT, self.on_critical_state_save)

        self.total_env_steps = 0

    @staticmethod
    def _generate_file_urls(names, path):
        urls = []
        for i in range(len(stat_file_names)):
            urls.append(path + "/" + names[i])
        return urls

    def process_worker_rollout(self, rewards, ep_steps):
        if len(rewards) != len(ep_steps):
            raise ValueError("Unequal number of rewards and steps is not possible!")
        self.worker_rollout_counter += 1
        self.episodes += len(rewards)

        self.file_writer.write([str(rewards[i]) + ',' + str(self.train_iter_counter * self.flags.batch_size * self.flags.r_f_steps) + ',' + str(self.total_env_steps) + ',' + str((dt.datetime.now() - self.START_TIME).total_seconds()) for i in range(len(rewards))], 0)  # score_file
        self.file_writer.write([str(ep_steps[i]) + ',' + str(self.train_iter_counter * self.flags.batch_size * self.flags.r_f_steps)+ ',' + str(self.total_env_steps) + ',' + str((dt.datetime.now() - self.START_TIME).total_seconds()) for i in range(len(ep_steps))], 2) # ep_step_file

        if len(rewards) > 0:
            local_max_reward = np.max(rewards)
            if local_max_reward > self.max_reward:
                self.max_reward = local_max_reward

            self.file_writer.write([self.max_reward for _ in range(len(rewards))], 5)

        for k in range(len(rewards)):
            if self.score_queue.full():
                self.score_queue.get()
            self.score_queue.put(rewards[k])

        new_max_rew = False
        rew_avg = -sys.maxsize
        if not self.score_queue.empty():
            rew_avg = np.average(list(self.score_queue.queue))
            if rew_avg > self.max_avg_reward:
                self.max_avg_reward = rew_avg
                new_max_rew = True

        if len(rewards) > 0:
            self.file_writer.write([str(len(rewards)) + ',' + str(dt.datetime.now() - self.START_TIME) + ',' + str((dt.datetime.now() - self.START_TIME).total_seconds()) + "," + str(self.train_iter_counter)], 1)
        if self.verbose:
            self._verbose_process_rollout(new_max_rew, rew_avg)

        self.total_env_steps += self.flags.envs_per_worker * self.flags.r_f_steps

        if self.total_env_steps >= self.flags.environment_max_steps:
            self.stop_event.set()
            self.close()

    def _verbose_process_rollout(self, new_max_rew, rew_avg):
        if new_max_rew:
            print("New MAX avg rew per 100/ep: ", "{:.2f}".format(self.max_avg_reward))

        if self.worker_rollout_counter % self.flags.verbose_worker_out_int == 0:
            print('Episode ', self.episodes, '  Iteration: ', self.worker_rollout_counter, "  Avg(100)rew: ",
                  "{:.2f}".format(rew_avg), " Time_steps: ", self.train_iter_counter * self.flags.batch_size * self.flags.r_f_steps, "Total_env_steps: ", self.total_env_steps)

    def change_batch_size(self, new_batch_size):
        self.dyn_batch_size = new_batch_size

    def process_learning_iter(self, policy_loss, baseline_loss, entropy_loss, lr):
        self.train_iter_counter += 1
        self.last_lr = lr
        self.file_writer.write([str(policy_loss) + ',' + str(baseline_loss) + ',' + str(entropy_loss)], 3)
        self.file_writer.write([str(lr)], 4)

        if self.verbose and self.train_iter_counter % self.flags.verbose_learner_out_int == 0:
            print("Train iter: ", self.train_iter_counter, " Lr:", "{:.9f}".format(lr), " Total_loss:", "{:.4f}".format(policy_loss+baseline_loss+entropy_loss),
                  " Run time:", dt.datetime.now() - self.START_TIME)

    def close(self):
        if not self.background_file_save:
            logger.info("Writing collected data to txt files.")
            self.file_writer.block_on_get = False
            self.file_writer.finished = True
            self.file_writer.internal_writer()
            self.file_writer.close_data_operators()
        else:
            self.file_writer.close()

        stats_file_desc = open(self.file_save_dir_url + "/training_summary.txt", "w", 1)
        stats_file_desc.write("Max_reach_reward: " + str(self.max_reward) + '\n')
        stats_file_desc.write("Max_avg(100)_reward: " + str(self.max_avg_reward) + '\n')
        stats_file_desc.write("Total_episodes: " + str(self.episodes) + '\n')
        stats_file_desc.write("Total_worker_rollout_iter: " + str(self.worker_rollout_counter) + '\n')
        stats_file_desc.write("Total_training_steps: " + str(self.train_iter_counter * self.flags.batch_size * self.flags.r_f_steps) + '\n')
        stats_file_desc.write("Total_environment_steps: " + str(self.total_env_steps) + '\n')
        stats_file_desc.write("Last_lr: " + str(self.last_lr) + '\n')
        stats_file_desc.write("Batch_size_used_without_out_of_memory_error: " + str(self.dyn_batch_size) + '\n')
        current_time = dt.datetime.now()
        stats_file_desc.write("Total_execution_time: " + str(current_time - self.START_TIME) + '\n')

        stats_file_desc.write("Optimizer: " + self.optimizer_str_desc + '\n')
        stats_file_desc.write("Scheduler: " + self.scheduler_str_desc + '\n')
        stats_file_desc.flush()
        stats_file_desc.close()

    def on_critical_state_save(self, *args):
        self.stop_event.set()
        self.close()
        if args[0] == signal.SIGABRT:
            logger.warning("Program execution aborted by OS - Possible reason: unable to allocate more virtual/physical memory.")

