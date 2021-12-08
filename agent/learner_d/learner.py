
import time
import numpy as np
import torch
import threading
import torch.backends.cudnn
import json
from agent.algorithms.v_trace import v_trace
from agent.learner_d.builder.learner_builder import LearnerBuilder
from agent.learner_d.builder.learner_builder_sync import LearnerBuilderSync
from scheduler.polynomial_lr_scheduler import PolynomialLRDecay

from model.network import ModelNetwork
from stats.stats import Statistics
from utils import logger


class Learner(object):
    def __init__(self, flags, run_id):
        self.run_id = run_id
        self.flags = flags
        self.file_save_dir_url = "results/" + self.flags.env + "_" + str(self.run_id)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            logger.info("Learner is using CUDA - GPU execution")
        else:
            logger.info("CUDA not available - CPU execution")
        self.model = ModelNetwork(self.flags.actions_count, self.flags).to(self.device)

        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(),
            lr=self.flags.lr,
            eps=0.1,

        )
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.flags.lr)
        if not hasattr(self.optimizer, '__str__'):
            raise MissingMethod("Optimizer doesnt have __str__ method implemented, which is required")

        self.lr_scheduler = PolynomialLRDecay(self.optimizer, self.flags.scheduler_steps, 0.00001, 2)
        if not hasattr(self.lr_scheduler, '__str__'):
            raise MissingMethod("Scheduler doesnt have __str__ method implemented, which is required")

        if self.flags.op_mode == "train_w_load":
            self._load_model_state(self.flags.load_model_uri, self.device)
            logger.info("Model state successfully loaded from file save")

        self.learning_lock = threading.Lock()
        self.batch_lock = threading.Lock()
        self.mini_batcher_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.training_event = threading.Event()

        self.current_batch_size = self.flags.batch_size

        self.stats = Statistics(self.stop_event, self.file_save_dir_url, self.flags, str(self.optimizer),
                                str(self.lr_scheduler), self.flags.basic_verbose, self.flags.background_save)

        self.training_iteration = 0

        self.replay_buffers = []

        if self.flags.reproducible:
            builder = LearnerBuilderSync(self.flags)
        else:
            builder = LearnerBuilder(self.flags)

        self.replay_writer, self.replay_buffers, self.worker_manager, self.strategy = builder. \
            add_replay(self.file_save_dir_url, self.training_event). \
            add_elite((self.flags.feature_out_layer_size,), self.file_save_dir_url, self.training_event). \
            create_replay_writer(self.stop_event).create_manager(self.stop_event, self.training_event, self.model, self.stats,
                                                  self.file_save_dir_url). \
            create_strategy(self.stop_event).get_result()

        if len(self.replay_buffers) == 0:
            raise ForbiddenSetting(
                "No replay buffer has been selected - application has not been modified to work without any buffer to store worker trajectories")

        self.replay_writer.start()

        data_ratio = 0
        for replay in self.replay_buffers:
            data_ratio += replay.replay_data_ratio
        if data_ratio != 1:
            raise ForbiddenSetting("Total ratio of data used in batches from all replays has to be 1, but now it is: " + str(data_ratio))

    def start(self):
        threads = []
        thread = threading.Thread(target=self.worker_manager.manage_workers, name="manage_workers")
        thread.start()
        threads.append(thread)

        for i in range(self.flags.learner_thread_count):
            thread = threading.Thread(target=self.learning, name="learning_thread-%d" % i)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        logger.info("Training has ended. Beginning to clean up resources and process collected data.")
        self._save_current_model_state()
        self.replay_writer.close()
        self.worker_manager.reset()
        self.stats.close()
        return 0

    def learning(self):
        local_random = np.random.RandomState(self.flags.seed)
        try:
            for p in range(len(self.replay_buffers)):
                self.replay_buffers[p].replay_filled_event.wait()

            self.stats.warm_up_period_event.wait()

            self.training_event.set()

            while self.training_iteration < self.flags.training_max_steps:
                if self.stop_event.is_set():
                    break
                self._learning_iteration(local_random)
                if self.training_iteration % self.flags.save_model_period == 0:
                    self._save_current_model_state()
        except Exception as exp:
            logger.exception("Learning thread raise new exception - ending execution")
        self.strategy.clean_up(model=self.model)

        self.worker_manager.update_model_data(self.model)
        for p in range(len(self.replay_buffers)):
            self.replay_buffers[p].close()
        if self.replay_writer.queue.full():
            self.replay_writer.remove_queue_element()
        # for i in range(len(self.replay_buffers)):
        #     if not self.replay_buffers[i].finished:
        #         self.replay_buffers[i].cache(1)
        #         self.replay_buffers[i].sample(1, local_random)


    def _learning_iteration(self, local_random):

        try:
            actions, beh_logits, not_done, rewards, states, counter = self._prepare_batch(local_random)
            if actions is None:
                return

            states, actions, rewards, beh_logits, not_done = states.to(self.device), actions.to(self.device), rewards.to(self.device), beh_logits.to(self.device), not_done.to(self.device)

            res = self.strategy.after_batching(states=states, current_batch_size=self.current_batch_size, counter=counter)
            if not res:
                if self.flags.reproducible and not self.stop_event.is_set():
                    raise BlockingError("Leaving iteration in this state will result in deadlock!")
                return

            with self.learning_lock:
                self.training_iteration += 1
                self.strategy.before_learning()

                bootstrap_value, current_logits, current_values = self._forward_pass(states)
                baseline_loss, entropy_loss, policy_loss = v_trace(actions, beh_logits, bootstrap_value,
                                                                   current_logits, current_values, not_done, rewards, self.flags)

                self._backprop(policy_loss, baseline_loss, entropy_loss)
                self.strategy.after_learning(model=self.model, training_iteration=self.training_iteration)
        except RuntimeError as exp:
            if 'out of memory' in str(exp):
                self._handle_out_of_mem_error(exp)
                return
            else:
                raise exp

    def _handle_out_of_mem_error(self, exp):
        logger.warning("System ran out of memory, trying to lower the batch_size")
        for p in self.model.parameters():
            if p.grad is not None:
                del p.grad
        torch.cuda.empty_cache()
        time.sleep(5)
        self.current_batch_size -= 2
        if self.current_batch_size <= 0:
            logger.exception("Unable to lower batch_size anymore - cannot handle memory overflow")
            raise exp
        if self.flags.reproducible:
            logger.exception("Dynamic batch_size is NOT allowed in SYNCHRONIZED - REPLICABLE mode")
            raise exp
        self.stats.change_batch_size(self.current_batch_size)

    def _forward_pass(self, states):
        target_logits, target_values = self.model(states)

        bootstrap_value = target_values[-1].squeeze(-1)
        target_values = target_values.squeeze(-1)[:-1]
        target_logits = target_logits[:-1]

        return bootstrap_value, target_logits, target_values

    def _backprop(self, policy_loss, baseline_loss, entropy_loss):
        loss = policy_loss + baseline_loss + entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.flags.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.stats.process_learning_iter(policy_loss.item(), baseline_loss.item(), entropy_loss.item(), self.lr_scheduler.get_last_lr()[0])

    def _prepare_batch(self, local_random):
        if self.flags.reproducible:

            self.mini_batcher_lock.acquire()
            if self.stop_event.is_set():
                self.mini_batcher_lock.release()
                return None, None, None, None, None, 0

        states, actions, rewards, beh_logits, not_done, counter = self.replay_buffers[0].sample(
            self.current_batch_size, local_random)

        if states is None or self.stop_event.is_set():
            if self.flags.reproducible:
                self.mini_batcher_lock.release()
            return None, None, None, None, None, 0

        for i in range(1, len(self.replay_buffers)):
            states_n, actions_n, rewards_n, beh_logits_n, not_done_n, ctr = self.replay_buffers[i].sample(
                 self.current_batch_size, local_random)
            if self.flags.reproducible and counter != ctr:
                raise ThreadingSyncError("Data indexes of mini-batches should be the same ! - synchronization error")
            if states_n is None or self.stop_event.is_set():
                if self.flags.reproducible:
                    self.mini_batcher_lock.release()
                return None, None, None, None, None, 0
            states = torch.cat((states, states_n), 0)
            actions = torch.cat((actions, actions_n), 0)
            rewards = torch.cat((rewards, rewards_n), 0)
            beh_logits = torch.cat((beh_logits, beh_logits_n), 0)
            not_done = torch.cat((not_done, not_done_n), 0)
        if self.flags.reproducible:
            self.mini_batcher_lock.release()

        states, actions, rewards, beh_logits, not_done = states.transpose(1, 0), actions.transpose(1, 0), rewards.transpose(1, 0), beh_logits.transpose(1, 0), not_done.transpose(1, 0)

        return actions, beh_logits, not_done, rewards, states, counter

    def _save_current_model_state(self):
        torch.save({"model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.lr_scheduler.state_dict(),
                    "flags": vars(self.flags),
                    },
                   self.file_save_dir_url + '/regular_model_save_.pt')
        with open(self.file_save_dir_url + '/options_flags.json', 'w') as file:
            json.dump(self.flags.__dict__, file, indent=2)

    def _load_model_state(self, url, device):
        state_dict = torch.load(url, map_location=device)
        self.model.load_state_dict(state_dict["model_state_dict"])
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        self.flags = state_dict["flags"]


class MissingMethod(Exception):
    pass


class ForbiddenSetting(Exception):
    pass


class ThreadingSyncError(Exception):
    pass


class BlockingError(Exception):
    pass
