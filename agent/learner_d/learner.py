import signal
import time
import numpy as np
import torch
import threading
import torch.backends.cudnn
import json
from agent.algorithms.v_trace import v_trace
from agent.learner_d.builder.learner_builder import LearnerBuilder
from agent.learner_d.builder.learner_builder_sync import LearnerBuilderSync
from rollout_storage.custom_replay.custom_replay import CustomReplay
from scheduler.polynomial_lr_scheduler import PolynomialLRDecay
from utils.parameter_schema import validate_config, replay_schema, custom_replay_schema

from model.network import ModelNetwork
from stats.stats import Statistics
from utils.logger import logger
from setuptools_scm import get_version
from option_flags import change_args, set_defaults


class Learner(object):
    def __init__(self, flags, run_id, additional_args):
        self.run_id = run_id
        self.flags = flags

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            logger.info("Learner is using CUDA - GPU execution")
        else:
            logger.info("CUDA not available - CPU execution")

        if self.flags.op_mode == "train_w_load":
            self._load_model_state(self.flags.load_model_url, self.device, additional_args)
            logger.info("Model state successfully loaded from file save")
        else:
            self.model = ModelNetwork(self.flags.actions_count, self.flags.frames_stacked,
                                      self.flags.feature_out_layer_size, self.flags.use_additional_scaling_FC_layer).to(
                self.device)
            self._init_optimizer_and_scheduler()

        self.file_save_dir_url = "results/" + self.flags.env + "_" + str(self.run_id)

        self.learning_lock = threading.Lock()
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

        self.replay_writer, self.replay_buffers, self.worker_manager, self.strategy = self._build_agent(builder)

        if len(self.replay_buffers) == 0:
            raise ForbiddenSetting("No replay buffer has been selected!")

        self.replay_writer.start()

        data_ratio = 0
        for replay in self.replay_buffers:
            data_ratio += replay.replay_sample_ratio
            if (replay.replay_sample_ratio * self.flags.batch_size) % 1 > 0:
                raise ForbiddenSetting("Only whole number mini-batch size is allowed! PLease change your sample-ratio / batch-size accordingly. Current values: " + str(replay.replay_sample_ratio) + " / " + self.flags.batch_size + ".  Minibatch size: " + str(replay.replay_sample_ratio * self.flags.batch_size))
        if data_ratio != 1:
            raise ForbiddenSetting("Total sampling ratio must be 1, but now it is: " + str(data_ratio))

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
        return 0

    def learning(self):
        local_random = np.random.RandomState(self.flags.seed)
        try:
            for p in range(len(self.replay_buffers)):
                self.replay_buffers[p].replay_filled_event.wait()

            self.training_event.set()

            while self.training_iteration < self.flags.environment_max_steps:
                if self.stop_event.is_set():
                    break
                self._learning_iteration(local_random)
                if self.training_iteration % self.flags.save_model_period == 0:
                    self._save_current_model_state()
        except Exception as exp:
            logger.exception("Learning thread exception: " + str(exp))
            signal.raise_signal(signal.SIGINT)
        self.strategy.clean_up(model=self.model)

        self.worker_manager.update_model_data(self.model)
        for p in range(len(self.replay_buffers)):
            self.replay_buffers[p].close()
        if self.replay_writer.queue.full():
            self.replay_writer.remove_queue_element()

    def _learning_iteration(self, local_random):

        try:
            actions, beh_logits, not_done, rewards, states, values, counter = self._prepare_batch(local_random)
            if actions is None:
                return

            states, actions, rewards, beh_logits, not_done, values = states.to(self.device), actions.to(self.device), rewards.to(self.device), beh_logits.to(self.device), not_done.to(self.device), values.to(self.device)

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
                                                                   current_logits, current_values, not_done, rewards, values, self.flags)

                self._backprop(policy_loss, baseline_loss, entropy_loss)
                self.strategy.after_learning(model=self.model, training_iteration=self.training_iteration)
                for replay in self.replay_buffers:
                    if type(replay.experience_replay) is CustomReplay:
                        replay.train_model.load_state_dict(self.model.state_dict())

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
        target_logits, target_values, _ = self.model(states)

        bootstrap_value = target_values[-1].squeeze(-1)
        target_values = target_values.squeeze(-1)[:-1]
        target_logits = target_logits[:-1]

        return bootstrap_value, target_logits, target_values

    def _backprop(self, policy_loss, value_loss, entropy_loss):
        loss = policy_loss + value_loss + entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.flags.gradient_clip_by_norm_threshold)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.stats.process_learning_iter(policy_loss.item(), value_loss.item(), entropy_loss.item(), self.lr_scheduler.get_last_lr()[0])

    def _prepare_batch(self, local_random):
        if self.flags.reproducible:

            self.mini_batcher_lock.acquire()
            if self.stop_event.is_set():
                self.mini_batcher_lock.release()
                return None, None, None, None, None, None, 0

        states, actions, rewards, beh_logits, not_done, values, counter = self.replay_buffers[0].sample(
            self.current_batch_size, local_random)

        if states is None or self.stop_event.is_set():
            if self.flags.reproducible:
                self.mini_batcher_lock.release()
            return None, None, None, None, None, None, 0

        for i in range(1, len(self.replay_buffers)):
            states_n, actions_n, rewards_n, beh_logits_n, not_done_n, values_n, ctr = self.replay_buffers[i].sample(
                 self.current_batch_size, local_random)
            if self.flags.reproducible and counter != ctr:
                raise ThreadingSyncError("Data indexes of mini-batches should be the same ! - synchronization error")
            if states_n is None or self.stop_event.is_set():
                if self.flags.reproducible:
                    self.mini_batcher_lock.release()
                return None, None, None, None, None, None, 0
            states = torch.cat((states, states_n), 0)
            actions = torch.cat((actions, actions_n), 0)
            rewards = torch.cat((rewards, rewards_n), 0)
            beh_logits = torch.cat((beh_logits, beh_logits_n), 0)
            not_done = torch.cat((not_done, not_done_n), 0)
            values = torch.cat((values, values_n), 0)
        if self.flags.reproducible:
            self.mini_batcher_lock.release()

        states, actions, rewards, beh_logits, not_done, values = states.transpose(1, 0), actions.transpose(1, 0), rewards.transpose(1, 0), beh_logits.transpose(1, 0), not_done.transpose(1, 0), values.transpose(1, 0)

        return actions, beh_logits, not_done, rewards, states, values, counter

    def _save_current_model_state(self):
        try:
            current_version = get_version()
        except Exception as exp:
            logger.warning("We were unable to obtain current project version from git tags.")
            current_version = "Unknown"
        torch.jit.script(self.model).save(self.file_save_dir_url + '/agent_model_scripted_save.pt')
        torch.save({"model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.lr_scheduler.state_dict(),
                    "flags": vars(self.flags),
                    "project_version": current_version
                    },
                   self.file_save_dir_url + '/checkpoint_data_save.pt')
        with open(self.file_save_dir_url + '/options_flags.json', 'w') as file:
            json.dump(self.flags.__dict__, file, indent=2)

    def _load_model_state(self, url, device, additional_args):
        try:
            state_dict = torch.load(url + '/checkpoint_data_save.pt', map_location=device)
            set_defaults(**state_dict["flags"])
            self.flags = change_args(**additional_args)
            self.model = torch.jit.load(url + '/agent_model_scripted_save.pt').to(device)
        except Exception as exp:
            logger.warning("Error encountered while loading model. " + str(exp))
            raise exp
        self._init_optimizer_and_scheduler()
        if self.flags.load_optimizer_save:
            self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            self.lr_scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        try:
            if get_version() != state_dict["project_version"]:
                logger.warning("Loaded model and project have mismatched versions - project_version:" + get_version() + "  loaded_model_version:" + state_dict["project_version"])
        except Exception as exp:
            logger.warning("We were unable to verify save VERSION - version mismatch may cause problems.")

    def _build_agent(self, builder):
        for replay_dict in self.flags.replay_parameters:
            if replay_dict["type"] == "queue":
                replay_dict, success, error_msg = validate_config(replay_schema, replay_dict)
                if not success:
                    raise ForbiddenSetting("Replay parameters dictionary is not valid: " + error_msg)
                builder.add_replay_queue(self.file_save_dir_url, self.training_event, replay_dict)
            elif replay_dict["type"] == "standard":
                replay_dict, success, error_msg = validate_config(replay_schema, replay_dict)
                if not success:
                    raise ForbiddenSetting("Replay parameters dictionary is not valid: " + error_msg)
                builder.add_replay(self.file_save_dir_url, self.training_event, replay_dict)
            elif replay_dict["type"] == "custom":
                replay_dict, success, error_msg = validate_config(custom_replay_schema, replay_dict)
                if not success:
                    raise ForbiddenSetting("Replay parameters dictionary is not valid: " + error_msg)
                builder.add_custom_replay(self.file_save_dir_url, self.training_event, self.model, self.device, replay_dict)
            else:
                raise ForbiddenSetting("Unknown replay buffer type: " + replay_dict["type"])

        builder.create_replay_writer(self.stop_event). \
                create_manager(self.stop_event, self.training_event, self.model, self.stats, self.file_save_dir_url). \
                create_strategy(self.stop_event)

        return builder.get_result()

    def _init_optimizer_and_scheduler(self):
        if self.flags.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.flags.lr)
        elif self.flags.optimizer == "rmsprop":
            self.optimizer = torch.optim.RMSprop(
                self.model.parameters(),
                lr=self.flags.lr,
                eps=self.flags.rmsprop_eps,
            )
        else:
            raise ForbiddenSetting("Unknown optimizer selected - try 'adam' or 'rmsprop' instead")

        if not hasattr(self.optimizer, '__str__'):
            raise MissingMethod("Optimizer doesnt have __str__ method implemented, which is required")

        self.lr_scheduler = PolynomialLRDecay(self.optimizer, self.flags.lr_scheduler_steps, self.flags.lr_end_value, 2)
        if not hasattr(self.lr_scheduler, '__str__'):
            raise MissingMethod("Scheduler doesnt have __str__ method implemented, which is required")

class MissingMethod(Exception):
    pass


class ForbiddenSetting(Exception):
    pass


class ThreadingSyncError(Exception):
    pass


class BlockingError(Exception):
    pass
