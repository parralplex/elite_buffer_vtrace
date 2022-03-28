from agent.learner_d.strategy.learn_async_strategy import LearnAsyncStrategy
from agent.manager.native_manager_async import NativeManagerAsync

from rollout_storage.custom_replay.custom_replay import CustomReplay
from rollout_storage.custom_replay.strategies.elite_insertion import EliteInsertStrategy
from rollout_storage.custom_replay.strategies.elite_sampling import EliteSampleStrategy

from rollout_storage.custom_replay.strategies.attentive_sampling import AttentiveSampleStrategy
from rollout_storage.experience_replay import ExperienceReplayTorch
from rollout_storage.experience_replay_proxy import ExperienceReplayProxy
from rollout_storage.experience_replay_queue import ReplayQueue
from rollout_storage.writer_queue.alternating_strategy import AlternatingStrategy
from rollout_storage.writer_queue.keep_latest_strategy import KeepLatestStrategy
from rollout_storage.writer_queue.keep_oldest_strategy import KeepOldestStrategy
from rollout_storage.writer_queue.replay_buffer_writer import ReplayWriterQueue
import torch.multiprocessing as mp


class InitializationError(Exception):
    pass


class ForbiddenSetting(Exception):
    pass


class LearnerBuilder(object):
    def __init__(self, flags):
        self.flags = flags
        self.replay_writer = None
        self.replay_buffers = []
        self.worker_manager = None
        self.strategy = None

    def reset(self):
        self.replay_writer = None
        self.replay_buffers = []
        self.worker_manager = None
        self.strategy = None

    def add_replay(self, file_save_dir_url, training_event, replay_dict):
        replay = ExperienceReplayTorch(self.flags, training_event, replay_dict)
        replay = ExperienceReplayProxy(replay, file_save_dir_url, self.flags.caching)
        replay.start()
        self.replay_buffers.append(replay)
        return self

    def add_replay_queue(self, file_save_dir_url, training_event, replay_dict):
        replay = ReplayQueue(self.flags, training_event, replay_dict)
        replay = ExperienceReplayProxy(replay, file_save_dir_url, self.flags.caching)
        replay.start()
        self.replay_buffers.append(replay)

    def add_custom_replay(self, file_save_dir_url, training_event, model, device, replay_dict):
        sampling_strategy = None
        insertion_strategy = None
        elite_insert_strategy = None
        elite_sample_strategy = None
        if "insert_strategy" in replay_dict.keys():
            insertion_strategy = replay_dict["insert_strategy"]
        if "sample_strategy" in replay_dict.keys():
            sampling_strategy = replay_dict["sample_strategy"]

        if sampling_strategy == "elite_sampling":
            elite_sample_strategy = EliteSampleStrategy(self.flags, replay_dict["alfa_annealing_factor"], replay_dict["lambda_batch_multiplier"], replay_dict["elite_sampling_strategy"], replay_dict["dist_function"], replay_dict["p"])
        elif sampling_strategy == "attentive_sampling":
            elite_sample_strategy = AttentiveSampleStrategy(self.flags, replay_dict["alfa_annealing_factor"], replay_dict["lambda_batch_multiplier"], replay_dict["elite_sampling_strategy"], replay_dict["dist_function"], replay_dict["p"])
        if insertion_strategy == "elite_insertion":
            elite_insert_strategy = EliteInsertStrategy(self.flags, replay_dict["elite_batch_size"], replay_dict["dist_function"], replay_dict["p"])

        elite_set = CustomReplay(elite_insert_strategy, elite_sample_strategy, self.flags, file_save_dir_url, training_event, model, device, replay_dict)
        elite_set = ExperienceReplayProxy(elite_set, file_save_dir_url, self.flags.caching)
        elite_set.start()
        self.replay_buffers.append(elite_set)
        return self

    def create_replay_writer(self, stop_event):
        if len(self.replay_buffers) == 0:
            raise InitializationError("Replay buffer has to be created before creating replay writer")
        replay_writer_strategy = None
        if self.flags.discarding_strategy == "keep_latest":
            replay_writer_strategy = KeepLatestStrategy()
        elif self.flags.discarding_strategy == "keep_oldest":
            replay_writer_strategy = KeepOldestStrategy()
        elif self.flags.discarding_strategy == "alternating":
            replay_writer_strategy = AlternatingStrategy()
        self.replay_writer = ReplayWriterQueue(self.replay_buffers, queue_size=self.flags.replay_writer_queue_size,
                                               fill_in_strategy=replay_writer_strategy, flags=self.flags, stop_event=stop_event)
        return self

    def create_manager(self, stop_event, training_event, model, stats, file_save_dir_url):
        if len(self.replay_buffers) == 0 or self.replay_writer is None:
            raise InitializationError("Replay buffer and writer have to be created before creating manager")
        if self.flags.multiprocessing_backend == "ray":
            from agent.manager.ray_manager_async import RayManagerAsync  # local import so ray is not imported when using python_native multiprocessing
            self.worker_manager = RayManagerAsync(stop_event, training_event, self.replay_writer, self.replay_buffers,
                                             model, stats, self.flags, file_save_dir_url, self.flags.worker_verbose)
        elif self.flags.multiprocessing_backend == "python_native":
            if mp.get_start_method() != "spawn":
                self.replay_writer.close()
                stats.close()
                raise ForbiddenSetting("This app only supports 'spawn' type sub-processes when working with python-native backend")
            self.worker_manager = NativeManagerAsync(stop_event, training_event, self.replay_writer,
                                                self.replay_buffers, model, stats, self.flags, file_save_dir_url,
                                                self.flags.worker_verbose)
        else:
            raise ForbiddenSetting("Unknown multiprocessing module selected: " + self.flags.multiprocessing_backend)
        return self

    def create_strategy(self, stop_event):
        if self.worker_manager is None:
            raise InitializationError("Manager have to be created before creating learner strategy")
        self.strategy = LearnAsyncStrategy(self.worker_manager, stop_event, self.flags)
        return self

    def get_result(self):
        return self.replay_writer, self.replay_buffers, self.worker_manager, self.strategy

