from agent.learner_d.strategy.learn_async_strategy import LearnAsyncStrategy
from agent.manager.native_manager_async import NativeManagerAsync

from rollout_storage.elite_set.elite_set_replay import EliteSetReplay
from rollout_storage.elite_set.strategies.dist_input_filter import DistFilteringInputStrategy
from rollout_storage.elite_set.strategies.policy_sample import PolicySampleStrategy
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

    def add_replay(self, file_save_dir_url, training_event):
        replay = ReplayQueue(self.flags, training_event)
        replay = ExperienceReplayProxy(replay, file_save_dir_url, self.flags.caching)
        replay.start()
        self.replay_buffers.append(replay)
        if self.flags.use_replay_buffer:
            replay = ExperienceReplayTorch(self.flags, training_event)
            replay = ExperienceReplayProxy(replay, file_save_dir_url, self.flags.caching)
            replay.start()
            self.replay_buffers.append(replay)
        return self

    def add_elite(self, feature_vec_dim, file_save_dir_url, training_event):
        if self.flags.use_elite_set:
            elite_insert_strategy = None
            elite_sample_strategy = None
            if self.flags.elite_insert_strategy == "dist_input_filter":
                elite_insert_strategy = DistFilteringInputStrategy(self.flags)

            if self.flags.elite_sample_strategy == "policy_sample":
                elite_sample_strategy = PolicySampleStrategy(self.flags)

            elite_set = EliteSetReplay(feature_vec_dim, elite_insert_strategy, elite_sample_strategy, self.flags, file_save_dir_url, training_event)
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
        return self

    def create_strategy(self, stop_event):
        if self.worker_manager is None:
            raise InitializationError("Manager have to be created before creating learner strategy")
        self.strategy = LearnAsyncStrategy(self.worker_manager, stop_event, self.flags)
        return self

    def get_result(self):
        return self.replay_writer, self.replay_buffers, self.worker_manager, self.strategy

