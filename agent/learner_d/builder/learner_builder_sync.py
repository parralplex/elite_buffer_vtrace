from agent.learner_d.builder.learner_builder import LearnerBuilder, InitializationError, ForbiddenSetting

from agent.learner_d.strategy.learn_sync_strategy import LearnSyncStrategy
from agent.manager.native_manager_sync import NativeManagerSync
import torch.multiprocessing as mp


class LearnerBuilderSync(LearnerBuilder):
    def __init__(self, flags):
        super().__init__(flags)

    def create_manager(self, stop_event, training_event, model, stats, file_save_dir_url):
        if len(self.replay_buffers) == 0 or self.replay_writer is None:
            raise InitializationError("Replay buffer and writer have to be created before creating manager")
        if self.flags.multiprocessing_backend == "ray":
            from agent.manager.ray_manager_sync import RayManagerSync  # local import so ray is not imported when using python_native multiprocessing
            self.worker_manager = RayManagerSync(stop_event, training_event, self.replay_writer, self.replay_buffers,
                                             model, stats, self.flags, file_save_dir_url, verbose=False)
        elif self.flags.multiprocessing_backend == "python_native":
            if mp.get_start_method() != "spawn":
                self.replay_writer.close()
                stats.close()
                raise ForbiddenSetting("This app only supports 'spawn' type sub-processes when working with python-native backend")
            self.worker_manager = NativeManagerSync(stop_event, training_event, self.replay_writer,
                                                self.replay_buffers, model, stats, self.flags, file_save_dir_url,
                                                False)
        return self

    def create_strategy(self, stop_event):
        if self.worker_manager is None:
            raise InitializationError("Manager have to be created before creating learner strategy")
        self.strategy = LearnSyncStrategy(self.worker_manager, stop_event, self.flags)
        return self
