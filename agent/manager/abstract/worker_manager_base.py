import abc
import signal
from stats.prof_timer import Timer
from utils.logger import logger


class WorkerManagerBase(metaclass=abc.ABCMeta):
    def __init__(self, stop_event, training_event, replay_writer, replay_buffers, stats, flags, file_save_url):
        self.replay_writer = replay_writer
        self.replay_buffers = replay_buffers
        self.stop_event = stop_event
        self.stats = stats
        self.training_event = training_event
        self.flags = flags
        self.file_save_url = file_save_url
        self.cache_filled_wait_timer = Timer("Manager waited for cache to be filled ", 300, 1, "Manager waited for cache to be filled {:0.4f} seconds avg({:0.4f})")

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'plan_and_execute_workers') and
                callable(subclass.plan_and_execute_workers) and
                hasattr(subclass, 'pre_processing') and
                callable(subclass.pre_processing) and
                hasattr(subclass, 'update_model_data') and
                callable(subclass.update_model_data) and
                hasattr(subclass, 'reset') and
                callable(subclass.reset)
                )

    def manage_workers(self):
        try:
            first_caching = False
            while True:
                if self.stop_event.is_set():
                    self.clean_up()
                    break

                if self.flags.reproducible and self.training_event.is_set() and first_caching:
                    for p in range(len(self.replay_buffers)):
                        with self.cache_filled_wait_timer:
                            self.replay_buffers[p].cache_filled_event.wait()

                self.pre_processing()
                workers_data = self.plan_and_execute_workers()

                for i in range(len(workers_data)):
                    if self.stop_event.is_set():
                        break
                    self.store_worker_data(workers_data[i][0])

                    _, _, rewards, ep_steps = workers_data[i]
                    self.stats.process_worker_rollout(rewards, ep_steps)

                if self.flags.reproducible and self.training_event.is_set():
                    for p in range(len(self.replay_buffers)):
                        self.replay_buffers[p].cache(self.flags.cache_sample_size, True)
                    if not first_caching:
                        first_caching = True
        except Exception as exp:
            logger.exception("Manager thread exception: " + str(exp))
            signal.raise_signal(signal.SIGINT)
            if not self.flags.reproducible:
                logger.exception("Worker manager thread raised new exception - ending execution")
                self.stop_event.set()
                self.clean_up()
                for p in range(len(self.replay_buffers)):
                    self.replay_buffers[p].close()
            else:
                raise

    @abc.abstractmethod
    def plan_and_execute_workers(self):
        raise NotImplementedError

    def store_worker_data(self, worker_data):
        self.replay_writer.write(worker_data)

    @abc.abstractmethod
    def pre_processing(self):
        raise NotImplementedError

    @abc.abstractmethod
    def update_model_data(self, current_model):
        raise NotImplementedError

    def clean_up(self):
        if self.flags.reproducible:
            self.cache_filled_wait_timer.save_stats(self.file_save_url)
        for p in range(len(self.replay_buffers)):
            self.replay_buffers[p].close()

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

