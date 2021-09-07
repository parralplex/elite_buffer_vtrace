import abc
from utils import compress

from option_flags import flags


class WorkerManagerBase(metaclass=abc.ABCMeta):
    def __init__(self, stop_event, replay_writer, replay_buffers, stats):
        self.replay_writer = replay_writer
        self.replay_buffers = replay_buffers
        self.stop_event = stop_event
        self.stats = stats

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'plan_and_execute_workers') and
                callable(subclass.plan_and_execute_workers) and
                hasattr(subclass, 'pre_processing') and
                callable(subclass.pre_processing) and
                hasattr(subclass, 'update_model_data') and
                callable(subclass.update_model_data))

    def manage_workers(self, post_processing_func=None, *args):
        while True:
            if self.stop_event.is_set():
                self.clean_up()
                break
            self.pre_processing()
            workers_data = self.plan_and_execute_workers()
            for i in range(len(workers_data)):
                self.store_worker_data(workers_data[i][0])
                _, _, rewards, ep_steps = workers_data[i]
                self.stats.process_worker_rollout(rewards, ep_steps)
            if flags.reproducible:
                post_processing_func(args[0])

    @abc.abstractmethod
    def plan_and_execute_workers(self):
        raise NotImplementedError

    def store_worker_data(self, worker_data):
        if flags.reproducible:
            for i in range(len(worker_data)):
                for p in range(len(self.replay_buffers)):
                    state = worker_data[i].states
                    if flags.use_state_compression:
                        state = compress(state)
                    self.replay_buffers[p].store_next(state=state,
                                                      action=worker_data[i].actions,
                                                      reward=worker_data[i].rewards,
                                                      logits=worker_data[i].logits,
                                                      not_done=worker_data[i].not_done,
                                                      feature_vec=worker_data[i].feature_vec,
                                                      random_search=flags.random_search,
                                                      add_rew_feature=flags.add_rew_feature,
                                                      p=flags.p)
        else:
            self.replay_writer.write(worker_data)

    @abc.abstractmethod
    def pre_processing(self):
        raise NotImplementedError

    @abc.abstractmethod
    def update_model_data(self, current_model):
        raise NotImplementedError

    def clean_up(self):
        pass
