import abc


class LearnIterStrategyBase(metaclass=abc.ABCMeta):
    def __init__(self, worker_manager, stop_event, flags):
        self.flags = flags
        self.worker_manager = worker_manager
        self.stop_event = stop_event

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'after_batching') and
                callable(subclass.after_batching) and
                hasattr(subclass, 'before_learning') and
                callable(subclass.before_learning) and
                hasattr(subclass, 'after_learning') and
                callable(subclass.after_learning))

    @abc.abstractmethod
    def after_batching(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def before_learning(self):
        raise NotImplementedError

    @abc.abstractmethod
    def after_learning(self, **kwargs):
        raise NotImplementedError

    def clean_up(self, model):
        self.worker_manager.update_model_data(model)
        self.stop_event.set()
