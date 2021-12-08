import abc
from threading import Event


class ReplayBase(metaclass=abc.ABCMeta):
    def __init__(self):
        self.replay_filled_event = Event()

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'store_next') and
                callable(subclass.store_next) and
                hasattr(subclass, 'sample') and
                callable(subclass.sample) and
                hasattr(subclass, 'close') and
                callable(subclass.close) and
                hasattr(subclass, 'reset') and
                callable(subclass.reset) and
                hasattr(subclass, 'full') and
                callable(subclass.full))

    @abc.abstractmethod
    def store_next(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, batch_size):
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        raise NotImplementedError

    @abc.abstractmethod
    def full(self):
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError



