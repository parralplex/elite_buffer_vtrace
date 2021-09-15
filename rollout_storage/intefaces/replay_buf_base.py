import abc


class ReplayBufferBase(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'store_next') and
                callable(subclass.store_next) and
                hasattr(subclass, 'random_sample') and
                callable(subclass.random_sample) and
                hasattr(subclass, 'on_policy_sample') and
                callable(subclass.on_policy_sample) and
                hasattr(subclass, 'close') and
                callable(subclass.close))

    @abc.abstractmethod
    def store_next(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def random_sample(self, batch_size):
        raise NotImplementedError

    @abc.abstractmethod
    def on_policy_sample(self, batch_size):
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        raise NotImplementedError


