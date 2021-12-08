import abc


class EliteSetSampleStrategy(metaclass=abc.ABCMeta):
    def __init__(self, flags):
        self.flags = flags
        self.buf_size = flags.elite_set_size

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'sample') and
                callable(subclass.sample))

    @abc.abstractmethod
    def sample(self, batch_size, replay_ratio, last_vec_index, feature_vecs):
        raise NotImplementedError

