import abc


class EliteSetInputStrategy(metaclass=abc.ABCMeta):
    def __init__(self, flags):
        self.flags = flags
        self.buf_size = flags.elite_set_size

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'sample') and
                callable(subclass.sample) and
                hasattr(subclass, 'calc_index') and
                callable(subclass.calc_index))

    @abc.abstractmethod
    def before_sampling(self):
        raise NotImplementedError

    @abc.abstractmethod
    def calc_index(self,index, filled, feature_vecs, **kwargs):
        raise NotImplementedError

