import abc


class CustomInputStrategy(metaclass=abc.ABCMeta):
    def __init__(self, flags):
        self.flags = flags

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'calc_index') and
                callable(subclass.calc_index)
                )

    @abc.abstractmethod
    def calc_index(self, model, buffer, sample_indices_threshold, device, feature_vecs, local_random, **kwargs):
        raise NotImplementedError
