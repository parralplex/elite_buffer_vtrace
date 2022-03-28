import abc


class CustomSampleStrategy(metaclass=abc.ABCMeta):
    def __init__(self, flags):
        self.flags = flags

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'sample') and
                callable(subclass.sample))

    @abc.abstractmethod
    def sample(self, feature_vecs, batch_size, replay_ratio, model, buffer, allowed_values, local_random, device, last_vec_index, sampling_counter):
        raise NotImplementedError

