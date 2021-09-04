import abc


class ReplayFillQueueStrategy(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'load_data_source') and
                callable(subclass.load_data_source))

    @abc.abstractmethod
    def process_input(self, replay_queue, input_data):
        raise NotImplementedError
