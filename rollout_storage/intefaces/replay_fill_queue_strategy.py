import abc


class ReplayFillQueueStrategy(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'process_input') and
                callable(subclass.process_input))

    @abc.abstractmethod
    def process_input(self, replay_queue, input_data):
        raise NotImplementedError
