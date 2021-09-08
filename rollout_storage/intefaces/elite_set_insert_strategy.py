import abc
import torch
import random


class EliteSetInsertStrategy(metaclass=abc.ABCMeta):
    def __init__(self, flags):
        self.flags = flags

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'process_dist') and
                callable(subclass.process_dist) and
                hasattr(subclass, 'on_insert_before_filled') and
                callable(subclass.on_insert_before_filled))

    def calculate_best_index_pos(self, feature_vecs, new_feature_vec, new_reward, **kwargs):
        offset = 0
        if kwargs["random_search"]:
            offset = random.randint(0, self.flags.elite_set_size - 1)
        for i in range(self.flags.elite_set_size):
            if kwargs["random_search"]:
                index = (i + offset) % self.flags.elite_set_size
            else:
                index = i
            distance = torch.dist(feature_vecs[index], new_feature_vec, p=kwargs['p'])
            if not self.process_dist(distance, index):
                return

    @abc.abstractmethod
    def process_dist(self, distance, index) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def on_insert_before_filled(self, index, **kwargs):
        raise NotImplementedError
