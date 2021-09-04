import random
from rollout_storage.intefaces.replay_fill_queue_strategy import ReplayFillQueueStrategy


class AlternatingStrategy(ReplayFillQueueStrategy):
    def __init__(self, drop_input_chance=0.5):
        self.drop_input_chance = drop_input_chance

    def process_input(self, replay_queue, input_data):
        if random.random() < self.drop_input_chance:
            return
        if replay_queue.full():
            replay_queue.get()
            replay_queue.task_done()
        replay_queue.put(input_data)

    def set_drop_input_chance(self, new_value):
        self.drop_input_chance = new_value
