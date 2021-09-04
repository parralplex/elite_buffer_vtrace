
from rollout_storage.intefaces.replay_fill_queue_strategy import ReplayFillQueueStrategy


class KeepOldestStrategy(ReplayFillQueueStrategy):

    def process_input(self, replay_queue, input_data):
        if not replay_queue.full():
            replay_queue.put(input_data)

