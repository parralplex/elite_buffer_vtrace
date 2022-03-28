from rollout_storage.intefaces.replay_fill_queue_strategy import ReplayFillQueueStrategy


class KeepLatestStrategy(ReplayFillQueueStrategy):

    def process_input(self, replay_queue, input_data):
        if replay_queue.full():
            replay_queue.get()
            replay_queue.task_done()
        replay_queue.put(input_data)

