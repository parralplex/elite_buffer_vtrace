import time

from rollout_storage.intefaces.replay_base import ReplayBase
from queue import Queue


class ReplayQueue(ReplayBase):
    def __init__(self, flags, training_event):
        super().__init__()
        self.flags = flags
        self.replay_data_ratio = self.flags.replay_queue_ratio
        self.input_queue = Queue(maxsize=self.flags.batch_size)
        self.training_started = False
        self.training_event = training_event

    def store_next(self, **kwargs):
        if not self.replay_filled_event.is_set() and self.input_queue.full():
            self.replay_filled_event.set()
        if not self.training_event.is_set() and self.input_queue.full():
            self.input_queue.get()
            self.input_queue.task_done()
        if self.flags.reproducible:
            self.input_queue.put(kwargs['data'], timeout=0.5)
        else:
            self.input_queue.put(kwargs['data'])
        return True

    def sample(self, batch_size, local_random=None):
        if not self.training_started:
            self.training_event.wait()
            self.training_started = True

        data_batch = []

        for i in range(int(batch_size * self.replay_data_ratio)):
            if self.input_queue.empty():
                dd = 0
            data_part = self.input_queue.get()
            self.input_queue.task_done()
            if data_part is None:
                return None
            data_batch.append(data_part)

        return data_batch

    def close(self):
        if self.input_queue.empty():
            self.input_queue.put(None)
            time.sleep(1)
        while not self.input_queue.empty():
            self.input_queue.get()
            self.input_queue.task_done()
        self.input_queue.join()

    def full(self):
        return self.input_queue.full()

    def reset(self):
        pass

