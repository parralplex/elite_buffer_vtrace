from queue import Queue, Empty
from threading import Thread, Lock

from rollout_storage.intefaces.replay_fill_queue_strategy import ReplayFillQueueStrategy
from utils import compress


class ReplayWriterQueue:
    def __init__(self, replay_buffer, batch_lock: Lock, queue_size: int, fill_in_strategy: ReplayFillQueueStrategy):
        self.replay_buffer = replay_buffer
        self.queue = Queue(maxsize=queue_size)
        self.finished = False
        self.batch_lock = batch_lock
        self.fill_in_strategy = fill_in_strategy

    def start(self):
        Thread(name="ReplayWriter", target=self.internal_writer).start()

    def write(self, worker_data):
        self.fill_in_strategy.process_input(self.queue, worker_data)

    def internal_writer(self):
        while not self.finished:
            try:
                worker_data = self.queue.get(True, 1)
            except Empty:
                continue
            for i in range(len(worker_data)):
                self.replay_buffer.store(compress(worker_data[i].states),
                                         worker_data[i].actions,
                                         worker_data[i].rewards,
                                         worker_data[i].logits,
                                         worker_data[i].not_done,
                                         worker_data[i].feature_vec,
                                         self.batch_lock)
            self.queue.task_done()

    def close(self):
        assert not self.finished
        self.queue.join()
        self.finished = True
