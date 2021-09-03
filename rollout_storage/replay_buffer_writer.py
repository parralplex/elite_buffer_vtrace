from queue import Queue, Empty
from threading import Thread
from utils import compress


class ReplayWriter:
    def __init__(self, replay_buffer, batch_lock):
        self.replay_buffer = replay_buffer
        self.queue = Queue(maxsize=3)
        self.finished = False
        self.batch_lock = batch_lock

    def start(self):
        Thread(name="ReplayWriter", target=self.internal_writer).start()

    def write(self, worker_data):
        if self.queue.full():
            self.queue.get()
            self.queue.task_done()
        self.queue.put(worker_data)

    def internal_writer(self):
        while not self.finished:
            try:
                worker_data = self.queue.get(True, 1)
            except Empty:
                continue
            for i in range(len(worker_data)):
                self.replay_buffer.store(compress(worker_data[i].states.detach()),
                                         worker_data[i].actions.detach(),
                                         worker_data[i].rewards.detach(),
                                         worker_data[i].logits.detach(),
                                         worker_data[i].not_done.detach(),
                                         worker_data[i].feature_vec.detach(),
                                         self.batch_lock)
            self.queue.task_done()

    def close(self):
        assert not self.finished
        self.queue.join()
        self.finished = True
