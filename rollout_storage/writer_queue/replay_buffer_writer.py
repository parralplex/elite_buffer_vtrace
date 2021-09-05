from queue import Queue, Empty
from threading import Thread

from rollout_storage.intefaces.replay_fill_queue_strategy import ReplayFillQueueStrategy
from utils import compress


class ReplayWriterQueue:
    def __init__(self, replay_buffers, queue_size: int, fill_in_strategy: ReplayFillQueueStrategy):
        self.replay_buffers = replay_buffers
        self.queue = Queue(maxsize=queue_size)
        self.finished = False
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
                for j in range(len(self.replay_buffers)):
                    self.replay_buffers[j].store_next(state=compress(worker_data[i].states),
                                                      action=worker_data[i].actions,
                                                      reward=worker_data[i].rewards,
                                                      logits=worker_data[i].logits,
                                                      not_done=worker_data[i].not_done,
                                                      feature_vec=worker_data[i].feature_vec,
                                                      random_search=True,
                                                      add_rew_feature=True,
                                                      p=2)
            self.queue.task_done()

    def close(self):
        assert not self.finished
        self.queue.join()
        self.finished = True