import time
from queue import Queue, Empty, Full
from threading import Thread

from rollout_storage.intefaces.replay_fill_queue_strategy import ReplayFillQueueStrategy


class ReplayWriterQueue:
    def __init__(self, replay_buffers, queue_size: int, fill_in_strategy: ReplayFillQueueStrategy, flags, stop_event):
        self.flags = flags
        self.replay_buffers = replay_buffers
        self.queue = Queue(maxsize=queue_size)
        self.finished = False
        self.fill_in_strategy = fill_in_strategy
        self.internal_thread = None
        self.stop_event = stop_event

    def start(self):
        self.internal_thread = Thread(name="ReplayWriter", target=self.internal_writer).start()

    def write(self, worker_data):
        self.queue.put(worker_data)
        # self._write_to_replay(worker_data)
        # if self.flags.reproducible:
        #     self._write_to_replay(worker_data)
        # else:
        #     self.fill_in_strategy.process_input(self.queue, worker_data)

    def internal_writer(self):
        while not self.finished:
            try:
                worker_data = self.queue.get(timeout=2)
            except Empty:
                continue
            self._write_to_replay(worker_data)
            self.queue.task_done()
            if self.stop_event.is_set():
                self.finished = True

    def _write_to_replay(self, data):
        for i in range(len(data)):
            for j in range(len(self.replay_buffers)):
                try:
                    self.replay_buffers[j].store_next(data=data[i])
                except Full:
                    for p in range(len(self.replay_buffers)):
                        self.replay_buffers[p].cache(1)
                    try:
                        self.replay_buffers[j].store_next(data=data[i])
                    except Full as full:
                        if self.stop_event.is_set():
                            return
                        else:
                            raise full
                if self.stop_event.is_set():
                    return

    def remove_queue_element(self):
        self.queue.get()
        self.queue.task_done()

    def close(self):
        while not self.queue.empty():
            self.queue.get()
            self.queue.task_done()
        self.queue.join()
        self.finished = True
        if self.internal_thread is not None:
            self.internal_thread.join()
