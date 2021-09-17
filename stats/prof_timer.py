import time
import numpy as np
from queue import Queue
from utils import logger


class Timer:
    def __init__(self, name=None, output_freq=0, output_threshold=0, output_text=""):
        self.name = name
        self._start_time = None
        self.timer_queue = Queue(maxsize=1000)
        self._last_output_time = None
        self.output_freq = output_freq
        self.output_threshold = output_threshold
        self.output_text = output_text

    def __enter__(self):
        self._start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        current_time = time.perf_counter()
        elapsed_time = current_time - self._start_time

        if self.timer_queue.full():
            self.timer_queue.get()
        self.timer_queue.put(elapsed_time)

        if self.output_threshold != 0:
            if self._last_output_time is None:
                self._last_output_time = current_time
            else:
                if self._last_output_time - current_time >= self.output_freq:
                    if elapsed_time >= self.output_threshold:
                        avg_elapsed_time = np.average(self.timer_queue.queue)
                        logger.info(self.output_text.format(elapsed_time, avg_elapsed_time))
                        self._last_output_time = current_time

    def save_stats(self, file_save_url):
        with open(file_save_url + "/timings.txt", "a") as file:
            avg_elapsed_time = np.average(self.timer_queue.queue)
            file.write(self.name + " {:0.4f} seconds \n".format(avg_elapsed_time))


