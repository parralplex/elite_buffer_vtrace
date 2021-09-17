import time
import torch
import random

import psutil as psu

from rollout_storage.intefaces.replay_buf_base import ReplayBufferBase
from queue import Queue
from threading import Thread, Event, Lock

from stats.prof_timer import Timer
from utils import compress, decompress, logger


class ExperienceReplayProxy(ReplayBufferBase):
    def __init__(self, experience_replay, file_save_url, caching=True):
        self.experience_replay = experience_replay
        self.input_queue = Queue(maxsize=1)
        self.output_cache = Queue()
        self.internal_thread = None
        self.finished = False
        self.batch_size = self.experience_replay.flags.batch_size
        self.cache_filled_event = Event()
        if self.flags.use_state_compression:
            self.experience_replay.states = [None for i in range(self.experience_replay.actions.shape[0])]
        self.caching = caching
        self.cache_data_pos_pointer = 0
        self.cache_data_pos_pointer_lock = Lock()
        self.cache_wait_timer = Timer("Avg cache wait time ", 300, 1, "Learner has to wait to get batch from cache for too long = {:0.4f} seconds avg({:0.4f}). Try increasing cache size to speed up calculation.")
        self.file_save_url = file_save_url

    def store_next(self, **kwargs):
        state = kwargs['state']
        if self.flags.use_state_compression:
            state = compress(state)
        kwargs['state'] = state
        self.experience_replay.store_next(**kwargs)

    def random_sample(self, batch_size, local_random=None):
        if batch_size < self.batch_size:
            self.batch_size = batch_size
        cache_data_pos_pointer = 0
        if self.caching:

            with self.cache_data_pos_pointer_lock:
                with self.cache_wait_timer:
                    res = self.output_cache.get()

                cache_data_pos_pointer = self.cache_data_pos_pointer
                self.cache_data_pos_pointer = (self.cache_data_pos_pointer + 1) % self.experience_replay.flags.max_cache_pos_pointer
            if res[0] is None:
                self.output_cache.task_done()
                states, actions, rewards, logits, not_done, _ = res
                return states, actions, rewards, logits, not_done, cache_data_pos_pointer
        else:
            if local_random is None:
                res = self.experience_replay.random_sample(self.batch_size)
            else:
                res = self.experience_replay.random_sample(self.batch_size, local_random)
        if self.flags.use_state_compression:
            states = []
            for state in res[0]:
                states.append(decompress(state))
        else:
            states = res[0]
        res[0] = torch.stack(states)
        if self.caching:
            self.output_cache.task_done()
        states, actions, rewards, logits, not_done, _ = res

        return states, actions, rewards, logits, not_done, cache_data_pos_pointer

    def on_policy_sample(self, batch_size):
        raise NotImplementedError

    def start(self):
        self.internal_thread = Thread(name="ReplayReader", target=self.internal_reader).start()

    def cache(self, cache_size):
        self.cache_filled_event.clear()
        if self.output_cache.unfinished_tasks + cache_size > (2 * self.experience_replay.flags.replay_out_cache_size):
            logger.warning("Replay cache buffer is filling up! Beware for running out of memory.  Try lowering replay caching setting.")
        # memory_usage = psu.virtual_memory()[2] + psu.swap_memory()[3]
        # if memory_usage >= 180:
        #     memory_usage = (memory_usage / 200) * 100
        #     logger.warning(f"Running out of available  system memory! Usage:{memory_usage}%")
        self.input_queue.put(cache_size)

    def internal_reader(self):
        local_random = random.Random(self.experience_replay.flags.seed)
        while not self.finished:
            cache_size = self.input_queue.get()
            self.input_queue.task_done()
            for i in range(cache_size):
                states, actions, rewards, logits, not_done, _ = self.experience_replay.random_sample(self.batch_size, local_random)
                self.output_cache.put([states, actions, rewards, logits, not_done, None])
            self.cache_filled_event.set()

    def close(self):
        assert not self.finished
        if self.output_cache.empty():
            for i in range(self.experience_replay.flags.learner_thread_count):
                self.output_cache.put([None, None, None, None, None, None])
        time.sleep(1)
        while not self.output_cache.empty():
            self.output_cache.get()
            self.output_cache.task_done()
        self.output_cache.join()
        self.finished = True
        if self.input_queue.empty():
            self.input_queue.put(0)
        time.sleep(1)
        while not self.input_queue.empty():
            self.input_queue.get()
            self.input_queue.task_done()
        self.input_queue.join()
        if self.internal_thread is not None:
            self.internal_thread.join()
        self.experience_replay.close()
        if self.experience_replay.flags.reproducible:
            self.cache_wait_timer.save_stats(self.file_save_url)

    # pass through original object attributes if they doesnt exist on the proxy
    # https://stackoverflow.com/a/26092256
    def __getattr__(self, attr):
        return getattr(self.experience_replay, attr)

