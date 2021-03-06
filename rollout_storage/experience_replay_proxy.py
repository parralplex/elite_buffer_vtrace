import time
from concurrent.futures.thread import ThreadPoolExecutor

import torch
import numpy as np

from queue import Queue, Full
from threading import Thread, Event, Lock

from stats.prof_timer import Timer
from utils.compression import compress, decompress
from utils.logger import logger


class ExperienceReplayProxy:
    def __init__(self, experience_replay, file_save_url, caching=True):
        self.experience_replay = experience_replay
        self.input_queue = Queue(maxsize=1)
        self.output_cache = Queue(maxsize=experience_replay.flags.cache_output_buffer_size)
        self.internal_thread = None
        self.finished = False
        self.batch_size = self.experience_replay.flags.batch_size
        self.cache_filled_event = Event()
        self.caching = caching
        self.cache_data_pos_pointer = 0
        self.cache_data_pos_pointer_lock = Lock()
        self.cache_wait_timer = Timer("Avg cache wait time ", 300, 1, "Learner has to wait to get batch from cache for too long = {:0.4f} seconds avg({:0.4f}). Try increasing cache size to speed up calculation.")
        self.file_save_url = file_save_url
        self.output_cache_counter = 0
        self.additional_thread_count = self.experience_replay.flags.caching_threads-1
        if self.additional_thread_count > 0:
            self.executor = ThreadPoolExecutor(max_workers=self.additional_thread_count)

    def store_next(self, **kwargs):

        kwargs['feature_vec'] = kwargs['data'].feature_vec
        kwargs['state'] = kwargs['data'].states

        if self.flags.use_replay_compression:
            kwargs['data'] = compress(kwargs['data'], self.flags.lz4_compression_level)
        return self.experience_replay.store_next(**kwargs)

    def sample(self, batch_size, local_random=None):

        if self.finished:
            return None, None, None, None, None, None, 0
        if batch_size < self.batch_size:
            self.batch_size = batch_size
        cache_data_pos_pointer = 0
        if self.experience_replay.flags.reproducible or (self.caching):

            with self.cache_data_pos_pointer_lock:
                with self.cache_wait_timer:
                    res = self.output_cache.get()

                cache_data_pos_pointer = self.cache_data_pos_pointer
                self.cache_data_pos_pointer = (self.cache_data_pos_pointer + 1) % 1000000
                self.output_cache.task_done()
            if res is None:
                return None, None, None, None, None, None, cache_data_pos_pointer
        else:
            if local_random is None:
                res = self.experience_replay.sample(self.batch_size)
            else:
                res = self.experience_replay.sample(self.batch_size, local_random)

        batch_size = len(res)
        states = torch.zeros(batch_size, self.flags.r_f_steps + 1, *self.flags.observation_shape)
        actions = torch.zeros(batch_size, self.flags.r_f_steps, dtype=torch.int64)
        rewards = torch.zeros(batch_size, self.flags.r_f_steps)
        logits = torch.zeros(batch_size, self.flags.r_f_steps, self.flags.actions_count)
        not_done = torch.zeros(batch_size, self.flags.r_f_steps)
        values = torch.zeros(batch_size, self.flags.r_f_steps)

        for i in range(batch_size):
            if self.flags.use_replay_compression:
                data_sample = decompress(res[i])
            else:
                data_sample = res[i]
            states[i] = data_sample.states
            actions[i] = data_sample.actions
            rewards[i] = data_sample.rewards
            logits[i] = data_sample.logits
            not_done[i] = data_sample.not_done
            values[i] = data_sample.values

        return states, actions, rewards, logits, not_done, values, cache_data_pos_pointer

    def start(self):
        if self.caching or self.experience_replay.flags.reproducible:
            self.internal_thread = Thread(name="ReplayReader", target=self.internal_reader).start()

    def cache(self, cache_size, final=False):
        self.cache_filled_event.clear()
        if self.output_cache.unfinished_tasks + cache_size > (2 * self.experience_replay.flags.cache_output_buffer_size):
            logger.warning(f"Replay cache buffer is filling up {self.output_cache.unfinished_tasks}! Beware for running out of memory.  Try lowering replay caching setting.")
        # memory_usage = psu.virtual_memory()[2] + psu.swap_memory()[3]
        # if memory_usage >= 180:
        #     memory_usage = (memory_usage / 200) * 100
        #     logger.warning(f"Running out of available  system memory! Usage:{memory_usage}%")
        self.output_cache_counter += cache_size
        self.input_queue.put([cache_size, final])

    def internal_reader(self):
        local_random = np.random.RandomState(self.experience_replay.flags.seed)
        self.experience_replay.training_event.wait()
        while not self.finished:
            cache_data = None
            if self.experience_replay.flags.reproducible:
                cache_data = self.input_queue.get()
                self.input_queue.task_done()
                if self.finished:
                    break
                cache_size = cache_data[0]
            else:
                cache_size = 1

            while cache_size > 0:
                batch_data = None
                if self.additional_thread_count > 0:
                    future_results = []
                    for i in range(self.additional_thread_count):
                        future_results.append(self.executor.submit(self.experience_replay.sample, self.batch_size, local_random))
                        cache_size -= 1
                        if cache_size == 0:
                            break
                    if cache_size > 0:
                        batch_data = self.experience_replay.sample(self.batch_size, local_random)
                        cache_size -= 1
                    for future_exec in future_results:
                        self.output_cache.put(future_exec.result())
                    if batch_data is not None:
                        self.output_cache.put(batch_data)
                else:
                    self.output_cache.put(self.experience_replay.sample(self.batch_size, local_random))
                    cache_size -= 1

            if self.experience_replay.flags.reproducible and cache_data[1]:
                self.cache_filled_event.set()

    def close(self):
        if not self.finished:
            self.experience_replay.close()
            if self.caching or self.experience_replay.flags.reproducible:
                if self.output_cache.empty():
                    for i in range(self.experience_replay.flags.learner_thread_count):
                        try:
                            self.output_cache.put(None, timeout=1)
                        except Full:
                            pass
                time.sleep(1)
                while not self.output_cache.empty():
                    self.output_cache.get()
                    self.output_cache.task_done()
                self.output_cache.join()
            self.finished = True
            if self.caching or self.experience_replay.flags.reproducible:
                if self.input_queue.empty():
                    self.input_queue.put(0)
                time.sleep(1)
                while not self.input_queue.empty():
                    self.input_queue.get()
                    self.input_queue.task_done()
                self.input_queue.join()
                if self.internal_thread is not None:
                    self.internal_thread.join()
            if self.experience_replay.flags.reproducible:
                self.cache_wait_timer.save_stats(self.file_save_url)

    def reset(self):
        self.experience_replay.reset()

    def full(self):
        self.experience_replay.full()

    # pass through original object attributes if they doesnt exist on the proxy
    # https://stackoverflow.com/a/26092256
    def __getattr__(self, attr):
        return getattr(self.experience_replay, attr)

