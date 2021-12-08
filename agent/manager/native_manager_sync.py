import time

from agent.manager.abstract.native_manager import NativeManager
from model.network import StateTransformationNetwork
from stats.prof_timer import Timer
from utils import logger
from queue import Queue
from agent.worker.native_rollout_worker import start_worker_sync
import torch.multiprocessing as mp
from torch.multiprocessing import  Event, Barrier


class NativeManagerSync(NativeManager):
    def __init__(self, stop_event, training_event, replay_writer, replay_buffers, model, stats, flags, file_save_url, verbose=False):
        super().__init__(stop_event, training_event, replay_writer, replay_buffers, model, stats, flags, file_save_url)
        self.model_loaded_event = Event()
        self.worker_sync_barrier = Barrier(flags.worker_count)
        self.state_transf_network = StateTransformationNetwork(self.flags)
        for i in range(flags.worker_count):
            process = mp.Process(target=start_worker_sync, args=(
            i, self.worker_data_queue, self.shared_list, flags, self.model_loaded_event, self.worker_sync_barrier, self.state_transf_network.state_dict(), file_save_url, verbose))
            process.start()
            self.workers.append(process)
        self.expected_worker_id = 0
        self.worker_data_pos_buf = [None for i in range(flags.worker_count)]
        self.model_update_queue = Queue()
        self.last_model_get_time = time.perf_counter()
        self.model_update_wait_timer = Timer("Manager model update time ", 120, 1,
                                             "Manager is waiting for model update for quite some time = {:0.4f} seconds avg({:0.4f})! Try lowering replay caching setting.")
        self.model_dict_send_counter = 0

    def plan_and_execute_workers(self):
        prepared_data = []
        while len(prepared_data) < self.flags.worker_count:
            if self.worker_data_pos_buf[self.expected_worker_id] is not None:
                prepared_data.append(self.worker_data_pos_buf[self.expected_worker_id])
                self.worker_data_pos_buf[self.expected_worker_id] = None
                self.expected_worker_id = (self.expected_worker_id + 1) % self.flags.worker_count
            else:
                worker_data = self.worker_data_queue.get()
                if worker_data[1] == self.expected_worker_id:
                    prepared_data.append(worker_data)
                    self.expected_worker_id = (self.expected_worker_id + 1) % self.flags.worker_count
                else:
                    self.worker_data_pos_buf[worker_data[1]] = worker_data
        return prepared_data

    def pre_processing(self):
        if self.training_event.is_set():

            counter_start = self.model_dict_send_counter
            for i in range(counter_start, self.replay_buffers[0].output_cache_counter):
                self.model_dict_send_counter += 1
                with self.model_update_wait_timer:
                    model_dict = self.model_update_queue.get()
                self.model_update_queue.task_done()
                if self.stop_event.is_set():
                    return
            self.shared_list[0] = model_dict

        self.model_loaded_event.set()

    def update_model_data(self, current_model):
        self.model_update_queue.put({k: v.cpu() for k, v in current_model.state_dict().items()})

    def clean_up(self):
        super(NativeManagerSync, self).clean_up()
        self.model_loaded_event.set()
        for i in range(len(self.workers)):
            self.workers[i].join()
        self.shared_list[:] = []
        self.model_update_wait_timer.save_stats(self.file_save_url)