import time
from queue import Queue

from agent.manager.worker_manager_base import WorkerManagerBase
from agent.worker.native_rollout_worker import start_worker
import torch.multiprocessing as mp
from torch.multiprocessing import Manager, Event, Barrier

from stats.prof_timer import Timer
from utils import logger


class NativeWorkerManager(WorkerManagerBase):
    def __init__(self, stop_event, training_event, replay_writer, replay_buffers, model, stats, flags, file_save_url, verbose=False):
        super().__init__(stop_event, training_event, replay_writer, replay_buffers, stats, flags, file_save_url)
        self.workers = []
        self.manager = Manager()
        self.worker_data_queue = self.manager.Queue(maxsize=flags.shared_queue_size)
        self.shared_list = self.manager.list()
        self.shared_list.append({k: v.cpu() for k, v in model.state_dict().items()})
        self.shared_list.append(True)
        self.model_loaded_event = Event()
        self.worker_sync_barrier = Barrier(flags.worker_count)
        for i in range(flags.worker_count):
            process = mp.Process(target=start_worker, args=(i, self.worker_data_queue, self.shared_list, flags, self.model_loaded_event, self.worker_sync_barrier, verbose))
            process.start()
            self.workers.append(process)
        self.expected_worker_id = 0
        self.worker_data_pos_buf = [None for i in range(flags.worker_count)]
        self.model_update_queue = Queue()
        self.last_model_get_time = time.time()
        self.model_update_wait_timer = Timer("Manager model update time ", 120, 1,
                                             "Manager is waiting for model update for quite some time = {:0.4f} seconds avg({:0.4f})! Try lowering replay caching setting.")

    def plan_and_execute_workers(self):
        prepared_data = []
        if self.flags.reproducible:
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
        else:
            worker_data = self.worker_data_queue.get()
            prepared_data.append(worker_data)
        return prepared_data

    def pre_processing(self):
        if self.training_event.is_set() and self.flags.reproducible:

            with self.model_update_wait_timer:
                model_dict = self.model_update_queue.get()
            self.shared_list[0] = model_dict
            self.model_update_queue.task_done()
            if self.model_update_queue.unfinished_tasks >= 2 and (self.last_model_get_time - time.time()) > 2 * 60:
                self.last_model_get_time = time.time()
                logger.warning(
                    f"Worker model queue seems to be rising = {self.model_update_queue.unfinished_tasks}! Beware for running out of memory. Try bigger replay caching setting.")

        if self.flags.reproducible:
            self.model_loaded_event.set()

    def update_model_data(self, current_model):
        if self.flags.reproducible:
            self.model_update_queue.put({k: v.cpu() for k, v in current_model.state_dict().items()})
        else:
            self.shared_list[0] = {k: v.cpu() for k, v in current_model.state_dict().items()}

    def clean_up(self):
        self.shared_list[1] = False
        super(NativeWorkerManager, self).clean_up()
        while not self.worker_data_queue.empty():
            self.worker_data_queue.get()
            time.sleep(0.1)
        if self.flags.reproducible:
            self.model_loaded_event.set()
        for i in range(len(self.workers)):
            self.workers[i].join()
        self.shared_list[:] = []
        if self.flags.reproducible:
            self.model_update_wait_timer.save_stats(self.file_save_url)

    def reset(self):
        pass

