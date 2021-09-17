import time
import ray

from agent.worker.ray_rollout_worker import RayRolloutWorker
from agent.manager.worker_manager_base import WorkerManagerBase
from queue import Queue

from stats.prof_timer import Timer
from utils import logger


class RayWorkerManager(WorkerManagerBase):
    def __init__(self, stop_event, training_event, replay_writer, replay_buffers, model, stats, flags, file_save_url, verbose=False):
        super().__init__(stop_event, training_event, replay_writer, replay_buffers, stats, flags, file_save_url)
        if not ray.is_initialized():
            ray.init()
        self.workers = []

        for i in range(flags.worker_count):
            self.workers.append(RayRolloutWorker.remote(i, flags, verbose))
        self.done_ref = None
        self.rollouts = []
        self.training_event = training_event
        self.model_dict_ref = ray.put({k: v.cpu() for k, v in model.state_dict().items()})
        self.active_worker_indices = [i for i in range(len(self.workers))]
        self.reproducibility_index = 0
        self.model_update_queue = Queue()
        self.model_update_queue.put(self.model_dict_ref)
        self.last_model_get_time = time.time()
        self.model_update_wait_timer = Timer("Manager model update time ", 120, 1, "Manager is waiting for model update for quite some time = {:0.4f} seconds avg({:0.4f})! Try lowering replay caching setting.")

    def plan_and_execute_workers(self):
        if not self.flags.reproducible:
            self.done_ref, self.rollouts = ray.wait(self.rollouts, num_returns=1)
        else:
            self.done_ref, self.rollouts = ray.wait(self.rollouts, num_returns=len(self.workers))

        worker_handler = ray.get(self.done_ref)

        workers_data = []
        for j in range(len(self.done_ref)):
            worker_buffers, worker_index, rewards, ep_steps = worker_handler[j]
            self.active_worker_indices.append(worker_index)
            workers_data.append([worker_buffers, worker_index, rewards, ep_steps])
        return workers_data

    def pre_processing(self):
        if self.training_event.is_set() and self.flags.reproducible:
            with self.model_update_wait_timer:
                model_dict = self.model_update_queue.get()
            self.model_update_queue.task_done()
            if self.model_update_queue.unfinished_tasks >= 2 and (self.last_model_get_time - time.time()) > 2 * 60:
                self.last_model_get_time = time.time()
                logger.warning(f"Worker model queue seems to be rising = {self.model_update_queue.unfinished_tasks}! Beware for running out of memory. Try bigger replay caching setting.")
        else:
            model_dict = self.model_dict_ref
        for j in range(len(self.active_worker_indices)):
            if self.training_event.is_set() or self.flags.reproducible:
                self.rollouts.extend([self.workers[self.active_worker_indices[j]].performing.remote(model_dict, update=True)])
            else:
                self.rollouts.extend([self.workers[self.active_worker_indices[j]].performing.remote()])
        self.active_worker_indices = []

    def update_model_data(self, current_model):
        if self.flags.reproducible:
            self.model_update_queue.put({k: v.cpu() for k, v in current_model.state_dict().items()})
        else:
            self.model_dict_ref = ray.put({k: v.cpu() for k, v in current_model.state_dict().items()})

    def clean_up(self):
        super(RayWorkerManager, self).clean_up()
        if self.flags.reproducible:
            self.model_update_wait_timer.save_stats(self.file_save_url)

    def reset(self):
        for worker in self.workers:
            ray.kill(worker)
