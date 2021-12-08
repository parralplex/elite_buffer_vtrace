import time
import ray
from queue import Queue

from agent.manager.abstract.ray_manager import RayManager
from stats.prof_timer import Timer
from utils import logger


class RayManagerSync(RayManager):
    def __init__(self, stop_event, training_event, replay_writer, replay_buffers, model, stats, flags, file_save_url, verbose=False):
        super().__init__(stop_event, training_event, replay_writer, replay_buffers, model, stats, flags, file_save_url, verbose)
        self.model_update_queue = Queue()
        self.last_model_get_time = time.perf_counter()
        self.model_update_wait_timer = Timer("Manager model update time ", 120, 1,
                                             "Manager is waiting for model update for quite some time = {:0.4f} seconds avg({:0.4f})! Try lowering replay caching setting.")
        self.model_dict_send_counter = 0

    def plan_and_execute_workers(self):
        self.done_ref, self.rollouts = ray.wait(self.rollouts, num_returns=len(self.workers))
        return super(RayManagerSync, self).plan_and_execute_workers()

    def pre_processing(self):
        if self.training_event.is_set():
            counter_start = self.model_dict_send_counter
            counter_stop = self.replay_buffers[0].output_cache_counter
            for i in range(counter_start, counter_stop):
                self.model_dict_send_counter += 1
                with self.model_update_wait_timer:
                    self.model_dict_ref = self.model_update_queue.get()
                self.model_update_queue.task_done()
                if self.stop_event.is_set():
                    return
            update_rollouts = [self.workers[i].load_model.remote(self.model_dict_ref) for i in range(len(self.workers))]
            ray.wait(update_rollouts, num_returns=len(self.workers))

        if self.active_worker_indices.qsize() == len(self.workers):
            super(RayManagerSync, self).pre_processing()
        else:
            raise Exception("Application ahs entered forbidden state - not all workers has finished execution even though return data processing has started!")

    def update_model_data(self, current_model):
        self.model_update_queue.put({k: v.cpu() for k, v in current_model.state_dict().items()})

    def clean_up(self):
        super(RayManagerSync, self).clean_up()
        self.model_update_wait_timer.save_stats(self.file_save_url)
