import ray

from agent.worker.ray_rollout_worker import RayRolloutWorker
from agent.manager.abstract.worker_manager_base import WorkerManagerBase
from model.network import StateTransformationNetwork
from queue import Queue


class RayManager(WorkerManagerBase):
    def __init__(self, stop_event, training_event, replay_writer, replay_buffers, model, stats, flags, file_save_url, verbose=False):
        super().__init__(stop_event, training_event, replay_writer, replay_buffers, stats, flags, file_save_url)
        if not ray.is_initialized():
            if flags.debug:
                ray.init(local_mode=True)
            else:
                ray.init()
        self.workers = []

        self.state_transf_network = StateTransformationNetwork(self.flags)

        for i in range(flags.worker_count):
            self.workers.append(RayRolloutWorker.remote(i, flags, self.state_transf_network.state_dict(), file_save_url, verbose))

        self.done_ref = None
        self.rollouts = []
        self.training_event = training_event

        self.model_dict_ref = ray.put({k: v.cpu() for k, v in model.state_dict().items()})
        update_rollouts = [self.workers[i].load_model.remote(self.model_dict_ref) for i in range(len(self.workers))]
        ray.wait(update_rollouts, num_returns=len(self.workers))

        self.active_worker_indices = Queue()
        for i in range(len(self.workers)):
            self.active_worker_indices.put(i)

    def plan_and_execute_workers(self):
        worker_handler = ray.get(self.done_ref)

        workers_data = []
        for j in range(len(self.done_ref)):
            worker_buffers, worker_index, rewards, ep_steps = worker_handler[j]
            self.active_worker_indices.put(worker_index)
            workers_data.append([worker_buffers, worker_index, rewards, ep_steps])
        return workers_data

    def pre_processing(self):
        if self.active_worker_indices.qsize() == len(self.workers):
            for j in range(self.active_worker_indices.qsize()):
                self.rollouts.extend([self.workers[self.active_worker_indices.get()].exec_and_eval_rollout.remote()])

    def update_model_data(self, current_model):
        pass

    def clean_up(self):
        super(RayManager, self).clean_up()

    def reset(self):
        for worker in self.workers:
            ray.kill(worker)
