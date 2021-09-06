import ray

from agent.worker.ray_rollout_worker import RayRolloutWorker
from agent.manager.worker_manager_base import WorkerManagerBase


class RayWorkerManager(WorkerManagerBase):
    def __init__(self, async, stop_event, replay_writer, replay_buffers, options_flags, observation_shape, actions_count, training_event, model, stats):
        super().__init__(async, stop_event, replay_writer, replay_buffers, stats)
        self.workers = []

        for i in range(options_flags.actor_count):
            self.workers.append(RayRolloutWorker.remote(options_flags, observation_shape, actions_count, i))
        self.done_ref = None
        self.rollouts = []
        self.training_event = training_event
        self.model_dict_ref = ray.put({k: v.cpu() for k, v in model.state_dict().items()})
        self.active_worker_indices = [i for i in range(len(self.workers))]

    def plan_and_execute_workers(self):
        if self.async:
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
        for j in range(len(self.active_worker_indices)):
            if self.training_event.is_set() or self.async:
                self.rollouts.extend([self.workers[self.active_worker_indices[j]].performing.remote(self.model_dict_ref, update=True)])
            else:
                self.rollouts.extend([self.workers[self.active_worker_indices[j]].performing.remote()])
        self.active_worker_indices = []

    def update_model_data(self, current_model):
        self.model_dict_ref = ray.put({k: v.cpu() for k, v in current_model.state_dict().items()})
