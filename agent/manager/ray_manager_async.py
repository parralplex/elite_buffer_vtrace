import ray

from agent.manager.abstract.ray_manager import RayManager


class RayManagerAsync(RayManager):
    def __init__(self, stop_event, training_event, replay_writer, replay_buffers, model, stats, flags, file_save_url, verbose=False):
        super().__init__(stop_event, training_event, replay_writer, replay_buffers, model, stats, flags, file_save_url, verbose)

    def plan_and_execute_workers(self):
        self.done_ref, self.rollouts = ray.wait(self.rollouts, num_returns=1)
        return super(RayManagerAsync, self).plan_and_execute_workers()

    def update_model_data(self, current_model):
        self.model_dict_ref = ray.put({k: v.cpu() for k, v in current_model.state_dict().items()})
        update_rollouts = [self.workers[i].load_model.remote(self.model_dict_ref) for i in range(len(self.workers))]
        ray.wait(update_rollouts, num_returns=len(self.workers))
