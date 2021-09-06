from agent.manager.worker_manager_base import WorkerManagerBase
from agent.worker.native_rollout_worker import start_worker
import torch.multiprocessing as mp
from torch.multiprocessing import Queue, Manager


class NativeWorkerManager(WorkerManagerBase):
    def __init__(self, async, stop_event, replay_writer, replay_buffers, options_flags, observation_shape, actions_count, model, stats):
        super().__init__(async, stop_event, replay_writer, replay_buffers, stats)
        self.workers = []
        self.worker_data_queue = Queue(maxsize=10)
        self.shared_list = Manager().list()
        self.shared_list.append({k: v.cpu() for k, v in model.state_dict().items()})
        for i in range(options_flags.actor_count):
            process = mp.Process(target=start_worker, args=(options_flags, observation_shape, actions_count, i, self.worker_data_queue, self.shared_list))
            process.start()
            self.workers.append(process)

    def plan_and_execute_workers(self):
        # TODO IMPLEMENT ASYNC VERSION

        worker_data = self.worker_data_queue.get()
        return [worker_data]

    def pre_processing(self):
        pass

    def update_model_data(self, current_model):
        self.shared_list[0] = {k: v.cpu() for k, v in current_model.state_dict().items()}

    def clean_up(self):
        for i in range(len(self.workers)):
            self.workers[i].close()
            self.workers[i].join()


