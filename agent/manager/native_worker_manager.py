
from agent.manager.worker_manager_base import WorkerManagerBase
from agent.worker.native_rollout_worker import start_worker
import torch.multiprocessing as mp
from torch.multiprocessing import Manager


class NativeWorkerManager(WorkerManagerBase):
    def __init__(self, stop_event, training_event, replay_writer, replay_buffers, model, stats, flags, verbose=False):
        super().__init__(stop_event, training_event, replay_writer, replay_buffers, stats, flags)
        self.workers = []
        self.manager = Manager()
        self.worker_data_queue = self.manager.Queue(maxsize=flags.shared_queue_size)
        self.shared_list = self.manager.list()
        self.shared_list.append({k: v.cpu() for k, v in model.state_dict().items()})
        self.shared_list.append(True)
        for i in range(flags.worker_count):
            process = mp.Process(target=start_worker, args=(i, self.worker_data_queue, self.shared_list, flags, verbose))
            process.start()
            self.workers.append(process)

    def plan_and_execute_workers(self):
        # TODO IMPLEMENT SYNC VERSION
        worker_data = self.worker_data_queue.get()
        return [worker_data]

    def pre_processing(self):
        pass

    def update_model_data(self, current_model):
        self.shared_list[0] = {k: v.cpu() for k, v in current_model.state_dict().items()}

    def clean_up(self):
        self.shared_list[1] = False
        super(NativeWorkerManager, self).clean_up()
        while not self.worker_data_queue.empty():
            self.worker_data_queue.get()
        for i in range(len(self.workers)):
            self.workers[i].join()
        self.shared_list[:] = []

