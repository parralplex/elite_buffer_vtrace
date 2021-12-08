import time

from agent.manager.abstract.worker_manager_base import WorkerManagerBase
from torch.multiprocessing import Manager


class NativeManager(WorkerManagerBase):
    def __init__(self, stop_event, training_event, replay_writer, replay_buffers, model, stats, flags, file_save_url):
        super().__init__(stop_event, training_event, replay_writer, replay_buffers, stats, flags, file_save_url)
        self.workers = []
        self.manager = Manager()
        self.worker_data_queue = self.manager.Queue(maxsize=flags.shared_queue_size)
        self.shared_list = self.manager.list()
        self.shared_list.append({k: v.cpu() for k, v in model.state_dict().items()})
        self.shared_list.append(True)

    def plan_and_execute_workers(self):
        raise NotImplementedError

    def pre_processing(self):
        pass

    def update_model_data(self, current_model):
        pass

    def clean_up(self):
        self.shared_list[1] = False
        super(NativeManager, self).clean_up()
        while not self.worker_data_queue.empty():
            self.worker_data_queue.get()
            time.sleep(0.1)

    def reset(self):
        pass

