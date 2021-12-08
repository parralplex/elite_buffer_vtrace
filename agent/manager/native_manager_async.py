from agent.manager.abstract.native_manager import NativeManager
from agent.worker.native_rollout_worker import start_worker_async
import torch.multiprocessing as mp

from model.network import StateTransformationNetwork


class NativeManagerAsync(NativeManager):
    def __init__(self, stop_event, training_event, replay_writer, replay_buffers, model, stats, flags, file_save_url, verbose=False):
        super().__init__(stop_event, training_event, replay_writer, replay_buffers, model, stats, flags, file_save_url)
        self.state_transf_network = StateTransformationNetwork(self.flags)
        for i in range(flags.worker_count):
            process = mp.Process(target=start_worker_async, args=(
            i, self.worker_data_queue, self.shared_list, flags, self.state_transf_network.state_dict(), file_save_url, verbose))
            process.start()
            self.workers.append(process)

    def plan_and_execute_workers(self):
        return [self.worker_data_queue.get()]

    def update_model_data(self, current_model):
        self.shared_list[0] = {k: v.cpu() for k, v in current_model.state_dict().items()}
        self.replay_writer.current_model_dict = {k: v.cpu() for k, v in current_model.state_dict().items()}

    def clean_up(self):
        super(NativeManagerAsync, self).clean_up()
        for i in range(len(self.workers)):
            self.workers[i].join()
        self.shared_list[:] = []
