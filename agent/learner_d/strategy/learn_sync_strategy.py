from agent.learner_d.strategy.learn_iter_strategy import LearnIterStrategyBase
from threading import Condition


class LearnSyncStrategy(LearnIterStrategyBase):
    def __init__(self, worker_manager, stop_event, flags):
        super().__init__(worker_manager, stop_event, flags)
        self.data_pos_pointer = 0
        self.in_row_condition = Condition()
        self.condition_waiters = 0

    def after_batching(self, **kwargs):
        counter = kwargs['counter']

        def is_ready_for_processing():
            nonlocal counter
            if counter < self.data_pos_pointer:
                raise ValueError("COunter cannot be lower that global counter")
            return counter == self.data_pos_pointer

        if self.condition_waiters == (self.flags.learner_thread_count - 1):
            with self.in_row_condition:
                self.in_row_condition.notify(1)

        if counter > self.data_pos_pointer:
            with self.in_row_condition:
                self.condition_waiters += 1
                self.in_row_condition.wait_for(is_ready_for_processing)
                self.condition_waiters -= 1

        if self.stop_event.is_set():
            return False
        return True

    def before_learning(self):
        self.data_pos_pointer = (self.data_pos_pointer + 1) % self.flags.max_cache_pos_pointer
        if self.condition_waiters > 0:
            with self.in_row_condition:
                self.in_row_condition.notify(1)

    def after_learning(self, **kwargs):
        # if self.flags.reproducible and kwargs['training_iteration'] % self.flags.replay_out_cache_size == 0:
        self.worker_manager.update_model_data(kwargs['model'])

    def clean_up(self, model):
        super(LearnSyncStrategy, self).clean_up(model)
        with self.in_row_condition:
            self.in_row_condition.notify_all()
