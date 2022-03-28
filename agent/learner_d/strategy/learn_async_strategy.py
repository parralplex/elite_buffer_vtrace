from agent.learner_d.strategy.learn_iter_strategy import LearnIterStrategyBase
from utils.logger import logger


class LearnAsyncStrategy(LearnIterStrategyBase):
    def __init__(self, worker_manager, stop_event, flags):
        super().__init__(worker_manager, stop_event, flags)

    def after_batching(self, **kwargs):
        if kwargs['states'].shape[1] != kwargs['current_batch_size']:
            logger.exception(" BAD BATCH SIZE ", kwargs['states'].shape, kwargs['current_batch_size'])
            return False
        return True

    def before_learning(self):
        pass

    def after_learning(self, **kwargs):
        self.worker_manager.update_model_data(kwargs['model'])
