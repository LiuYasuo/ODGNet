import numpy as np


class EarlyStop:
    def __init__(self, patience, min_is_best):
        self.patience = patience
        self.min_is_best = min_is_best

        self.count, self.cur_values = None, None
        self.reset()

    def reset(self):
        self.count = 0
        finfo = np.finfo(np.float32)
        self.best_value = finfo.max if self.min_is_best else finfo.min

    def reach_stop_criteria(self, cur_value):
        if self.min_is_best:
            update_best = cur_value <= self.best_value
        else:
            update_best = cur_value >= self.best_value

        if update_best:
            self.best_value = cur_value
            self.count = 0
        else:
            self.count += 1

        if self.count == self.patience:
            return True

        return False
