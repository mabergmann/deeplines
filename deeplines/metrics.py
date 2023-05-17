import numpy as np
from scipy.optimize import linear_sum_assignment

from . import utils


class MetricAccumulator(object):
    correct_threshold = 20

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, predicted_batch, true_batch):
        for y_true, y_pred in zip(true_batch, predicted_batch):
            prediction_len = len(y_pred)
            gt_len = len(y_true)

            if prediction_len == 0:
                self.fn += gt_len
                continue

            distances = utils.get_distance_between_lines(y_true, y_pred)

            graph = np.zeros((gt_len, prediction_len))
            graph[distances <= self.correct_threshold] = 1
            row_ind, col_ind = linear_sum_assignment(-graph)

            pair_nums = graph[row_ind, col_ind].sum()

            tp = pair_nums
            fp = prediction_len - pair_nums
            fn = gt_len - pair_nums

            self.tp += tp
            self.fp += fp
            self.fn += fn

    def get_precision(self):
        if self.tp + self.fp == 0:
            return 1
        return self.tp / (self.tp + self.fp)

    def get_recall(self):
        if self.tp + self.fn == 0:
            return 1
        return self.tp / (self.tp + self.fn)

    def get_f1(self):
        recall = self.get_recall()
        precision = self.get_precision()
        if precision > 0 and recall > 0:
            f1_score = 2 / ((1 / precision) + (1 / recall))
        else:
            f1_score = 0
        return f1_score

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
