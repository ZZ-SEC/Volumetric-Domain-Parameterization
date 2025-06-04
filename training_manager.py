import numpy as np
from copy import deepcopy

import torch.nn


def check_update(tm):
    loss_terms = tm.loss_terms
    N_iter = len(loss_terms)
    weight = tm.weights
    best_idx = tm.best_idx
    loss_terms_best = loss_terms[best_idx]
    loss_terms_current = loss_terms[-1]
    loss_current = sum([loss_terms_current[i] * weight[i] for i in range(len(weight))])
    loss_best = sum([loss_terms_best[i] * weight[i] for i in range(len(weight))])

    if N_iter == 1:
        return True
    if loss_current < loss_best:
        return True
    return False


class TrainingManager():

    def __init__(self, net, loss_weights: dict, check_update=check_update, max_iter=-1):
        self.net = net
        loss_names = list(loss_weights)
        weights = [loss_weights[name] for name in loss_names]
        self.loss_names = loss_names
        self.weights = np.array(weights, dtype=np.float64)
        self.loss_terms = []
        self.loss = []
        self.best_idx = -1
        self.update = check_update
        self.status = []
        self.best_model = None
        self.max_iter = max_iter

    def record(self, list_loss, status_save=[0], print_iter=5):
        self.status.append(status_save)
        list_loss_num = [item.item() for item in list_loss]
        loss_sum = 0
        for i in range(len(self.loss_names)):
            loss_sum += list_loss[i] * self.weights[i]
        self.loss.append(loss_sum.item())
        self.loss_terms.append(list_loss_num)
        update = self.update(self)
        if update:
            self.best_idx = len(self.loss) - 1
            if isinstance(self.net, torch.nn.Module):
                self.best_model = deepcopy(self.net.state_dict())
            elif isinstance(self.net, list):
                self.best_model = deepcopy(self.net)
        N_iter = len(self.loss) - 1
        print_info = False
        if print_iter >= 1:
            if N_iter % print_iter == 0 or N_iter == self.max_iter - 1:
                self.print_iter()
                print_info = True

        return loss_sum, print_info

    def print_iter(self):
        best_idx = self.best_idx
        str_list = ["\t\t%5d iters, loss = %.08f / %.08f, \t" % (len(self.loss) - 1, self.loss[-1], self.loss[best_idx])]
        str_list.append("%6d / %6d" % (self.status[-1][0], self.status[self.best_idx][0]))
        N = len(self.loss_names)
        for i in range(N):
            if i % 4 == 0:
                str_list.append("\n\t\t\t\t\t\t\t\t\t")
            str_list.append("%6s = %.08f (%5.02f%%) / %.08f (%5.02f%%), " %
                            (self.loss_names[i], self.loss_terms[-1][i], self.loss_terms[-1][i] * self.weights[i] * 100 / self.loss[-1],
                             self.loss_terms[best_idx][i], self.loss_terms[best_idx][i] * self.weights[i] * 100 / self.loss[best_idx]))
        str_out = "".join(str_list)
        print(str_out)
