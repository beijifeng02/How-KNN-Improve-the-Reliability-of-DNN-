import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import softmax

from .ECELoss import _ECELoss


class TemperatureScaling:
    def __init__(self):
        self.temperature = nn.Parameter(torch.tensor([1.5]).cuda())

    def fit(self, logits, labels):
        logits = torch.tensor(logits)
        labels = torch.tensor(labels).long()
        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = _ECELoss()

        ece_before = ece_criterion(logits, labels)
        print("ece_before: %.4f" % ece_before.item())

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        print('Optimal temperature: %.3f' % self.temperature.item())
        out = logits / self.temperature
        ece_after = ece_criterion(out, labels)
        print("ece_after: %.4f" % ece_after.item())
        return ece_before, ece_after

    def calibrate(self, logits, softmax_bool=True):
        if softmax_bool:
            return softmax(logits / self.temperature)

        return logits / self.temperature
