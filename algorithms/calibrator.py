import torch
import torch.nn as nn
import torch.optim as optim

from .base import BaseCalibrator
from .ECELoss import _ECELoss


class TemperatureScaling(BaseCalibrator):
    def __init__(self, cfg):
        super().__init__(cfg)
        temperature = cfg.PREPROCESS.TEMPERATURE
        self.train_bool = False if temperature is not None else True
        self.temperature = nn.Parameter(torch.tensor([temperature]).cuda()) if temperature is not None else (
            nn.Parameter(torch.tensor([1.5]).cuda()))

    def train(self, logits, labels):
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

    def forward(self, logits, softmax=True):
        if softmax:
            softmax = nn.Softmax(dim=1)
            return softmax(logits / self.temperature)

        return logits / self.temperature
