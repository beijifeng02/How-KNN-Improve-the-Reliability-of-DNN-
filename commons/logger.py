import os
import numpy as np
import pandas as pd


class logger:
    def __init__(self, cfg):
        self.experiment = cfg.TEST.EXPERIMENT
        self.data = None
        self.atypicality = None
        dir = cfg.TEST.OUTPUT_DIR
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.atyp_file = dir + "atypicality.csv"
        self.data_file = dir + "data.csv"

    def update(self, atypicality, data):
        self.atypicality = atypicality
        self.data = data

    def write(self):
        np.savetxt(self.atyp_file, self.atypicality, delimiter=',')
        self.data.to_csv(self.data_file, index=False)
