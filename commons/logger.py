class logger:
    def __init__(self, cfg):
        self.experiment = cfg.TEST.EXPERIMENT
        self.data = None

    def update(self, data):
        self.data = data
