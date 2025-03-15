from torchmetrics.aggregation import RunningMean


class StreamingLossMeter(object):
    r"""A streaming meter for training losses"""

    def __init__(self, n_metrics, window_size):
        self.n_metrics = n_metrics
        self.meters = []
        for _ in range(n_metrics):
            self.meters.append(RunningMean(window=window_size))
        self._counter = 0

    def to(self, device, **kwargs):
        for meter in self.meters:
            meter.to(device=device, **kwargs)
        return self

    def update(self, *losses):
        for i, loss in enumerate(losses):
            self.meters[i].update(loss)
        self._counter += 1

    def compute(self):
        return [meter.compute() if self._counter else None for meter in self.meters]
