import os
import pandas as pd

from darts.meters.average import AverageMeter
from darts.meters.accuracy import MultitaskAccuracyMeter


class EpochMeter:
    """ Track epoch loss and accuracy """

    def __init__(self, tasks, name='train'):
        self.name = name
        self.loss_meter = AverageMeter(name)
        self.acc_meter = MultitaskAccuracyMeter(tasks)
        self.reset()

    def reset(self):
        self.loss = []
        self.acc = {task: [] for task, _ in self.acc_meter.meters.items()}

    def update_batch_loss(self, loss, batch_size):
        self.loss_meter.update(loss, batch_size)

    def update_batch_accuracy(self, acc, batch_size):
        self.acc_meter.update(acc, batch_size)

    def update_epoch(self):
        self.loss.append(self.loss_meter.avg)
        for task, acc in self.acc_meter.meters.items():
            self.acc[task].append(acc.avg)

    def dataframe(self):
        results = self.acc
        results['loss'] = self.loss
        return pd.DataFrame(results)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, f'{self.name}_epoch_results')
        self.dataframe().to_csv(path, index=False)
