import os
import pandas as pd

from darts.meters.average import AverageMeter


class MultitaskAccuracyMeter:

    def __init__(self, tasks):
        self.tasks = tasks
        self.reset()

    def reset(self):
        self.meters = self.create_meters()

    def create_meters(self):
        """ Create an average meter for each task """
        meters = {}
        for task, _ in self.tasks.items():
            meters[task] = AverageMeter('Acc@1', ':6.2f')
        return meters

    def get_avg_accuracy(self, task):
        return self.meters[task].avg

    def get_accuracy(self, task):
        return self.meters[task].val

    def update(self, accuracies, batch_size):
        for task, acc in accuracies.items():
            self.meters[task].update(acc[0].item(), batch_size)

    def dataframe(self):
        """ Get a dataframe of all task accuracies """
        avg_accuracy = {k: v.avgs for (k, v) in self.meters.items()}
        return pd.DataFrame(avg_accuracy)

    def save(self, path, filename):
        """ Save the task accuracies as a csv """
        path = os.path.join(path, f'{filename}_accuracy.csv')
        self.dataframe().to_csv(path, index=False)
