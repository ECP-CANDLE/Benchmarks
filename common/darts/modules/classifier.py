from typing import Dict
import torch.nn as nn


class MultitaskClassifier(nn.Module):

    def __init__(self, input_dim: int, tasks: Dict[str, int]):
        super(MultitaskClassifier, self).__init__()
        self.tasks = tasks

        for task, num_classes in tasks.items():
            self.add_module(
                task,
                nn.Linear(input_dim, num_classes)
            )

    def num_classes(self, task):
        """ Get number of classes for a task. """
        return self.tasks[task]

    def forward(self, x):
        logits = {}
        for task, _ in self.tasks.items():
            logits[task] = self._modules[task](x)

        return logits
