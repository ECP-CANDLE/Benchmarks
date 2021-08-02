from pytorch_lightning.metrics.classification import F1


class F1Meter:

    def __init__(self, tasks, average='micro'):
        self.metrics = self._create_metrics(tasks, average)

    def _create_metrics(self, tasks, avg):
        """Create F1 metrics for each of the tasks

        Args:
            tasks: dictionary of tasks and their respective number
                of classes
            avg: either 'micro' or 'macro'
        """
        return {t: F1(c, average=avg) for t, c in tasks.items()}

    def f1(self, y_hat, y):
        """Get the batch F1 score"""
        scores = {}
        for task, pred in y_hat.items():
            scores[task] = self.metrics[task](pred, y[task])

        return scores

    def compute(self):
        """Compute the F1 score over all batches"""
        return {t: f1.compute().item() for t, f1 in self.metrics.items()}
