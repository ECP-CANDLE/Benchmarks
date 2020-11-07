class AccuracyMeter:

    def __init__(self, tasks, dataloader):
        self.tasks = tasks
        self.loader = dataloader
        self.reset()

    def reset(self):
        self.correct = {task: 0 for task, _ in self.tasks.items()}
        self.accuracies = {}

    def update(self, logits, target):
        for task, logit in logits.items():
            pred = logit.argmax(dim=1, keepdim=True)
            correct = pred.eq(target[task].view_as(pred)).sum().item()
            self.correct[task] += correct

    def update_accuracy(self):
        for task, correct in self.correct.items():
            acc = 100. * correct / len(self.loader.dataset)
            self.accuracies[task] = acc

    def get_accuracy(self):
         return self.accuracies

    def print_task_accuracies(self):
        for task, acc in self.accuracies.items():
            print(f'\t{task}: {acc}')
