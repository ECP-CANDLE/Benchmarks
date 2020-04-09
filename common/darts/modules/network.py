from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from darts.api import Model
from darts.modules import Cell
from darts.modules.classifier import MultitaskClassifier
from darts.genotypes import Genotype


class Hyperparameters:
    c = 1
    num_nodes = 2
    num_cells = 3
    channel_multiplier = 1


class Network(Model):
    """ Collection of cells """

    def __init__(self,
                 stem: nn.Module,
                 cell_dim: int,
                 primitives: List[str],
                 ops: Dict[],
                 tasks: Dict[str, int],
                 criterion,
                 device: str = 'cpu',
                 hyperparams=Hyperparameters()):
        super(Network, self).__init__()
        self.primitives = primitives
        self.ops = ops
        self.cell_dim = cell_dim
        self.tasks = tasks
        self.criterion = criterion
        self.device = device
        self.num_cells = hyperparams.num_cells
        self.num_nodes = hyperparams.num_nodes

        self.stem = stem

        # c_curr means a factor of the output channels of current cell
        c_curr = cell_dim * hyperparams.channel_multiplier * hyperparams.c
        cpp, cp, c_curr = c_curr, c_curr, self.c
        self.cells = nn.ModuleList()
        for i in range(hyperparams.num_cells):

            cell = Cell(
                hyperparams.num_nodes,
                hyperparams.channel_multiplier,
                cpp,
                cp,
                c_curr
            ).to(self.device)

            self.cells += [cell]

        self.classifier = MultitaskClassifier(cell_dim, tasks)

        # k is the total number of edges inside single cell, 14
        k = sum(1 for i in range(self.num_nodes) for j in range(2 + i))
        num_ops = len(self.primitives)

        self.alpha_normal = nn.Parameter(torch.randn(k, num_ops))

        with torch.no_grad():
            # initialize to smaller value
            self.alpha_normal.mul_(1e-3)

        self._arch_parameters = [
            self.alpha_normal,
        ]

    def new(self):
        """ Create a new model initialzed with current alpha parameters.

        Weights are left untouched.

        Returns
        -------
        model : Network
            New model initialized with current alpha.
        """
        model = Network(
            self.stem,
            self.cell_dim,
            self.primitives,
            self.ops,
            self.tasks,
            self.criterion
        ).to(self.device)

        for x, y in zip(model.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)

        return model

    def forward(self, x):
        # s0 & s1 means the last cells' output
        s0 = s1 = self.stem(x) # [b, 3, 32, 32] => [b, 48, 32, 32]

        for i, cell in enumerate(self.cells):
            weights = F.softmax(self.alpha_normal, dim=-1) # [14, 8]
            # execute cell() firstly and then assign s0=s1, s1=result
            s0, out = s1, cell(s0, s1, weights) # [40, 64, 32, 32]

        logits = self.classifier(out.view(out.size(0), -1))

        return logits

    def loss(self, data, target, reduce='mean'):
        """ Calculate a value of loss function """
        logits = self(data)

        for task, logit in logits.items():
            logits[task] = logit.to(self.device)

        losses = {}
        for task, label in target.items():
            label = label.to(self.device)
            losses[task] = self.criterion(logits[task], label)

        if reduce:
            total = 0
            for _, value in losses.items():
                total += value

            if reduce == "mean":
                losses = total / len(losses)
            elif reduce == "sum":
                losses = total
            else:
                raise ValueError('Reduced loss must use either `mean` or `sum`!')

        return losses

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        """
        :return:
        """
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self.num_nodes): # for each node
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), # i+2 is the number of connection for node i
                            key=lambda x: -max(W[x][k] # by descending order
                                               for k in range(len(W[x])) # get strongest ops
                                               if k != self.primitives.index('none'))
                               )[:2] # only has two inputs
                for j in edges: # for every input nodes j of current node i
                    k_best = None
                    for k in range(len(W[j])): # get strongest ops for current input j->i
                        if k != self.primitives.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((self.primitives[k_best], j)) # save ops and input node
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alpha_normal, dim=-1).data.cpu().numpy())
        concat = range(2 + self.num_nodes - self.channel_multiplier, self.num_nodes + 2)

        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_normal, reduce_concat=concat
        )

        return genotype

