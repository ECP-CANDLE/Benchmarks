import torch
import torch.nn as nn
import torch.nn.functional as F

from darts.api import Model
from darts.modules.conv import Cell
from darts.modules.classifier import MultitaskClassifier
from darts.genotypes import PRIMITIVES, Genotype


class Hyperparameters:
    c = 8
    num_nodes = 2
    num_cells = 3
    channel_multiplier = 2
    stem_channel_multiplier = 2
    num_embeddings = 35095  # vocab size
    embedding_dim = 1500


class ConvNetwork(Model):
    """ Collection of cells """

    def __init__(self, tasks, criterion, device='cpu', hyperparams=Hyperparameters()):
        super(ConvNetwork, self).__init__()
        self.tasks = tasks
        self.criterion = criterion
        self.device = device
        self.c = hyperparams.c
        self.num_cells = hyperparams.num_cells
        self.num_nodes = hyperparams.num_nodes
        self.channel_multiplier = hyperparams.channel_multiplier

        # stem_multiplier is for stem network,
        # and multiplier is for general cell
        c_curr = hyperparams.stem_channel_multiplier * self.c  # 3*16
        # stem network, convert 3 channel to c_curr
        self.stem = nn.Sequential(
            nn.Embedding(
                num_embeddings=hyperparams.num_embeddings,
                embedding_dim=hyperparams.embedding_dim
            ),
            nn.Conv1d(hyperparams.embedding_dim, c_curr, 3, padding=1, bias=False),
            nn.BatchNorm1d(c_curr)
        ).to(self.device)

        # c_curr means a factor of the output channels of current cell
        # output channels = multiplier * c_curr
        cpp, cp, c_curr = c_curr, c_curr, self.c
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(hyperparams.num_cells):

            # for layer in the middle [1/3, 2/3], reduce via stride=2
            if i in [hyperparams.num_cells // 3, 2 * hyperparams.num_cells // 3]:
                c_curr *= 2
                reduction = True
            else:
                reduction = False

            # [cp, h, h] => [multiplier*c_curr, h/h//2, h/h//2]
            # the output channels = multiplier * c_curr
            cell = Cell(
                hyperparams.num_nodes,
                hyperparams.channel_multiplier,
                cpp,
                cp,
                c_curr,
                reduction,
                reduction_prev
            ).to(self.device)
            # update reduction_prev
            reduction_prev = reduction
            self.cells += [cell]
            cpp, cp = cp, hyperparams.channel_multiplier * c_curr

        # adaptive pooling output size to 1x1
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        # since cp records last cell's output channels
        # it indicates the input channel number
        self.classifier = MultitaskClassifier(cp, tasks)

        # k is the total number of edges inside single cell, 14
        k = sum(1 for i in range(self.num_nodes) for j in range(2 + i))
        num_ops = len(PRIMITIVES)  # 8

        self.alpha_normal = nn.Parameter(torch.randn(k, num_ops))
        self.alpha_reduce = nn.Parameter(torch.randn(k, num_ops))

        with torch.no_grad():
            # initialize to smaller value
            self.alpha_normal.mul_(1e-3)
            self.alpha_reduce.mul_(1e-3)

        self._arch_parameters = [
            self.alpha_normal,
            self.alpha_reduce,
        ]

    def new(self):
        """ Create a new model initialzed with current alpha parameters.

        Weights are left untouched.

        Returns
        -------
        model : Network
            New model initialized with current alpha.
        """
        model = ConvNetwork(
            self.tasks,
            self.criterion
        ).to(self.device)

        for x, y in zip(model.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)

        return model

    def forward(self, x):
        """
        in: torch.Size([3, 3, 32, 32])
        stem: torch.Size([3, 48, 32, 32])
        cell: 0 torch.Size([3, 64, 32, 32]) False
        cell: 1 torch.Size([3, 64, 32, 32]) False
        cell: 2 torch.Size([3, 128, 16, 16]) True
        cell: 3 torch.Size([3, 128, 16, 16]) False
        cell: 4 torch.Size([3, 128, 16, 16]) False
        cell: 5 torch.Size([3, 256, 8, 8]) True
        cell: 6 torch.Size([3, 256, 8, 8]) False
        cell: 7 torch.Size([3, 256, 8, 8]) False
        pool:   torch.Size([16, 256, 1, 1])
        linear: [b, 10]
        :param x:
        :return:
        """
        # print('network in:', x.shape)
        # s0 & s1 means the last cells' output
        s0 = s1 = self.stem(x)  # [b, 3, 32, 32] => [b, 48, 32, 32]
        # print('network stem:', s0.shape)
        # print('network stem1:', s1.shape)

        for i, cell in enumerate(self.cells):
            # weights are shared across all reduction cell or normal cell
            # according to current cell's type, it choose which architecture parameters
            # to use
            if cell.reduction:  # if current cell is reduction cell
                weights = F.softmax(self.alpha_reduce, dim=-1)
            else:
                weights = F.softmax(self.alpha_normal, dim=-1)  # [14, 8]
            # execute cell() firstly and then assign s0=s1, s1=result
            s0, s1 = s1, cell(s0, s1, weights)  # [40, 64, 32, 32]
            # print('cell:',i, s1.shape, cell.reduction, cell.reduction_prev)
            # print('\n')

        # s1 is the last cell's output
        out = self.global_pooling(s1)
        # logits = {}
        # for task, fc in self.classifier.items():
        #     logits[task] = fc(out.view(out.size(0), -1))
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
            """
            :param weights: [14, 8]
            :return:
            """
            gene = []
            n = 2
            start = 0
            for i in range(self.num_nodes):  # for each node
                end = start + n
                W = weights[start:end].copy()  # [2, 8], [3, 8], ...
                edges = sorted(range(i + 2),  # i+2 is the number of connection for node i
                               key=lambda x: -max(W[x][k]  # by descending order
                                                           for k in range(len(W[x]))  # get strongest ops
                                                           if k != PRIMITIVES.index('none'))
                               )[:2]  # only has two inputs
                for j in edges:  # for every input nodes j of current node i
                    k_best = None
                    for k in range(len(W[j])):  # get strongest ops for current input j->i
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))  # save ops and input node
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alpha_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alpha_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self.num_nodes - self.channel_multiplier, self.num_nodes + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )

        return genotype


def new(c, num_classes, num_layers, criterion, device, steps=4, multiplier=4, stem_multiplier=3):
    """
    create a new model and initialize it with current alpha parameters.
    However, its weights are left untouched.
    :return:
    """
    model = ConvNetwork(c, num_classes, num_layers, criterion, steps, multiplier, stem_multiplier).to(device)

    for x, y in zip(model.arch_parameters(), model.arch_parameters()):
        x.data.copy_(y.data)

    return model
