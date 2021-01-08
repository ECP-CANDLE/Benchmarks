import torch
import torch.nn as nn
import torch.nn.functional as F

from darts.api import Model
from darts.modules.linear.cell import Cell
from darts.modules.classifier import MultitaskClassifier
from darts.genotypes import LINEAR_PRIMITIVES, Genotype


class Hyperparameters:
    c = 100
    num_nodes = 2
    num_cells = 3
    channel_multiplier = 1
    stem_channel_multiplier = 1
    intermediate_dim = 100


class LinearNetwork(Model):
    """ Collection of cells """

    def __init__(self, input_dim, tasks, criterion, device='cpu', hyperparams=Hyperparameters()):
        super(LinearNetwork, self).__init__()
        self.tasks = tasks
        self.criterion = criterion
        self.device = device
        self.c = hyperparams.c
        self.num_cells = hyperparams.num_cells
        self.num_nodes = hyperparams.num_nodes
        self.channel_multiplier = hyperparams.channel_multiplier

        # stem_multiplier is for stem network,
        # and multiplier is for general cell
        c_curr = hyperparams.stem_channel_multiplier * self.c

        self.stem = nn.Sequential(
            nn.Linear(
                input_dim, hyperparams.intermediate_dim
            ),
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

        self.classifier = MultitaskClassifier(hyperparams.intermediate_dim, tasks)

        # k is the total number of edges inside single cell
        k = sum(1 for i in range(self.num_nodes) for j in range(2 + i))
        num_ops = len(LINEAR_PRIMITIVES)  # 8

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

    def fc_layers(self, cp, tasks):
        """ Create fully connnected layers for each task """
        fc_layers = {}
        for task, dim in tasks.items():
            fc_layers[task] = nn.Linear(cp, dim).to(self.device)
        return fc_layers

    def new(self):
        """ Create a new model initialzed with current alpha parameters.

        Weights are left untouched.

        Returns
        -------
        model : Network
            New model initialized with current alpha.
        """
        model = LinearNetwork(
            self.tasks,
            self.criterion
        ).to(self.device)

        for x, y in zip(model.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)

        return model

    def forward(self, x):
        # s0 & s1 means the last cells' output
        s0 = s1 = self.stem(x)  # [b, 3, 32, 32] => [b, 48, 32, 32]

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

        # s1 is the last cell's output
        logits = self.classifier(s1.view(s1.size(0), -1))

        return logits

    def loss_value(self, x_data, y_true, y_pred, reduce='mean'):
        """ Calculate a value of loss function """
        y_pred = self(x_data)

        losses = {}
        for key, value in y_true.items():
            losses[key] = F.nll_loss(F.log_softmax(y_pred[key], dim=1), y_true[key])

        if reduce:
            total = 0
            for _, value in losses.items():
                total += value

            if reduce == "mean":
                losses = total / len(losses)
            elif reduce == "sum":
                losses = total

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
                                                           if k != LINEAR_PRIMITIVES.index('none'))
                               )[:2]  # only has two inputs
                for j in edges:  # for every input nodes j of current node i
                    k_best = None
                    for k in range(len(W[j])):  # get strongest ops for current input j->i
                        if k != LINEAR_PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((LINEAR_PRIMITIVES[k_best], j))  # save ops and input node
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
