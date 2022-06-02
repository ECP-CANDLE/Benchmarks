import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ResNet import ResNet, BasicBlock, Bottleneck

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

from extra_uts import right_shift, down_shift, concat_elu

from utils import MS_SSIM


class customLoss(nn.Module):
    def __init__(self, annealing_rate=0.1, annealing_on=True, use_crispy_loss=True):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.use_annealing = annealing_on
        self.annealing_rate = annealing_rate
        self.use_crispy_loss = use_crispy_loss
        if self.use_crispy_loss:
            self.crispyLoss = MS_SSIM()
        else:
            self.crispyLoss = None

    def compute_kernel(self, x, y):
        x_size = x.shape[0]
        y_size = y.shape[0]
        dim = x.shape[1]

        tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
        tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)

        return torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=2) / dim * 1.0)

    def compute_mmd(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

    def forward(self, x_recon, x, mu, logvar, epoch):
        loss_MSE = self.mse_loss(x_recon, x)
        # loss_mmd = self.compute_mmd(x_recon, x)

        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        if self.use_annealing:
            loss = loss_MSE + min(1.0, self.annealing_rate * epoch) * loss_KLD
        else:
            loss = loss_MSE + loss_KLD

        if self.use_crispy_loss:
            loss += self.crispyLoss(x_recon, x)

        return loss


class nin(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(nin, self).__init__()
        self.lin_a = (nn.Linear(dim_in, dim_out))
        self.dim_out = dim_out

    def forward(self, x):
        og_x = x
        # assumes pytorch ordering
        """ a network in network layer (1x1 CONV) """
        # TODO : try with original ordering
        x = x.permute(0, 2, 3, 1)
        shp = [int(y) for y in x.size()]
        out = self.lin_a(x.contiguous().view(shp[0] * shp[1] * shp[2], shp[3]))
        shp[-1] = self.dim_out
        out = out.view(shp)
        return out.permute(0, 3, 1, 2)


class down_shifted_conv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2, 3), stride=(1, 1),
                 shift_output_down=False, norm='weight_norm'):
        super(down_shifted_conv2d, self).__init__()

        assert norm in [None, 'batch_norm', 'weight_norm']
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride)
        self.shift_output_down = shift_output_down
        self.norm = norm
        self.pad = nn.ZeroPad2d((int((filter_size[1] - 1) / 2),  # pad left
                                 int((filter_size[1] - 1) / 2),  # pad right
                                 filter_size[0] - 1,  # pad top
                                 0))  # pad down

        if norm == 'weight_norm':
            self.conv == wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_down:
            self.down_shift = lambda x: down_shift(x, pad=nn.ZeroPad2d((0, 0, 1, 0)))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.down_shift(x) if self.shift_output_down else x


class down_shifted_deconv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2, 3), stride=(1, 1)):
        super(down_shifted_deconv2d, self).__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size, stride,
                                            output_padding=1))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        return x[:, :, :(xs[2] - self.filter_size[0] + 1),
                 int((self.filter_size[1] - 1) / 2):(xs[3] - int((self.filter_size[1] - 1) / 2))]


class down_right_shifted_conv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2, 2), stride=(1, 1),
                 shift_output_right=False, norm='weight_norm'):
        super(down_right_shifted_conv2d, self).__init__()

        assert norm in [None, 'batch_norm', 'weight_norm']
        self.pad = nn.ZeroPad2d((filter_size[1] - 1, 0, filter_size[0] - 1, 0))
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride=stride)
        self.shift_output_right = shift_output_right
        self.norm = norm

        if norm == 'weight_norm':
            self.conv == wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_right:
            self.right_shift = lambda x: right_shift(x, pad=nn.ZeroPad2d((1, 0, 0, 0)))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.right_shift(x) if self.shift_output_right else x


class down_right_shifted_deconv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2, 2), stride=(1, 1),
                 shift_output_right=False):
        super(down_right_shifted_deconv2d, self).__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size,
                                            stride, output_padding=1))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        x = x[:, :, :(xs[2] - self.filter_size[0] + 1):, :(xs[3] - self.filter_size[1] + 1)]
        return x


'''
skip connection parameter : 0 = no skip connection
                            1 = skip connection where skip input size === input size
                            2 = skip connection where skip input size === 2 * input size
'''


class gated_resnet(nn.Module):
    def __init__(self, num_filters, conv_op, nonlinearity=concat_elu, skip_connection=0):
        super(gated_resnet, self).__init__()
        self.skip_connection = skip_connection
        self.nonlinearity = nonlinearity
        self.conv_input = conv_op(2 * num_filters, num_filters)  # cuz of concat elu

        if skip_connection != 0:
            self.nin_skip = nin(2 * skip_connection * num_filters, num_filters)

        self.dropout = nn.Dropout2d(0.5)
        self.conv_out = conv_op(2 * num_filters, 2 * num_filters)

    def forward(self, og_x, a=None):
        x = self.conv_input(self.nonlinearity(og_x))
        if a is not None:
            x += self.nin_skip(self.nonlinearity(a))
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv_out(x)
        a, b = torch.chunk(x, 2, dim=1)
        c3 = a * F.sigmoid(b)
        return og_x + c3


class PixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                                    resnet_nonlinearity, skip_connection=0)
                                       for _ in range(nr_resnet)])

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                                     resnet_nonlinearity, skip_connection=1)
                                        for _ in range(nr_resnet)])

    def forward(self, u, ul):
        u_list, ul_list = [], []

        for i in range(self.nr_resnet):
            u = self.u_stream[i](u)
            ul = self.ul_stream[i](ul, a=u)
            u_list += [u]
            ul_list += [ul]

        return u_list, ul_list


class PixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                                    resnet_nonlinearity, skip_connection=1)
                                       for _ in range(nr_resnet)])

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                                     resnet_nonlinearity, skip_connection=2)
                                        for _ in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list):
        for i in range(self.nr_resnet):
            u = self.u_stream[i](u, a=u_list.pop())
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1))

        return u, ul


class PixelCNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=10, nr_logistic_mix=6,
                 resnet_nonlinearity='concat_elu', input_channels=3):
        super(PixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu':
            self.resnet_nonlinearity = lambda x: concat_elu(x)
        else:
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad = nn.ZeroPad2d((0, 0, 1, 0))

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters,
                                                             self.resnet_nonlinearity) for i in range(3)])

        self.up_layers = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters,
                                                         self.resnet_nonlinearity) for _ in range(3)])

        self.downsize_u_stream = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters,
                                                                    stride=(2, 2)) for _ in range(2)])

        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters,
                                                                           nr_filters, stride=(2, 2)) for _ in
                                                 range(2)])

        self.upsize_u_stream = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters,
                                                                    stride=(2, 2)) for _ in range(2)])

        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters,
                                                                           nr_filters, stride=(2, 2)) for _ in
                                               range(2)])

        self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2, 3),
                                          shift_output_down=True)

        self.ul_init = nn.ModuleList([down_shifted_conv2d(input_channels + 1, nr_filters,
                                                          filter_size=(1, 3), shift_output_down=True),
                                      down_right_shifted_conv2d(input_channels + 1, nr_filters,
                                                                filter_size=(2, 1), shift_output_right=True)])

        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)

    def forward(self, x, sample=False):
        # similar as done in the tf repo :
        # print("Sample: ", sample)
        # print("SHAPE: ", x.shape, x.dtype)

        if sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            padding = padding.cuda() if x.is_cuda else padding
            x = torch.cat((x, padding), 1)
        else:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            init_padding = padding.cuda() if x.is_cuda else padding
            x = torch.cat((x, init_padding), 1)
        # else:
        #     x = torch.cat((x, self.init_padding), 1)
        # print("PADDING: ", self.init_padding.shape)

        # ##      UP PASS    ###
        # print("final: ", x.shape)
        u_list = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        for i in range(3):
            # resnet block
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1])
            u_list += u_out
            ul_list += ul_out

            if i != 2:
                # downscale (only twice)
                u_list += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]

        # ##    DOWN PASS    ###
        u = u_list.pop()
        ul = ul_list.pop()

        for i in range(3):
            # resnet block
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)

            # upscale (only twice)
            if i != 2:
                u = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        x_out = self.nin_out(F.elu(ul))

        assert len(u_list) == len(ul_list) == 0

        return x_out


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class Repeat(nn.Module):

    def __init__(self, rep):
        super(Repeat, self).__init__()

        self.rep = rep

    def forward(self, x):
        size = tuple(x.size())
        size = (size[0], 1) + size[1:]
        x_expanded = x.view(*size)
        n = [1 for _ in size]
        n[1] = self.rep
        return x_expanded.repeat(*n)


class Flatten(nn.Module):

    def forward(self, x):
        size = x.size()  # read in N, C, H, W
        return x.view(size[0], -1)


# class SmilesDecoder(nn.Module):
#     def __init__(self,  vocab_size, max_length_sequence, rep_size = 200 , embedder = None):
#         super(SmilesDecoder, self).__init__()
#         self.rep_size = rep_size
#         self.embeder = embedder
#         self.vocab_size = vocab_size
#         self.max_length_sequence = max_length_sequence
#         self.repeat_vector = Repeat(self.max_length_sequence)
#         self.gru1 = nn.GRU(input_size = rep_size, num_layers=3, hidden_size=501, batch_first=True)
#         self.dense = nn.Sequential(nn.Linear(501, vocab_size), nn.Softmax())
#         self.timedib = TimeDistributed(self.dense, batch_first=True)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.tanh = nn.Tanh()
#
#
#     def forward(self, x):
#         x = self.repeat_vector(x)
#         x, _ = self.gru1(x)
#         x = self.timedib(x)
#         return x
#
#
# class SmilesEncoder(nn.Module):
#
#     def __init__(self,  vocab_size, max_length_sequence, rep_size = 200 , embedder = None):
#         super(SmilesEncoder, self).__init__()
#         self.rep_size = rep_size
#         self.embeder = embedder
#         self.vocab_size = vocab_size
#         self.max_length_sequence = max_length_sequence
#
#         ##layers
#
#         self.conv1 = nn.Conv1d(in_channels=self.max_length_sequence, out_channels=90, kernel_size=9, stride=1)
#         self.conv2 = nn.Conv1d(in_channels=90, out_channels=300, kernel_size=10, stride=1)
#         self.conv3 = nn.Conv1d(in_channels=300, out_channels=900, kernel_size=10, stride=1)
#
#         self.relu = nn.ReLU()
#
#         # Latent vectors mu and sigma
#         self.fc22 = nn.Linear(900, rep_size)
#         self.fc21 = nn.Linear(900, rep_size)
#
#
#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.relu(self.conv3(x))
#         x = Flatten()(x)
#
#         return self.fc21(x), self.fc22(x)

# class PictureEncoder(nn.Module):
#     def __init__(self, rep_size=500):
#         super(PictureEncoder, self).__init__()
#         self.rep_size = rep_size
#         self.encoder = ResNet(BasicBlock, [3, 2, 2, 3], num_classes=rep_size)
#         self.mu = nn.Linear(rep_size, rep_size)
#         self.logvar = nn.Linear(rep_size, rep_size)
#
#     def forward(self, x):
#         x = self.encoder(x)
#
#         return self.mu(x), self.logvar(x)
#


class SeparableConv3(nn.Module):
    def __init__(self, kernel_size, stride=3, padding=1, bias=False):
        super(SeparableConv3, self).__init__()
        self.channels = 3

        self.ch1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias), nn.ReLU())
        self.ch2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias), nn.ReLU())
        self.ch3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=bias), nn.ReLU())

    def forward(self, x):
        return torch.cat((self.ch1(x[:, 0, ...]),
                          self.ch2(x[:, 1, ...]),
                          self.ch3(x[:, 2, ...])),
                         dim=1)


class PictureEncoder(nn.Module):
    def __init__(self, rep_size=512):
        super(PictureEncoder, self).__init__()
        self.rep_size = rep_size

        self.encoder = ResNet(BasicBlock, [2, 3, 2, 3], num_classes=rep_size, in_classes=1)
        self.encoder_color = ResNet(BasicBlock, [2, 3, 2, 3], num_classes=rep_size, in_classes=3)
        self.lc1 = nn.Sequential(nn.Linear(rep_size, rep_size), nn.LeakyReLU(), nn.Linear(rep_size, rep_size),
                                 nn.LeakyReLU())
        self.lc2 = nn.Sequential(nn.Linear(rep_size, rep_size), nn.LeakyReLU(), nn.Linear(rep_size, rep_size),
                                 nn.LeakyReLU())

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        color_enc = self.encoder_color(x)
        color_enc = color_enc.view(-1, self.rep_size)
        color_enc = self.lc2(color_enc)

        x = torch.mean(x, dim=1, keepdim=True)
        black_enc = self.encoder(x).view(-1, self.rep_size)
        black_enc = self.lc1(black_enc)
        return color_enc, black_enc


def conv3x3T(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)


def conv4x4T(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=stride,
                              padding=1, bias=False)


def conv1x1T(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class TransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, upscale=None):
        super(TransposeBlock, self).__init__()
        self.inc = in_channels
        self.ouc = out_channels
        self.conv1 = conv1x1T(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if stride > 1:
            self.conv2 = conv4x4T(out_channels, out_channels, stride)
        else:
            self.conv2 = conv3x3T(out_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv1x1T(out_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.unpool = nn.Upsample(scale_factor=(2, 2))
        self.upconv = conv1x1T(in_channels, out_channels)
        self.stride = stride

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        if self.stride > 1:
            identity = self.unpool(identity)
            identity = self.upconv(identity)
        elif self.inc != self.ouc:
            identity = self.upconv(identity)

        x = x + identity
        return self.relu(x)


# Dense Net

class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)

        return output


class BindingAffModel(nn.Module):
    def __init__(self, rep_size=500, dropout=None):
        super(BindingAffModel, self).__init__()
        self.rep_size = rep_size

        self.attention = MultiHeadAttention(2, 500)
        self.model = nn.Sequential(
            nn.Linear(500, 128),
            nn.SELU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()
        )

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.attention(x.view(-1, 1, self.rep_size), x.view(-1, 1, self.rep_size), x.view(-1, 1, self.rep_size))
        return self.model(out.view(batch_size, -1))


class TranposeConvBlock(nn.Module):
    def __init__(self, in_plane, out_plane, padding=(0, 0), stride=(0, 0), kernel_size=(0, 0), dropout=None):
        super(TranposeConvBlock, self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_plane, out_plane, kernel_size=kernel_size[0], padding=padding[0],
                                        stride=stride[0], bias=False)
        self.conv2 = nn.ConvTranspose2d(out_plane, out_plane, kernel_size=kernel_size[1], padding=padding[1],
                                        stride=stride[1], bias=False)
        # self.conv3 = nn.ConvTranspose2d(out_plane, out_plane, kernel_size=kernel_size[1], padding=padding[1], stride=stride[1], bias=False)
        self.bn1 = nn.BatchNorm2d(out_plane)
        self.bn2 = nn.BatchNorm2d(out_plane)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class PictureDecoder(nn.Module):
    def __init__(self, rep_size=512):
        super(PictureDecoder, self).__init__()
        self.rep_size = rep_size
        # Sampling vector
        self.fc3 = nn.Linear(rep_size, rep_size * 3)
        self.fc4 = nn.Linear(rep_size * 3, rep_size * 6)

        # Decoder
        self.preconv = nn.ConvTranspose2d(12, 6, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv15 = nn.ConvTranspose2d(6, 6, kernel_size=2, stride=1, padding=0, bias=False)
        self.conv15_ = nn.ConvTranspose2d(6, 6, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn15 = nn.BatchNorm2d(6)
        self.conv16 = nn.ConvTranspose2d(6, 6, kernel_size=3, stride=2, padding=0, bias=False)
        self.conv16_ = nn.ConvTranspose2d(6, 4, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn16 = nn.BatchNorm2d(4)
        self.conv20 = nn.ConvTranspose2d(4, 4, kernel_size=4, stride=2, padding=0, bias=False)
        self.conv20_ = nn.ConvTranspose2d(4, 4, kernel_size=4, stride=1, padding=0, bias=False)
        self.conv17 = nn.ConvTranspose2d(4, 4, kernel_size=4, stride=2, padding=0, bias=False)
        self.conv17_ = nn.ConvTranspose2d(4, 4, kernel_size=20, stride=1, padding=0, bias=False)
        self.bn21 = nn.BatchNorm2d(4)
        self.conv18 = nn.ConvTranspose2d(4, 4, kernel_size=12, stride=1, padding=0, bias=False)
        self.conv18_ = nn.ConvTranspose2d(4, 3, kernel_size=20, stride=1, padding=0, bias=False)
        self.conv19 = nn.ConvTranspose2d(3, 3, kernel_size=16, stride=1, padding=0, bias=False)
        self.convlast = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, out):
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out).view(-1, 12, 16, 16)
        out = self.relu(self.preconv(out))
        out = self.relu(self.conv15(out))
        out = self.relu(self.conv15_(out))
        out = self.bn15(out)
        out = self.relu(self.conv16(out))
        out = self.relu(self.conv16_(out))
        out = self.bn16(out)

        out = self.relu(self.conv20(out))
        out = self.relu(self.conv20_(out))
        out = self.relu(self.conv17(out))
        out = self.relu(self.conv17_(out))
        out = self.bn21(out)

        out = self.relu(self.conv18(out))
        out = self.relu(self.conv18_(out))
        out = self.relu(self.conv19(out))
        out = self.convlast(out)

        out = self.sigmoid(out)
        return out


class GeneralVae(nn.Module):
    def __init__(self, encoder_model, decoder_model, rep_size=500):
        super(GeneralVae, self).__init__()
        self.rep_size = rep_size

        self.encoder = encoder_model
        self.decoder = decoder_model

    def encode(self, x):
        return self.encoder(x)

    # def reparameterize(self, mu, logvar):
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return mu + eps * std

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar.mul(0.5))
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
        # return mu + logvar

    def decode(self, z):
        return self.decoder(z)

    def encode_latent_(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


class GeneralVaeBinding(nn.Module):
    def __init__(self, encoder_model, decoder_model, binding_model, rep_size=500):
        super(GeneralVaeBinding, self).__init__()
        self.rep_size = rep_size

        self.encoder = encoder_model
        self.decoder = decoder_model
        self.binding = binding_model

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar.mul(0.5))
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        return self.decoder(z)

    def calc_binding(self, z):
        return self.binding(z)

    def encode_latent_(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, self.calc_binding(z)


class Lambda(nn.Module):

    def __init__(self, i=1000, o=500, scale=1E-2):
        super(Lambda, self).__init__()

        self.scale = scale

    def forward(self, x):
        self.mu = self.z_mean(x)
        self.log_v = self.z_log_var(x)
        eps = self.scale * Variable(torch.randn(*self.log_v.size())
                                    ).type_as(self.log_v)
        return self.mu + torch.exp(self.log_v / 2.) * eps


class ComboVAE(nn.Module):
    def __init__(self, encoder_model_1, encoder_model_2, decoder_model_1, decoder_model_2, rep_size=500):
        super(ComboVAE, self).__init__()
        self.rep_size = rep_size

        self.scale = 1E-2
        self.encoder1 = encoder_model_1
        self.encoder2 = encoder_model_2
        self.decoder1 = decoder_model_1
        self.decoder2 = decoder_model_2

        self.z_mean = nn.Linear(rep_size * 2, rep_size)
        self.z_log_var = nn.Linear(rep_size * 2, rep_size)

    def encode(self, x1, x2):  # returns single values encoded
        return self.encoder1(x1), self.encoder2(x2)

    def reparam(self, logvar, mu):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        return self.decoder1(z), self.decoder2(z)

    def encode_latent_(self, x1, x2):
        x1, x2 = self.encode(x1, x2)
        x = torch.cat([x1, x2], dim=1)
        mu, logvar = (self.z_mean(x), self.z_log_var(x))
        z = self.reparam(logvar, mu)

        return z, mu, logvar

    def forward(self, x1, x2):
        z, mu, logvar = self.encode_latent_(x1, x2)

        y_1, y_2 = self.decode(z)
        return y_1, y_2, mu, logvar

# class SELU(nn.Module):
#
#     def __init__(self, alpha=1.6732632423543772848170429916717,
#                  scale=1.0507009873554804934193349852946, inplace=False):
#         super(SELU, self).__init__()
#
#         self.scale = scale
#         self.elu = nn.ELU(alpha=alpha, inplace=inplace)
#
#     def forward(self, x):
#         return self.scale * self.elu(x)
#
#
# def ConvSELU(i, o, kernel_size=3, padding=0, p=0.):
#     model = [nn.Conv1d(i, o, kernel_size=kernel_size, padding=padding),
#              SELU(inplace=True)
#              ]
#     if p > 0.:
#         model += [nn.Dropout(p)]
#     return nn.Sequential(*model)
#
#
#
#
# class MolEncoder(nn.Module):
#
#     def __init__(self, i=60, o=500, c=27):
#         super(MolEncoder, self).__init__()
#
#         self.i = i
#
#         self.conv_1 = ConvSELU(i, 9, kernel_size=9)
#         self.conv_2 = ConvSELU(9, 9, kernel_size=9)
#         self.conv_3 = ConvSELU(9, 10, kernel_size=11)
#         self.dense_1 = nn.Sequential(nn.Linear((c - 29 + 3) * 10, 435),
#                                      SELU(inplace=True))
#
#         self.z_mean = nn.Linear(435, o)
#         self.z_log_var = nn.Linear(435, o)
#
#
#     def forward(self, x):
#         out = self.conv_1(x)
#         out = self.conv_2(out)
#         out = self.conv_3(out)
#         out = Flatten()(out)
#         out = self.dense_1(out)
#
#         return self.z_mean(out), self.z_log_var(out)
#
#
#
# class MolDecoder(nn.Module):
#
#     def __init__(self, i=500, o=60, c=27):
#         super(MolDecoder, self).__init__()
#
#         self.latent_input = nn.Sequential(nn.Linear(i, i),
#                                           SELU(inplace=True))
#         self.repeat_vector = Repeat(o)
#         self.gru = nn.GRU(i, 501, 3, batch_first=True)
#         self.decoded_mean = TimeDistributed(nn.Sequential(nn.Linear(501, c),
#                                                           nn.Softmax())
#                                             )
#
#     def forward(self, x):
#         out = self.latent_input(x)
#         out = self.repeat_vector(out)
#         out, h = self.gru(out)
#         return self.decoded_mean(out
#       )


class SELU(nn.Module):

    def __init__(self, alpha=1.6732632423543772848170429916717,
                 scale=1.0507009873554804934193349852946, inplace=False):
        super(SELU, self).__init__()

        self.scale = scale
        self.elu = nn.ELU(alpha=alpha, inplace=inplace)

    def forward(self, x):
        return self.scale * self.elu(x)


def ConvSELU(i, o, kernel_size=3, padding=0, p=0.):
    model = [nn.Conv1d(i, o, kernel_size=kernel_size, padding=padding),
             SELU(inplace=True)
             ]
    if p > 0.:
        model += [nn.Dropout(p)]
    return nn.Sequential(*model)


class Lambda(nn.Module):

    def __init__(self, i=435, o=292, scale=1E-2):
        super(Lambda, self).__init__()

        self.scale = scale
        self.z_mean = nn.Linear(i, o)
        self.z_log_var = nn.Linear(i, o)

    def forward(self, x):
        self.mu = self.z_mean(x)
        self.log_v = self.z_log_var(x)

        std = self.log_v.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(self.mu)

    # def forward(self, x):
    #     self.mu = self.z_mean(x)
    #     self.log_v = self.z_log_var(x)
    #     eps = self.scale * Variable(torch.randn(*self.log_v.size())
    #                                 ).type_as(self.log_v)
    #     return self.mu + torch.exp(self.log_v / 2.) * eps


class MolEncoder(nn.Module):

    def __init__(self, i=60, o=500, c=27):
        super(MolEncoder, self).__init__()

        self.i = i

        self.conv_1 = ConvSELU(i, 9, kernel_size=9)
        self.conv_2 = ConvSELU(9, 9, kernel_size=9)
        self.conv_3 = ConvSELU(9, 10, kernel_size=11)
        self.dense_1 = nn.Sequential(nn.Linear((c - 29 + 3) * 10, 435),
                                     SELU(inplace=True))

        # self.lmbd = Lambda(435, o)
        self.z_mean = nn.Linear(435, o)
        self.z_log_var = nn.Linear(435, o)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = Flatten()(out)
        out = self.dense_1(out)

        return out

    def vae_loss(self, x_decoded_mean, x):
        z_mean, z_log_var = self.lmbd.mu, self.lmbd.log_v

        bce = nn.BCELoss(size_average=True)
        xent_loss = self.i * bce(x_decoded_mean, x.detach())
        kl_loss = -0.5 * torch.mean(1. + z_log_var - z_mean ** 2. - torch.exp(z_log_var))

        return kl_loss + xent_loss


class DenseMolEncoder(nn.Module):

    def __init__(self, i=60, o=500, c=27):
        super(DenseMolEncoder, self).__init__()

        self.i = i

        self.conv_1 = ConvSELU(i, 9, kernel_size=9)
        self.conv_2 = ConvSELU(9, 9, kernel_size=9)
        self.conv_3 = ConvSELU(9, 10, kernel_size=11)

        self.dense_0 = nn.Sequential(Flatten(),
                                     nn.Linear(60 * 27, 500),
                                     SELU(inplace=True),
                                     nn.Linear(500, 500),
                                     SELU(inplace=True),
                                     nn.Linear(500, 500),
                                     SELU(inplace=True))
        self.dense_1 = nn.Sequential(nn.Linear((c - 29 + 3) * 10, 500),
                                     SELU(inplace=True))

        self.z_mean = nn.Linear(500, o)
        self.z_log_var = nn.Linear(500, o)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        out = self.conv_3(out)
        out = Flatten()(out)
        out = self.dense_1(out) + self.dense_0(x)

        return out

    def vae_loss(self, x_decoded_mean, x):
        z_mean, z_log_var = self.lmbd.mu, self.lmbd.log_v

        bce = nn.BCELoss(size_average=True)
        xent_loss = self.i * bce(x_decoded_mean, x.detach())
        kl_loss = -0.5 * torch.mean(1. + z_log_var - z_mean ** 2. - torch.exp(z_log_var))

        return kl_loss + xent_loss


class MolDecoder(nn.Module):

    def __init__(self, i=500, o=60, c=27):
        super(MolDecoder, self).__init__()

        self.latent_input = nn.Sequential(nn.Linear(i, i),
                                          SELU(inplace=True))
        self.repeat_vector = Repeat(o)
        self.gru = nn.GRU(i, 501, 3, batch_first=True)
        self.decoded_mean = TimeDistributed(nn.Sequential(nn.Linear(501, c),
                                                          nn.Softmax())
                                            )

    def forward(self, x):
        out = self.latent_input(x)
        out = self.repeat_vector(out)
        out, h = self.gru(out)
        return self.decoded_mean(out)


class ZSpaceTransform(nn.Module):
    def __init__(self, i=500, o=60, ):
        super(ZSpaceTransform, self).__init__()

        self.mu = nn.Sequential(nn.Linear(i, i),
                                SELU(inplace=True),
                                nn.Linear(i, i), SELU(inplace=True),
                                nn.Linear(i, i), SELU(inplace=True), nn.Linear(i, i))

        self.logvar = nn.Sequential(nn.Linear(i, i),
                                    SELU(inplace=True),
                                    nn.Linear(i, i), SELU(inplace=True),
                                    nn.Linear(i, i), SELU(inplace=True), nn.Linear(i, i))

    def forward(self, mu, log):
        mu = self.mu(mu)
        log = self.logvar(log)
        return mu, log


class TestVAE(nn.Module):

    def __init__(self, encoder, transformer, decoder):
        super(TestVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.transformer = transformer

    def encode(self, x):
        self.mu, self.log_v = self.encoder(x)
        self.mu, self.log_v = self.transformer(self.mu, self.log_v)

        std = self.log_v.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        y = eps.mul(std).add_(self.mu)
        return y

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x, return_y=False):
        self.mu, self.log_v = self.encoder(x)
        self.mu, self.log_v = self.transformer(self.mu, self.log_v)
        std = self.log_v.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        y = eps.mul(std).add_(self.mu)
        if return_y:
            return y, self.decoder(y)
        return self.decoder(y)

    def vae_loss(self, x_decoded_mean, x):
        z_mean, z_log_var = self.mu, self.log_v

        # bce = nn.BCELoss(size_average=True)
        bce = nn.MSELoss(reduction="sum")

        xent_loss = bce(x_decoded_mean, x.detach())
        kl_loss = -0.5 * torch.mean(1. + z_log_var - z_mean ** 2. - torch.exp(z_log_var))

        return kl_loss + xent_loss


class AutoModel(nn.Module):

    def __init__(self, encoder, decoder):
        super(AutoModel, self).__init__()
        self.encoder = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=500)
        self.attention = nn.Linear(500, 500)
        self.reduce = nn.Linear(500, 292)
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        atten = nn.Softmax()(self.attention(x))
        x = nn.ReLU()(self.reduce(atten * x))
        x = self.decoder(x)
        return x
