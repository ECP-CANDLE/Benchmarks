from collections import namedtuple


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


PRIMITIVES = [
    'none',
    'max_pool_3',
    'avg_pool_3',
    'skip_connect',
    'sep_conv_3',
    'sep_conv_5',
    'dil_conv_3',
    'dil_conv_5',
]


LINEAR_PRIMITIVES = [
    'linear_block',
    'skip_connect',
    'linear_conv',
    'linear_drop',
    'encoder',
    'none',
]


AmoebaNet = Genotype(
    normal=[
        ('avg_pool_3', 0),
        ('max_pool_3', 1),
        ('sep_conv_3', 0),
        ('sep_conv_5', 2),
        ('sep_conv_3', 0),
        ('avg_pool_3', 3),
        ('sep_conv_3', 1),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('avg_pool_3', 0),
        ('sep_conv_3', 1),
        ('max_pool_3', 0),
        ('sep_conv_7', 2),
        ('sep_conv_7', 0),
        ('avg_pool_3', 1),
        ('max_pool_3', 0),
        ('max_pool_3', 1),
        ('conv_7x1_1', 0),
        ('sep_conv_3', 5),
    ],
    reduce_concat=[3, 4, 6]
)


GradeNet36 = Genotype(
    normal=[
        ('sep_conv_5', 1),
        ('dil_conv_3', 0),
        ('sep_conv_5', 2),
        ('max_pool_3', 1),
        ('max_pool_3', 2),
        ('max_pool_3', 1),
        ('skip_connect', 4),
        ('max_pool_3', 1),
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('sep_conv_5', 0),
        ('sep_conv_5', 1),
        ('max_pool_3', 2),
        ('sep_conv_3', 1),
        ('dil_conv_5', 3),
        ('sep_conv_5', 2),
        ('sep_conv_5', 3),
        ('dil_conv_5', 4)
    ],
    reduce_concat=[4, 5, 6]
)


Multitask = Genotype(
    normal=[
        ('avg_pool_3', 1),
        ('sep_conv_3', 0),
        ('avg_pool_3', 1),
        ('sep_conv_5', 2),
        ('max_pool_3', 2),
        ('max_pool_3', 1),
        ('skip_connect', 4),
        ('avg_pool_3', 1)
    ],
    normal_concat=[4, 5, 6],
    reduce=[
        ('sep_conv_5', 1),
        ('sep_conv_5', 0),
        ('sep_conv_5', 2),
        ('sep_conv_3', 0),
        ('sep_conv_5', 3),
        ('sep_conv_5', 2),
        ('sep_conv_5', 4),
        ('sep_conv_5', 3)
    ],
    reduce_concat=[4, 5, 6]
)


MultitaskN2C3 = Genotype(
    normal=[
        ('max_pool_3', 0),
        ('sep_conv_5', 1),
        ('sep_conv_5', 1),
        ('sep_conv_5', 0)
    ],
    normal_concat=[2, 3, 4],
    reduce=[
        ('sep_conv_5', 1),
        ('sep_conv_5', 0),
        ('sep_conv_5', 1),
        ('sep_conv_3', 2)
    ],
    reduce_concat=[2, 3, 4]
)
