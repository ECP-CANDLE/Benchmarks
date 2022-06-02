data_sets = {
    '3k_Disordered': ('3k_run10_10us.35fs-DPPC.10-DOPC.70-CHOL.20.dir', 'ece75b704ec63ac9c39afd74b63497dc'),
    '3k_Ordered': ('3k_run32_10us.35fs-DPPC.50-DOPC.10-CHOL.40.dir', '211e1bcf46a3f19a978e4af63f067ce0'),
    '3k_Ordered_and_gel': ('3k_run43_10us.35fs-DPPC.70-DOPC.10-CHOL.20.dir', '87032ff78e4d01739aef5c6c0f5e4f04'),
    '6k_Disordered': ('6k_run10_25us.35fs-DPPC.10-DOPC.70-CHOL.20.dir', '13404cb8225819577e4821a976e9203b'),
    '6k_Ordered': ('6k_run32_25us.35fs-DPPC.50-DOPC.10-CHOL.40.dir', '95ef068b8deb69302c97f104b631d108'),
    '6k_Ordered_and_gel': ('6k_run43_25us.35fs-DPPC.70-DOPC.10-CHOL.20.dir', '3353e86d1cc2670820678c4c0c356206')
}

from collections import OrderedDict


def gen_data_set_dict():
    # Generating names for the data set
    names = {'x': 0, 'y': 1, 'z': 2,
             'CHOL': 3, 'DPPC': 4, 'DIPC': 5,
             'Head': 6, 'Tail': 7}
    for i in range(12):
        temp = 'BL' + str(i + 1)
        names.update({temp: i + 8})

    # dictionary sorted by value
    fields = OrderedDict(sorted(names.items(), key=lambda t: t[1]))

    return fields
