"""
plangen command line argument definitions
"""

import argparse

partitioning_strategies = ['windows', 'undefine1', 'undefined2']         # to be completed ?????????????


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='feature-set partioning'
    )

    parser.add_argument('--in_dir',
                        type=str,
                        help='Directory containing feature-set list files')

    parser.add_argument('--out_dir',
                        default='results',
                        type=str,
                        help='Directory to contain generated plan files')

    parser.add_argument('--json',
                        action='store_true',
                        help='Generate plan in JSON format')

    parser.add_argument('--overwrite',
                        action='store_true',
                        help='Accept non-empty out_dir, contents overwritten')

    parser.add_argument('--partition_strategy',
                        choices=partitioning_strategies,
                        default=partitioning_strategies[0],
                        help='Specify a feature-set partitioning strategy')

    # The following fs_* arguments are required, the number of values specified for each
    # must match, and at least two values are required for each

    parser.add_argument('--fs_names',
                        required=True,
                        type=str,
                        nargs='+',
                        help='Specify a list of (arbitrary) feature-set names')

    parser.add_argument('--fs_paths',
                        required=True,
                        type=str,
                        nargs='+',
                        help='Specify a list of feature-set file paths')

    parser.add_argument('--fs_parts',
                        required=True,
                        type=int,
                        nargs='+',
                        help='Specify a list of partition counts')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Verbosity')

    parser.add_argument('--debug',
                        action='store_true',
                        help='Data structure dumps, etc')

    parser.add_argument('--test',
                        action='store_true',
                        help='Test plan navigation and entry retrieval')

    parser.add_argument('--maxdepth', type=int, default=0)
    parser.add_argument('--print_tree', action='store_true')
    args = parser.parse_args()
    return args
