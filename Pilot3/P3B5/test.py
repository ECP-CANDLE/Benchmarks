import os
import sys


file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common',))
sys.path.append(lib_path2)


from darts import Architecture
from darts import ConvNetwork
from darts.genotypes import Genotype


print(Architecture)
print(ConvNetwork)
print(Genotype)
