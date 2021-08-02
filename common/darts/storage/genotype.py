import os
import json
from typing import List

from darts.genotypes import Genotype


class GenotypeStorage:
    """ Disk storage for Genotypes

    Args:
        root: rooth path to save genotype
    """

    def __init__(self, root: str):
        self.root = root

    def save_genotype(self, genotype: Genotype, filename='genotype.json') -> None:
        """ Save a genotype to disk

        Args:
            genotype: genotype to be saved
            filename: name of the save file
        """
        genotype = self._replace_range(genotype)
        os.makedirs(self.root, exist_ok=True)
        path = os.path.join(self.root, filename)
        with open(path, 'w') as outfile:
            json.dump(genotype, outfile)

    def load_genotype(self, filename='genotype.json') -> Genotype:
        """ Load a genotype from disk

        Args:
            filename: name of the save file

        Returns:
            the genotype
        """
        path = os.path.join(self.root, filename)
        with open(path, 'r') as infile:
            saved = json.load(infile)

        genotype = self._convert_serialized(saved)
        return genotype

    def _replace_range(self, genotype: Genotype) -> Genotype:
        """ Replace the range values with lists

        Python's `range` is not serializable as json objects.
        We convert the genotype's ranges to lists first.

        Args:
            genotype: the genotype to be serialized

        Returns
            genotype: with proper lists.
        """
        genotype = genotype._replace(normal_concat=list(genotype.normal_concat))
        genotype = genotype._replace(reduce_concat=list(genotype.reduce_concat))
        return genotype

    def _convert_serialized(self, save: list) -> Genotype:
        """ Convert json serialized form to Genotype

        Args:
            save: serialized form of the the genotype

        Returns:
            the genotype
        """
        # Serialized genotypes have a consistent structure
        normal = self._convert_to_tuple(save[0])
        normal_concat = save[1]
        reduce = self._convert_to_tuple(save[2])
        reduce_concat = save[3]
        return Genotype(normal, normal_concat, reduce, reduce_concat)

    def _convert_to_tuple(self, block: list) -> List[tuple]:
        """ Convert list to list of tuples

        Used when converting part of a serialized form of
        the genotype

        Args:
            block: part of the serialized genotype

        Returns:
            list of tuples that constitute that block
        """
        return [tuple(x) for x in block]
