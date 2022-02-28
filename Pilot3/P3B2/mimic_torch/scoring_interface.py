import abc
from typing import List, Dict

class ScoringInterface(metaclass=abc.ABCMeta):
    """Abstract class to score sequences"""

    @abc.abstractmethod
    def generate_scores(self, prepared_sequences: List) -> Dict[str, float]:
        """Generate scores for prepared sequences.
        
        Note:
            Sequences should be prepared using prepare_data_for_scoring before being passed to this function. Any None object should be removed.
        
        Args:
            prepared_sequences (List): List of prepared sequences. The element type is decided by prepare_data_for_scoring

        Returns:
            Dict[str, List[float]] where keys are the scoring function name and the values are the scores

        """
        raise NotImplementedError

    @abc.abstractmethod
    def prepare_data_for_scoring(self, sequence: str):
        """Prepate data for scoring.
        
        Note:
            The purpose of this function is to format the input for scoring (e.g. smiles -> mol) and make sure that a given sequence is valid.

        Args:
            sequence (str): Sequence to prepare for scoring

        Returns:
            prepared sequence (may be a different type than input and may be None if invalid input sequence)

        """
        raise NotImplementedError

    @abc.abstractmethod
    def make_canonical(self, sequence) -> str:
        """Make canonical string from prepared sequence.
        
        Note:
            The purpose of this function is to genrete a canonical string for comparisons with previously discovered sequences.

        Args:
            sequence: Prepared sequence (output of prepare_data_for_scoring)

        Returns:
            str that can be used to search previously generated sequences

        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def column_names(self) -> List[str]:
        """Get names of metrics used for scoring.
    
        Returns:
            List[str] with names of functions used for scoring
        
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def selection_names(self) -> List[str]:
        """Get list of names used for selection.

        Returns:
            List[str] of names using for selection (i.e. fitness) scoring

        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def data_column_name(self) -> str:
        """Get name of data column.
        
        Returns:
            str with name of data column

        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def fitness_column_name(self) -> str:
        """Get name of fitness column
                
        Returns:
            str with name of fitness column

        """
        raise NotImplementedError