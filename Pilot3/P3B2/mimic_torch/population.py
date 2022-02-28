from evolution import Evolution
import numpy as np
import math

class Population():
    """Class to store a population of sequences and apply mutation/recombination/selection."""

    def __init__(self, evolution_operators, scoring_operator, niche_column_name=None):
        """Contructor for Population class.

        Args:
            evoluation_operators (List[MaskedEvoluation]): List of MaskedEvoluation objects to apply mutation/recombination to population
            scoring_operator (ScoringInterface): Scoring operator to score population sequences

        """
        super().__init__()

        # store evoluation_operators and scoring_operator
        self._evolution_operators = evolution_operators
        self._scoring_operator = scoring_operator

        # intialize population dict
        self._population_dict = {}
        self._data_column_name = self._scoring_operator.data_column_name
        self._fitness_column_name = self._scoring_operator.fitness_column_name
        self._column_names = self._scoring_operator.column_names
        self._niche_column_name = niche_column_name

        # check that niche name is included in scoring
        if self._niche_column_name is not None:
            if self._niche_column_name not in self._column_names:
                raise KeyError('%s not found in scoring operator: %s' % (self._niche_column_name, ', '.join(self._column_names)))

    @property
    def population_dict(self):
        """Get population dictionary.

        Returns:
            Dict[str,] with data and scores for population

        """
        return self._population_dict

    @property
    def population_size(self):
        """ Get population size.

        Returns;
            int with population size
        """
        if len(self._population_dict) > 0:
            return len(self._population_dict[self._data_column_name])
        else:
            return 0

    @property
    def population_sequences(self):
        """Get population sequences.
        
        Returns:
            List[str] with values for data column name key in population dict

        """
        return self._population_dict[self._data_column_name]

    def read_population_dict_from_file(self, population_file, population_size=1000, delimiter='\t', generate_metrics=False, fill_to_size=True):
        """Read a population file and store contents in population_dict

        Args:
            population_file (str): Path to population file (should be a delimited file with header row with column names)
            population_size (int): Number of data rows in the population.
            delimiter (str): Delimiter for parsing population file

        """
        # create a dictionary to store column names and order from file
        header_dict = {}
        reverse_header_dict = {}

        # clear population_dict
        self._population_dict = {}

        # store sequences that must be scored
        sequences_to_score = []

        # flag to generate metrics if not provided
        # generate_metrics = False

        with open(population_file, 'r') as input_file:
            row_counter = -1
            for row in input_file:
                row_counter += 1
                # read header row
                if row_counter == 0:
                    header_names = [x.strip() for x in row.split(delimiter)]
                    column_counter = 0
                    for name in header_names:
                        header_dict[name] = column_counter
                        column_counter += 1

                    # construct reverse map
                    for key in header_dict:
                        reverse_header_dict[header_dict[key]] = key

                    # make sure that file has data column
                    if self._data_column_name not in header_dict:
                        raise ValueError('Error: %s does not have %s as a header column' % (population_file, self._data_column_name))
                
                    # check that all metrics are provided
                    required_column_names = self._scoring_operator.column_names
                    for column_name in required_column_names:
                        self._population_dict[column_name] = []
                        if column_name not in header_dict:
                            generate_metrics = True

                    continue

                # non-header rows
                row_split = [x.strip() for x in row.split(delimiter)]

                if not generate_metrics:
                    for i in range(len(row_split)):
                        if reverse_header_dict[i] != self._data_column_name:
                            self._population_dict[reverse_header_dict[i]].append(float(row_split[i]))
                        else:
                            self._population_dict[reverse_header_dict[i]].append(row_split[i])
                else:
                    sequences_to_score.append(row_split[header_dict[self._data_column_name]])

                # stop reading if population_size is met
                if population_size >= 0 and row_counter >= population_size:
                    break

        # check if scoring is needed
        if len(sequences_to_score) > 0:
            self._population_dict = self._sequences_to_population_dict(sequences_to_score)

        # fill population by making random copies to desired size
        if (len(self._population_dict[self._data_column_name]) < population_size) and fill_to_size:
            self._fill_population_dict(population_size)

    def generate_child_population_dict(self, mutation_samples, recombination_samples, weighted=False, previous_set=None, db_dict=None, batch_size=256, return_valid=False):
        """Generate a child population dict from current population dict

        Args:
            mutation_samples (List[int]): Number of mutation samples for each Evolution object
            recombination_samples (List[int]): Number of recombination samples for each Evolution object
            weight (bool): Option to weight sampling for mutation and recombination
            previous_set (set[str]): Set of previously visited sequences
            db_dict (Dict[str,]): Dictionary with keys cursor and query_string
            batch_size (int): Batch size for generating mutations and recombinations

        Returns:
            Dict[str,] with child population dict

        """
        # check that mutation samples are valid
        if len(mutation_samples) != len(self._evolution_operators):
            raise ValueError('Error length of mutation samples is not equal to length of evolution operators')

        # check that recombination samples are valid
        if len(recombination_samples) != len(self._evolution_operators):
            raise ValueError('Error length of recombination samples is not equal to length of evoluation operators')

        # setup defaults
        if previous_set is None:
            previous_set = set()

        if db_dict is None:
            db_dict = {}

        # store generated sequences from mutations and recombinations
        possible_sequences = []
        
        # mutation samples
        for i in range(len(mutation_samples)):
            samples = self._sample_population_dict(mutation_samples[i], weighted)
            batch_results = []
            for j in range(0, mutation_samples[i], batch_size):
                start_index = j
                end_index = min(start_index + batch_size, mutation_samples[i])
                task = np.random.choice(['replace', 'insert', 'delete'])
                batch_results = self._evolution_operators[i].mutate_sequences(samples[start_index:end_index], task=task)

                for result in batch_results:
                    for r in result:
                        possible_sequences.append(r['sequence'])

        # recombination samples
        for i in range(len(recombination_samples)):
            samples_1 = self._sample_population_dict(recombination_samples[i], weighted)
            samples_2 = self._sample_population_dict(recombination_samples[i], weighted=False)
            batch_results = []
            for j in range(0, recombination_samples[i], batch_size):
                start_index = j
                end_index = min(start_index + batch_size, recombination_samples[i])
                batch_results = self._evolution_operators[i].recombine_sequences(samples_1[start_index:end_index], samples_2[start_index:end_index])

                for result in batch_results:
                    for r in result:
                        possible_sequences.append(r['sequence'])   

        # generate child population dict
        return self._sequences_to_population_dict(possible_sequences, previous_set, db_dict, return_valid)

    def merge_child_population_dict(self, child_population_dict, children_only=False, exclude_sequences=None, max_size=-1, max_niche_size=-1, pad_niches=True):
        """Merge child population dict with current population dict.

        Args:
            child_population_dict (Dict[str,]): Dictionary for child population
            random_merge (bool): Option to randomly merge, otherwise use top fitness scores

        Returns:
            int number of child population merged into original population

        """
        # save population size before merge
        original_population_size = self.population_size
        # print('Original population size:', original_population_size)

        # append to current population
        for key in self._column_names:
            if children_only:
                self._population_dict[key] = child_population_dict[key]
            else:
                self._population_dict[key] += child_population_dict[key]

        # set fitness values and niches for exclude sequences to zero
        if exclude_sequences is not None:
            for i in range(self.population_size):
                if self._population_dict[self._data_column_name][i] in exclude_sequences:
                    self._population_dict[self._fitness_column_name][i] = 0.0
                    if self._niche_column_name is not None:
                        self._population_dict[self._niche_column_name][i] = 0

        # allow for niche selection
        selection_index = None
        if self._niche_column_name is not None:

            # convert to niche populations
            niche_dict = {}
            for i in range(self.population_size):
                niche_index = self._population_dict[self._niche_column_name][i]

                # special 0 index is always removed
                if niche_index == 0:
                    continue

                if niche_index not in niche_dict:
                    niche_dict[niche_index] = []
                niche_dict[niche_index].append(i)

            # allow population growth if max_size is set
            cutoff_size = original_population_size
            if max_size > 0:
                cutoff_size = min(max_size, len(self._population_dict[self._data_column_name]))

            # equal distribution amongst niches
            niche_size = 1
            if max_niche_size > 0:
                niche_size = max_niche_size
            else:
                niche_size = max(1, int(math.floor(cutoff_size/len(niche_dict))))

            # apply selection for each niche
            selection_index = []
            for niche_index in niche_dict:
                current_size = len(niche_dict[niche_index])
                selected_size = min(current_size, niche_size)
                if len(self._scoring_operator.selection_names) < 1:
                    random_selection = np.random.choice(current_size, selected_size, replace=False)
                    selection_index += [niche_dict[niche_index][x] for x in random_selection]
                else:
                    niche_fitness = [self._population_dict[self._fitness_column_name][x] for x in niche_dict[niche_index]]
                    fitness_selection = list(np.argsort(-1.0*np.array(niche_fitness))[:selected_size])
                    selection_index += [niche_dict[niche_index][x] for x in fitness_selection]

            selection_index = np.array(selection_index)
            # print(len(selection_index))

            if pad_niches:
                if len(selection_index) < cutoff_size:
                    selection_index_global = None
                    if len(self._scoring_operator.selection_names) < 1:
                        selection_index_global = np.random.choice(self.population_size, cutoff_size, replace=False)
                    else:
                        selection_index_global = np.argsort(-1.0*np.array(self._population_dict[self._fitness_column_name]))[:cutoff_size]

                    selection_index_global = np.setdiff1d(selection_index_global, selection_index, assume_unique=True)
                    selection_index = np.concatenate([selection_index, selection_index_global[:cutoff_size-len(selection_index)]])
                    # print(len(selection_index))
            
            # only retain top niches to keep population size fixed
            if len(selection_index) > cutoff_size:
                selection_index = selection_index[np.argsort(-1.0*np.array([self._population_dict[self._fitness_column_name][x] for x in selection_index]))[:cutoff_size]]

        else:

            # allow population growth if max_size is set
            cutoff_size = original_population_size
            if max_size > 0:
                cutoff_size = min(max_size, len(self._population_dict[self._data_column_name]))

            # population size is maintained by the merge
            selection_index = None
            if len(self._scoring_operator.selection_names) < 1:
                selection_index = np.random.choice(self.population_size, cutoff_size, replace=False)
            else:
                selection_index = np.argsort(-1.0*np.array(self._population_dict[self._fitness_column_name]))[:cutoff_size]

        # apply selection
        for key in self._column_names:
            self._population_dict[key] = [self._population_dict[key][x] for x in selection_index]

        # count children accepted
        children_accepted = 0
        if children_only:
            children_accepted = self.population_size
        else:
            children_accepted = np.sum(selection_index >= original_population_size)

        return children_accepted

    def write_population_dict_header(self, output_file):
        """Write header for population dict

        Args:
            output_file (file object): Write enabled file object

        """
        output_file.write('\t'.join(self._column_names) + '\n')

    def write_population_dict_values(self, output_file, population_dict=None):
        """Write values for population dict

        Args:
            output_file (file object): Write enabled file object
            population_dict (Dict[str,]): Dictionary used for output

        """
        # default is to write current population
        population_dict_to_write = self._population_dict if population_dict is None else population_dict
        population_size = len(population_dict_to_write[self._data_column_name])
        for i in range(population_size):
            row_data = []
            for key in self._column_names:
                if key == self._data_column_name:
                    row_data.append(population_dict_to_write[key][i])
                elif key == self._niche_column_name:
                    row_data.append(self._niche_to_string(population_dict_to_write[key][i]))
                elif key == 'positive_classes':
                    row_data.append(population_dict_to_write[key][i])
                else:
                    row_data.append('%.6f' % (population_dict_to_write[key][i]))
            output_file.write('\t'.join(row_data) + '\n')

    def write_tokenized_data(self, output_file):
        data = self._population_dict[self._data_column_name]
        for row in data:
            tokens = self._scoring_operator.txt_to_tokens([row])[0]
            tokens = [str(x) for x in tokens]
            output_file.write('%s\n' % ' '.join(tokens))

    def write_class_data(self, output_file):
        if self._niche_column_name is None:
            return
        
        class_data = self._population_dict[self._niche_column_name]
        for niche in class_data:
            classes = self._niche_to_string(niche).split(',')
            output_file.write('%s\n' % ' '.join(classes))

    def _niche_to_string(self, niche_index):
        # return str(niche_index-1)
        format_string = "{0:0%db}" % self._scoring_operator._number_classes
        niche_bit_string = format_string.format(int(niche_index))
        positive_classes = []
        counter = 0
        for c in niche_bit_string:
            if c == '1':
                positive_classes.append(str(counter))
            counter += 1
        if len(positive_classes) == 0:
            positive_classes.append('-1')
        return ','.join(positive_classes)


    def get_population_averages(self):
        """Get average of scoring metrics for population
        
        Returns:
            Dict[str, float] with averages for population metrics

        """
        averages_dict = {}
        for key in self._column_names:
            if (key != self._data_column_name) and (key != self._niche_column_name) and (key != 'positive_classes'):
                averages_dict[key] = np.mean(self._population_dict[key])

        # report number of niches if requested
        if self._niche_column_name is not None:
            niches_covered = len(np.unique(self._population_dict[self._niche_column_name]))
            averages_dict[self._niche_column_name] = niches_covered

        if 'positive_classes' in self._population_dict:
            averages_dict['positive_classes'] = np.mean([len(x.split(',')) for x in self._population_dict['positive_classes']])

        return averages_dict

    def _sample_population_dict(self, number_of_samples, weighted):
        """Return sample from data column of population dict

        Args:
            number_of_samples (int): Number of samples to draw
            weighted (bool): Option to weight samples by softmax of fitness

        Returns:
            List[str] with sampled sequences from data column of population dict

        """
        if weighted:
            # softmax weights from fitness
            weights = np.exp(self._population_dict[self._fitness_column_name])
            weights /= np.sum(weights)
            return np.random.choice(self._population_dict[self._data_column_name], number_of_samples, p=weights)
        else:
            return np.random.choice(self._population_dict[self._data_column_name], number_of_samples)

    def _sequences_to_population_dict(self, sequences, previous_set=None, db_dict=None, return_valid=False):
        """Generation a population_dict from a list of sequences

        Args:
            sequences (List[str]): List of sequences for population
            previous_set (set[str]): Set of previously visited sequences
            db_dict (Dict[str,]): Dictionary with keys cursor and query_string

        Returns:
            Dict[str,] produced by the scoring operator

        """
        # setup defaults
        if previous_set is None:
            previous_set = set()

        if db_dict is None:
            db_dict = {}

        sequences_to_keep = []

        # valid sequences produced
        valid_counter = 0

        for sequence in sequences:

            # check if sequence is viable
            prepared_data = self._scoring_operator.prepare_data_for_scoring(sequence)
            if prepared_data is not None:

                valid_counter += 1

                # attempt to make canonical - for cases like molecules generation where cleaned_data and sequence don't have same type
                canonical_data = self._scoring_operator.make_canonical(prepared_data)

                # check if data has already been recorded
                if canonical_data in previous_set:
                    continue

                # check if data has already been recorded in db
                if ('cursor' in db_dict) and ('query_string' in db_dict):
                    canonical_query = (canonical_data,)
                    cursor = db_dict['cursor']
                    cursor.execute(db_dict['query_string'], canonical_query)
                    if cursor.fetchone()[0] == 1:
                        continue

                # add to population
                sequences_to_keep.append(prepared_data)
                previous_set.add(canonical_data)

        # generate population
        if return_valid:
            return self._scoring_operator.generate_scores(sequences_to_keep), valid_counter
        else:
            return self._scoring_operator.generate_scores(sequences_to_keep)

    def _fill_population_dict(self, desired_size):
        """Fill a population_dict to a desired size by making random copies.

        Args:
            desired_size (int): Desired number of elements for each key in the population_dict

        """
        current_size = len(self._population_dict[self._data_column_name])
        if current_size < desired_size:
            copy_indices = np.random.choice(current_size, desired_size - current_size)
            for index in copy_indices:
                for key in self._population_dict:
                    self._population_dict[key].append(self._population_dict[key][index])