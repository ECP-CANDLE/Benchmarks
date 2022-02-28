from evolution import Evolution
from population import Population
from cnn_scoring import DocumentScoring
import json
import sys
import argparse
from datetime import datetime
import numpy as np

##########################################################################################
# Initial Setup
##########################################################################################

parser = argparse.ArgumentParser()

# config for run
parser.add_argument('config', help='json file with config for run')
parser.add_argument('input_file', type=str, help='input file for optimization')
parser.add_argument('run_id', type=str, help='run_id for output')
parser.add_argument('--device', type=int, default=1)
args = parser.parse_args()
print(args)

# read config for run
with open(args.config, 'r') as f:
    config = json.load(f)

# run parameters from config
input_file = args.input_file
run_id = args.run_id
run_params = config['run_params']
output_directory = run_params['output_directory']
generations = run_params['generations']
population_size = run_params['population_size']
mutation_samples = run_params['mutation_samples']
recombination_samples = run_params['recombination_samples']
weighted_evolution = bool(run_params['weighted_evolution'])
batch_size = run_params['batch_size']
mimic_prefix = run_params['mimic_prefix']

# open file for final population
population_output_file = open(output_directory + '/' + run_id + '_population.tsv', 'w')

# special file locations for mimic data
mimic_train_x_file = open(mimic_prefix + '_train_x_' + run_id + '.txt', 'w')
mimic_train_y_file = open(mimic_prefix + '_train_y_' + run_id + '.txt', 'w')

# special file locations for initial population data
mimic_train_x_init_file = open(mimic_prefix + '_train_x_' + run_id + '_init.txt', 'w')
mimic_train_y_init_file = open(mimic_prefix + '_train_y_' + run_id + '_init.txt', 'w')

##########################################################################################
# Setup Evolution Operators
##########################################################################################

print('Initializing Mutation Operators', flush=True)

evolution_operators = []
for operator_config in config['evolution_operators']:
    operator_config['device'] = args.device
    evolution_operators.append(Evolution(**operator_config))

##########################################################################################
# Setup Scoring
##########################################################################################

print('Initializing Scoring Operator', flush=True)

config['scoring_operator']['scoring_parameters']['device'] = args.device
scoring_operator = DocumentScoring(**config['scoring_operator'])

##########################################################################################
# Generate Starting Population from Input File
##########################################################################################

# construct population
population = Population(evolution_operators, scoring_operator, 'niche_index')

print('Reading Initial Sequences', flush=True)

population.read_population_dict_from_file(input_file, population_size=population_size, fill_to_size=False)

print('Population size:', population.population_size)

# check that population is not emtpy
if population.population_size < 1:
    print('Error: unable to read any sequences for population')
    sys.exit(1)

# set to keep track of generated sequences
previous_sequences = set(population.population_sequences)

# set to store original sequences
original_sequences = set(population.population_sequences)

# write headers for output files
population.write_population_dict_header(population_output_file)

# write init output files
population.write_tokenized_data(mimic_train_x_init_file)
population.write_class_data(mimic_train_y_init_file)
mimic_train_x_init_file.close()
mimic_train_y_init_file.close()

##########################################################################################
# Generations of Mutation and Recombination
##########################################################################################

print('Iterating Population', flush=True)

for generation in range(generations):

    # generate new sequences with mutation and recombination
    child_population_dict = population.generate_child_population_dict(mutation_samples, recombination_samples, weighted_evolution, previous_sequences, {}, batch_size)
    children_total = len(child_population_dict[scoring_operator.data_column_name])

    # eliminate children with zero fitness
    children_fitness = child_population_dict[population._fitness_column_name]
    non_zero_indices = np.nonzero(children_fitness)[0]
    children_positive = children_total
    if len(non_zero_indices) < len(children_fitness):
        for key in child_population_dict:
            child_population_dict[key] = [child_population_dict[key][x] for x in non_zero_indices]
        children_positive = len(non_zero_indices)

    # merge population
    children_accepted = 0
    if generation < (generations - 1):
        children_accepted = population.merge_child_population_dict(child_population_dict, max_size=population_size)
    else:
        children_accepted = population.merge_child_population_dict(child_population_dict, max_size=population_size, exclude_sequences=original_sequences)

    # updated population averages
    population_averages = population.get_population_averages()

    # report on sequences surviving from original sequences
    current_sequences = set(population.population_sequences)
    overlap_count = len(original_sequences.intersection(current_sequences))

    # write to std out - time and novel
    print('[%d/%d] time: %s\tnovel: %d\taccepted: %d\toriginal: %d\ttotal: %d'
        % (generation+1, generations,
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        children_total,
        children_accepted,
        overlap_count,
        population.population_size), flush=True, end='\t')

    # write to std out - population averages
    for key in population_averages:
        print('%s: %.4f' % (key, population_averages[key]), end='\t')
    print(flush=True)
    
##########################################################################################
# Clean-up
##########################################################################################

# write population to file
population.write_population_dict_values(population_output_file)
population.write_tokenized_data(mimic_train_x_file)
population.write_class_data(mimic_train_y_file)

# close population output file
population_output_file.close()
mimic_train_x_file.close()
mimic_train_y_file.close()