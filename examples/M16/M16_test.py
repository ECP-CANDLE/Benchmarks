import os
import sys
import pandas as pd
import numpy as np
import keras

#from Milestone_16_Functions import select_features_by_missing_values, select_features_by_variation, select_decorrelated_features, \
#quantile_normalization, generate_cross_validation_partition, generate_gene_set_data, combat_batch_effect_removal

file_path = os.path.dirname(os.path.realpath(__file__))
# lib_path = os.path.abspath(os.path.join(file_path, '..'))
# sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, "..", "..", "common"))
sys.path.append(lib_path2)

import candle

# download all the data if needed from the repo
data_url = 'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Data_For_Testing/'
file_name = 'small_drug_descriptor_data_unique_samples.txt'
unique_samples = candle.get_file(file_name, data_url+file_name, cache_subdir='examples')

file_name = 'small_drug_response_data.txt'
response_data = candle.get_file(file_name, data_url+file_name, cache_subdir='examples')

file_name = 'Gene_Expression_Full_Data_Unique_Samples.txt'
gene_expression = candle.get_file(file_name, data_url+file_name, cache_subdir='examples')

file_name = 'CCLE_NCI60_Gene_Expression_Full_Data.txt'
ccle_nci60 = candle.get_file(file_name, data_url+file_name, cache_subdir='examples')

# download all the gene_set files needed
data_url = 'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Gene_Sets/MSigDB.v7.0/'
for gene_set_category in ['c2.cgp','c2.cp.biocarta','c2.cp.kegg','c2.cp.pid','c2.cp.reactome','c5.bp','c5.cc','c5.mf','c6.all']:
    for gene_name_type in ['entrez', 'symbols']:
        file_name = gene_set_category+'.v7.0.'+gene_name_type+'.gmt'
        local_file = candle.get_file(file_name, data_url+file_name, cache_subdir='examples/Gene_Sets/MSigDB.v7.0')
# extract base directory for gene_set data files
data_dir = local_file.split(file_name)[0]
print('Gene Set data is locally stored at ',data_dir)

# Select features based on_missing_values

print('Original dataframe')
data = pd.read_csv(unique_samples, sep='\t', engine='c',
                   na_values=['na', '-', ''], header=0, index_col=0, low_memory=False)
print(data)

print('Testing select_features_by_missing values')
print('Threshold - 0.1')
id = candle.select_features_by_missing_values(data, threshold=0.1)
print(id)
print('Threshold - 0.3')
id = candle.select_features_by_missing_values(data.values, threshold=0.3)
print(id)

# Select features based on variation

#data = pd.read_csv(unique_samples, sep='\t', engine='c',
#                   na_values=['na', '-', ''], header=0, index_col=0, low_memory=False)
print('Testing select_features_by_variation')
print('Variabce, 100')
id = candle.select_features_by_variation(data, variation_measure='var', threshold=100, portion=None,
                             draw_histogram=False)
print(id)
print('std, 0.2')
id = candle.select_features_by_variation(data, variation_measure='std', portion=0.2)
print(id)



# Select uncorrelated features

#data = pd.read_csv(unique_samples, sep='\t', engine='c',
#                   na_values=['na', '-', ''], header=0, index_col=0, low_memory=False)
print('Testing select_decorrelated_features')
print('Pearson')
id = candle.select_decorrelated_features(data, method='pearson', threshold=None, random_seed=None)
print(id)
print('Spearman')
id = candle.select_decorrelated_features(data, method='spearman', threshold=0.8, random_seed=10)
print(id)



# Generate cross-validation partitions of data

data = pd.read_csv(response_data,
                   sep='\t', engine='c', na_values=['na', '-', ''], header=0, index_col=None, low_memory=False)
p = candle.generate_cross_validation_partition(range(10), n_folds=5, n_repeats=2, portions=None, random_seed=None)
p = candle.generate_cross_validation_partition(data.CELL, n_folds=5, n_repeats=1, portions=[1, 1, 1, 2], random_seed=1)



# Generate gene-set-level data

data = pd.read_csv(gene_expression, sep='\t', engine='c',
                   na_values=['na', '-', ''], header=0, index_col=[0, 1], low_memory=False)
data = data.iloc[:5000, :]
gene_set_data = candle.generate_gene_set_data(np.transpose(data), [i[0] for i in data.index], gene_name_type='entrez',
                                       gene_set_category='c6.all', metric='mean', standardize=False, data_dir=data_dir)
gene_set_data = candle.generate_gene_set_data(np.transpose(data.values), [i[1] for i in data.index], gene_name_type='symbols',
                                       gene_set_category='c2.cp.kegg', metric='sum', standardize=False, data_dir=data_dir)



# Quantile normalization of gene expression data

data = pd.read_csv(gene_expression, sep='\t', engine='c',
                   na_values=['na', '-', ''], header=0, index_col=[0, 1], low_memory=False)
norm_data = candle.quantile_normalization(np.transpose(data))



# Combat batch normalization on gene expression data

data = pd.read_csv(ccle_nci60,
                   sep='\t', engine='c', na_values=['na', '-', ''], header=0, index_col=[0, 1], low_memory=False)

resource = np.array([i.split('.')[0] for i in data.columns])
id = np.where(resource == 'NCI60')[0]
norm_data_NCI60 = candle.quantile_normalization(np.transpose(data.iloc[:, id]))
id = np.where(resource == 'CCLE')[0]
norm_data_CCLE = candle.quantile_normalization(np.transpose(data.iloc[:, id]))
norm_data = pd.concat((norm_data_NCI60, norm_data_CCLE), axis=0)
norm_data = np.transpose(norm_data)
corrected_data = candle.combat_batch_effect_removal(norm_data, pd.Series([i.split('.')[0] for i in norm_data.columns], index=norm_data.columns))

