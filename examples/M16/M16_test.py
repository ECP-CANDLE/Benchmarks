import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path2 = os.path.abspath(os.path.join(file_path, "..", "..", "common"))
sys.path.append(lib_path2)

import candle

# download all the data if needed from the repo
data_url = 'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Data_For_Testing/'
file_name = 'small_drug_descriptor_data_unique_samples.txt'
drug_descriptor = candle.get_file(file_name, data_url + file_name, cache_subdir='examples')

file_name = 'small_drug_response_data.txt'
response_data = candle.get_file(file_name, data_url + file_name, cache_subdir='examples')

file_name = 'Gene_Expression_Full_Data_Unique_Samples.txt'
gene_expression = candle.get_file(file_name, data_url + file_name, cache_subdir='examples')

file_name = 'CCLE_NCI60_Gene_Expression_Full_Data.txt'
ccle_nci60 = candle.get_file(file_name, data_url + file_name, cache_subdir='examples')


# download all the gene_set files needed
data_url = 'http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Gene_Sets/MSigDB.v7.0/'
for gene_set_category in ['c2.cgp', 'c2.cp.biocarta', 'c2.cp.kegg', 'c2.cp.pid', 'c2.cp.reactome', 'c5.bp', 'c5.cc', 'c5.mf', 'c6.all']:
    for gene_name_type in ['entrez', 'symbols']:
        file_name = gene_set_category + '.v7.0.' + gene_name_type + '.gmt'
        local_file = candle.get_file(file_name, data_url + file_name, cache_subdir='examples/Gene_Sets/MSigDB.v7.0')
# extract base directory for gene_set data files
data_dir = local_file.split(file_name)[0]
print('Gene Set data is locally stored at ', data_dir)


# Select features based on_missing_values
print('\n')
print('Testing select_features_by_missing_values')
print('Drug descriptor dataframe includes 10 drugs (rows) and 10 drug descriptor features (columns)')
data = pd.read_csv(drug_descriptor, sep='\t', engine='c', na_values=['na', '-', ''], header=0, index_col=0, low_memory=False)
print(data)
print('Select features with missing rates smaller than 0.1')
id = candle.select_features_by_missing_values(data, threshold=0.1)
print('Feature IDs', id)
print('Select features with missing rates smaller than 0.3')
id = candle.select_features_by_missing_values(data.values, threshold=0.3)
print('Feature IDs', id)


# Select features based on variation
print('\n')
print('Testing select_features_by_variation')
print('Select features with a variance larger than 100')
id = candle.select_features_by_variation(data, variation_measure='var', threshold=100, portion=None, draw_histogram=False)
print('Feature IDs', id)
print('Select the top 2 features with the largest standard deviation')
id = candle.select_features_by_variation(data, variation_measure='std', portion=0.2)
print('Feature IDs', id)


# Select decorrelated features
print('\n')
print('Testing select_decorrelated_features')
print('Select features that are not identical to each other and are not all missing.')
id = candle.select_decorrelated_features(data, threshold=None, random_seed=None)
print('Feature IDs', id)
print('Select features whose absolute mutual Spearman correlation coefficient is smaller than 0.8')
id = candle.select_decorrelated_features(data, method='spearman', threshold=0.8, random_seed=10)
print('Feature IDs', id)


# Generate cross-validation partitions of data
print('\n')
print('Testing generate_cross_validation_partition')
print('Generate 5-fold cross-validation partition of 10 samples twice')
p = candle.generate_cross_validation_partition(range(10), n_folds=5, n_repeats=2, portions=None, random_seed=None)
print(p)
print('Drug response data of 5 cell lines treated by various drugs.')
data = pd.read_csv(response_data, sep='\t', engine='c', na_values=['na', '-', ''], header=0, index_col=None, low_memory=False)
print(data)
print('Generate partition indices to divide the data into 4 sets without shared cell lines for 5 times.')
p = candle.generate_cross_validation_partition(data.CELL, n_folds=5, n_repeats=1, portions=[1, 1, 1, 2], random_seed=1)
print(p)


# Quantile normalization of gene expression data
print('\n')
print('Testing quantile_normalization')
print('Gene expression data of 897 cell lines (columns) and 17741 genes (rows).')
data = pd.read_csv(gene_expression, sep='\t', engine='c', na_values=['na', '-', ''], header=0, index_col=[0, 1], low_memory=False)
print(data)
print('Before normalization')
third_quartile = data.quantile(0.75, axis=0)
print('Max difference of third quartile between cell lines is ' + str(np.round(a=np.max(third_quartile) - np.min(third_quartile), decimals=2)))
second_quartile = data.quantile(0.5, axis=0)
print('Max difference of median between cell lines is ' + str(np.round(a=np.max(second_quartile) - np.min(second_quartile), decimals=2)))
first_quartile = data.quantile(0.25, axis=0)
print('Max difference of first quartile between cell lines is ' + str(np.round(a=np.max(first_quartile) - np.min(first_quartile), decimals=2)))
norm_data = candle.quantile_normalization(np.transpose(data))
norm_data = np.transpose(norm_data)
print('After normalization')
third_quartile = norm_data.quantile(0.75, axis=0)
print('Max difference of third quartile between cell lines is ' + str(np.round(a=np.max(third_quartile) - np.min(third_quartile), decimals=2)))
second_quartile = norm_data.quantile(0.5, axis=0)
print('Max difference of median between cell lines is ' + str(np.round(a=np.max(second_quartile) - np.min(second_quartile), decimals=2)))
first_quartile = norm_data.quantile(0.25, axis=0)
print('Max difference of first quartile between cell lines is ' + str(np.round(a=np.max(first_quartile) - np.min(first_quartile), decimals=2)))


# Generate gene-set-level data
print('\n')
print('Testing generate_gene_set_data')
gene_set_data = candle.generate_gene_set_data(np.transpose(norm_data), [i[0] for i in norm_data.index], gene_name_type='entrez',
                                              gene_set_category='c6.all', metric='mean', standardize=True, data_dir=data_dir)
print('Generate gene-set-level data of 897 cell lines and 189 oncogenic signature gene sets')
print(gene_set_data)
gene_set_data = candle.generate_gene_set_data(np.transpose(norm_data), [i[1] for i in norm_data.index], gene_name_type='symbols',
                                              gene_set_category='c2.cp.kegg', metric='sum', standardize=True, data_dir=data_dir)
print('Generate gene-set-level data of 897 cell lines and 186 KEGG pathways')
print(gene_set_data)


# Combat batch normalization on gene expression data
print('\n')
print('Testing combat_batch_effect_removal')
print('Gene expression data of 60 NCI60 cell lines and 1018 CCLE cell lines with 17741 genes.')
data = pd.read_csv(ccle_nci60, sep='\t', engine='c', na_values=['na', '-', ''], header=0, index_col=[0, 1], low_memory=False)
print(data)
resource = np.array([i.split('.')[0] for i in data.columns])

print('Before removal of batch effect between NCI60 and CCLE datasets')

# Identify NCI60 cell lines and quantile normalize their gene expression data
id = np.where(resource == 'NCI60')[0]
norm_data_NCI60 = candle.quantile_normalization(np.transpose(data.iloc[:, id]))
print('Average third quartile of NCI60 cell lines is ' + str(np.round(a=np.mean(norm_data_NCI60.quantile(0.75, axis=1)), decimals=2)))
print('Average median of NCI60 cell lines is ' + str(np.round(a=np.mean(norm_data_NCI60.quantile(0.5, axis=1)), decimals=2)))
print('Average first quartile of NCI60 cell lines is ' + str(np.round(a=np.mean(norm_data_NCI60.quantile(0.25, axis=1)), decimals=2)))

# Identify CCLE cell lines and quantile normalize their gene expression data
id = np.where(resource == 'CCLE')[0]
norm_data_CCLE = candle.quantile_normalization(np.transpose(data.iloc[:, id]))
print('Average third quartile of CCLE cell lines is ' + str(np.round(a=np.mean(norm_data_CCLE.quantile(0.75, axis=1)), decimals=2)))
print('Average median of CCLE cell lines is ' + str(np.round(a=np.mean(norm_data_CCLE.quantile(0.5, axis=1)), decimals=2)))
print('Average first quartile of CCLE cell lines is ' + str(np.round(a=np.mean(norm_data_CCLE.quantile(0.25, axis=1)), decimals=2)))

# Combine normalized data of NCI60 cell lines and CCLE cell lines
norm_data = pd.concat((norm_data_NCI60, norm_data_CCLE), axis=0)
norm_data = np.transpose(norm_data)

# Apply ComBat algorithm to remove the batch effect between NCI60 and CCLE
corrected_data = candle.combat_batch_effect_removal(norm_data, pd.Series([i.split('.')[0] for i in norm_data.columns], index=norm_data.columns))

print('After removal of batch effect between NCI60 and CCLE datasets')

resource = np.array([i.split('.')[0] for i in corrected_data.columns])
id = np.where(resource == 'NCI60')[0]
corrected_data_NCI60 = np.transpose(corrected_data.iloc[:, id])
print('Average third quartile of NCI60 cell lines is ' + str(np.round(a=np.mean(corrected_data_NCI60.quantile(0.75, axis=1)), decimals=2)))
print('Average median of NCI60 cell lines is ' + str(np.round(a=np.mean(corrected_data_NCI60.quantile(0.5, axis=1)), decimals=2)))
print('Average first quartile of NCI60 cell lines is ' + str(np.round(a=np.mean(corrected_data_NCI60.quantile(0.25, axis=1)), decimals=2)))

# Identify CCLE cell lines and quantile normalize their gene expression data
id = np.where(resource == 'CCLE')[0]
corrected_data_CCLE = np.transpose(corrected_data.iloc[:, id])
print('Average third quartile of CCLE cell lines is ' + str(np.round(a=np.mean(corrected_data_CCLE.quantile(0.75, axis=1)), decimals=2)))
print('Average median of CCLE cell lines is ' + str(np.round(a=np.mean(corrected_data_CCLE.quantile(0.5, axis=1)), decimals=2)))
print('Average first quartile of CCLE cell lines is ' + str(np.round(a=np.mean(corrected_data_CCLE.quantile(0.25, axis=1)), decimals=2)))
