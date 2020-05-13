# Feature Selection examples

The code is for demonstrate feature selection methods that CANDLE provides.


```
$ python M16_test.py

Importing candle utils for keras
Origin =  http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Data_For_Testing/small_drug_descriptor_data_unique_samples.txt
Origin =  http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Data_For_Testing/small_drug_response_data.txt
Origin =  http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Data_For_Testing/Gene_Expression_Full_Data_Unique_Samples.txt
Origin =  http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Data_For_Testing/CCLE_NCI60_Gene_Expression_Full_Data.txt
Origin =  http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Gene_Sets/MSigDB.v7.0/c2.cgp.v7.0.entrez.gmt
Origin =  http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Gene_Sets/MSigDB.v7.0/c2.cgp.v7.0.symbols.gmt
Origin =  http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Gene_Sets/MSigDB.v7.0/c2.cp.biocarta.v7.0.entrez.gmt
Origin =  http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Gene_Sets/MSigDB.v7.0/c2.cp.biocarta.v7.0.symbols.gmt
Origin =  http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Gene_Sets/MSigDB.v7.0/c2.cp.kegg.v7.0.entrez.gmt
Origin =  http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Gene_Sets/MSigDB.v7.0/c2.cp.kegg.v7.0.symbols.gmt
Origin =  http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Gene_Sets/MSigDB.v7.0/c2.cp.pid.v7.0.entrez.gmt
Origin =  http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Gene_Sets/MSigDB.v7.0/c2.cp.pid.v7.0.symbols.gmt
Origin =  http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Gene_Sets/MSigDB.v7.0/c2.cp.reactome.v7.0.entrez.gmt
Origin =  http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Gene_Sets/MSigDB.v7.0/c2.cp.reactome.v7.0.symbols.gmt
Origin =  http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Gene_Sets/MSigDB.v7.0/c5.bp.v7.0.entrez.gmt
Origin =  http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Gene_Sets/MSigDB.v7.0/c5.bp.v7.0.symbols.gmt
Origin =  http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Gene_Sets/MSigDB.v7.0/c5.cc.v7.0.entrez.gmt
Origin =  http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Gene_Sets/MSigDB.v7.0/c5.cc.v7.0.symbols.gmt
Origin =  http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Gene_Sets/MSigDB.v7.0/c5.mf.v7.0.entrez.gmt
Origin =  http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Gene_Sets/MSigDB.v7.0/c5.mf.v7.0.symbols.gmt
Origin =  http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Gene_Sets/MSigDB.v7.0/c6.all.v7.0.entrez.gmt
Origin =  http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/uno/Candle_Milestone_16_Version_12_15_2019/Data/Gene_Sets/MSigDB.v7.0/c6.all.v7.0.symbols.gmt
Gene Set data is locally stored at  /Users/hsyoo/projects/CANDLE/Benchmarks/common/../Data/examples/Gene_Sets/MSigDB.v7.0/


Testing select_features_by_missing_values
Drug descriptor dataframe includes 10 drugs (rows) and 10 drug descriptor features (columns)
               MW     AMW      Sv      Se  ...     Mv  Psi_e_1d  Psi_e_1s  VE3sign_X
Drug_1     475.40   8.804  34.718  54.523  ...  0.643       NaN       NaN        NaN
Drug_10    457.71  10.898  29.154  43.640  ...  0.694       NaN       NaN     -2.752
Drug_100   561.80   6.688  49.975  83.607  ...  0.595       NaN       NaN     -4.335
Drug_1000  362.51   6.840  32.794  52.461  ...  0.619       NaN       NaN     -9.968
Drug_1001  628.83   7.763  51.593  81.570  ...  0.637       NaN       NaN     -2.166
Drug_1002  377.19  10.777  26.191  36.578  ...  0.748       NaN       NaN     -1.526
Drug_1003  371.42   8.254  30.896  45.473  ...  0.687       NaN       NaN     -4.983
Drug_1004  453.60   8.100  37.949  55.872  ...  0.678       NaN       NaN     -4.100
Drug_1005  277.35   7.704  23.940  35.934  ...  0.665       NaN       NaN     -5.234
Drug_1006  409.47   8.189  34.423  50.356  ...  0.688       NaN       NaN     -2.513

[10 rows x 10 columns]
Select features with missing rates smaller than 0.1
Feature IDs [0 1 2 3 4 5 6]
Select features with missing rates smaller than 0.3
Feature IDs [0 1 2 3 4 5 6 9]


Testing select_features_by_variation
Select features with a variance larger than 100
Feature IDs [0 3 5]
Select the top 2 features with the largest standard deviation
Feature IDs [0 5]


Testing select_decorrelated_features
Select features that are not identical to each other and are not all missing.
Feature IDs [0 1 2 3 4 5 6 9]
Select features whose absolute mutual Spearman correlation coefficient is smaller than 0.8
Feature IDs [0 2 6 9]


Testing generate_cross_validation_partition
Generate 5-fold cross-validation partition of 10 samples twice
[[[0, 5], [1, 2, 3, 4, 6, 7, 8, 9]], [[1, 6], [0, 2, 3, 4, 5, 7, 8, 9]], [[2, 7], [0, 1, 3, 4, 5, 6, 8, 9]], [[3, 8], [0, 1, 2, 4, 5, 6, 7, 9]], [[4, 9], [0, 1, 2, 3, 5, 6, 7, 8]], [[5, 8], [0, 1, 2, 3, 4, 6, 7, 9]], [[3, 9], [0, 1, 2, 4, 5, 6, 7, 8]], [[2, 4], [0, 1, 3, 5, 6, 7, 8, 9]], [[1, 7], [0, 2, 3, 4, 5, 6, 8, 9]], [[0, 6], [1, 2, 3, 4, 5, 7, 8, 9]]]
Drug response data of 5 cell lines treated by various drugs.
   SOURCE        CELL     DRUG     AUC   EC50   EC50se   R2fit      HS
0    CCLE  CCLE.22RV1   CCLE.1  0.7153  5.660   0.6867  0.9533  0.6669
1    CCLE  CCLE.22RV1  CCLE.10  0.9579  7.023   0.7111  0.4332  4.0000
2    CCLE  CCLE.22RV1  CCLE.11  0.4130  7.551   0.0385  0.9948  1.3380
3    CCLE  CCLE.22RV1  CCLE.12  0.8004  5.198  11.7100  0.9944  4.0000
4    CCLE  CCLE.22RV1  CCLE.13  0.5071  7.149   0.3175  0.8069  1.0150
..    ...         ...      ...     ...    ...      ...     ...     ...
95   CCLE    CCLE.697  CCLE.12  0.7869  5.278  20.1200  0.8856  4.0000
96   CCLE    CCLE.697  CCLE.13  0.4433  7.474   0.0265  0.9978  3.7080
97   CCLE    CCLE.697  CCLE.14  0.4337  7.466   0.0106  0.9996  3.4330
98   CCLE    CCLE.697  CCLE.15  0.8721  3.097  29.1300  0.4884  0.2528
99   CCLE    CCLE.697  CCLE.16  0.7955  7.496   0.1195  0.9396  1.9560

[100 rows x 8 columns]
Generate partition indices to divide the data into 4 sets without shared cell lines for 5 times.
[[[68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91], [44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67], [92, 93, 94, 95, 96, 97, 98, 99], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]], [[44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67], [92, 93, 94, 95, 96, 97, 98, 99], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91]], [[92, 93, 94, 95, 96, 97, 98, 99], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43], [44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91]], [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43], [68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91], [44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 92, 93, 94, 95, 96, 97, 98, 99]], [[24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43], [68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91], [44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 92, 93, 94, 95, 96, 97, 98, 99]]]
Using TensorFlow backend.
...
found 2 batches
found 0 numerical covariates...
found 0 categorical variables:	
Standardizing Data across genes.
Fitting L/S model and finding priors
Finding parametric adjustments



Testing quantile_normalization
Gene expression data of 897 cell lines (columns) and 17741 genes (rows).
                     CCL_61  CCL_62  CCL_63  ...  CCL_1076  CCL_1077  CCL_1078
entrezID geneSymbol                          ...                              
1        A1BG          0.99    0.03    0.36  ...      2.56      3.55      3.04
29974    A1CF          4.03    3.03    0.00  ...      0.00      0.03      0.00
2        A2M           2.68    0.03    0.16  ...      0.77      0.31      1.20
144568   A2ML1         0.07    0.07    0.01  ...      0.01      0.00      1.09
127550   A3GALT2       0.15    0.00    0.06  ...      2.34      0.00      0.03
...                     ...     ...     ...  ...       ...       ...       ...
440590   ZYG11A        0.41    0.06    1.70  ...      0.75      3.44      2.44
79699    ZYG11B        4.45    4.23    3.08  ...      4.25      3.61      3.68
7791     ZYX           4.65    5.72    6.67  ...      7.78      4.12      5.97
23140    ZZEF1         4.14    3.98    3.90  ...      4.62      3.76      3.54
26009    ZZZ3          4.77    5.01    3.90  ...      4.38      3.46      3.60

[17741 rows x 897 columns]
Before normalization
Max difference of third quartile between cell lines is 1.86
Max difference of median between cell lines is 2.25
Max difference of first quartile between cell lines is 0.5
After normalization
Max difference of third quartile between cell lines is 0.01
Max difference of median between cell lines is 0.02
Max difference of first quartile between cell lines is 0.06


Testing generate_gene_set_data
Generate gene-set-level data of 897 cell lines and 189 oncogenic signature gene sets
          GLI1_UP.V1_DN  GLI1_UP.V1_UP  ...  LEF1_UP.V1_DN  LEF1_UP.V1_UP
CCL_61        -0.031096       0.283946  ...       0.096461      -0.329343
CCL_62         0.362855      -0.101684  ...       0.426951      -0.477634
CCL_63        -0.304989      -0.165160  ...       0.036932      -0.201916
CCL_64        -0.037737      -0.043124  ...       0.154256      -0.210188
CCL_65         0.102477       0.438871  ...      -0.166487       0.287382
...                 ...            ...  ...            ...            ...
CCL_1074       0.508978       0.137934  ...       0.148213       0.166717
CCL_1075      -0.145029       0.216169  ...      -0.067391       0.258455
CCL_1076      -0.357758       0.337235  ...       0.008950       0.186134
CCL_1077       0.086597      -0.266070  ...       0.217244      -0.276022
CCL_1078       0.374237      -0.428383  ...       0.312984      -0.303721

[897 rows x 189 columns]
Generate gene-set-level data of 897 cell lines and 186 KEGG pathways
          KEGG_GLYCOLYSIS_GLUCONEOGENESIS  ...  KEGG_VIRAL_MYOCARDITIS
CCL_61                           6.495365  ...              -30.504868
CCL_62                          30.679006  ...               -7.205641
CCL_63                          10.534238  ...               -5.414998
CCL_64                           6.142140  ...              -10.555601
CCL_65                          -0.303868  ...               -9.784998
...                                   ...  ...                     ...
CCL_1074                        -1.945281  ...                6.891960
CCL_1075                       -21.373730  ...                0.612092
CCL_1076                       -11.711818  ...              -10.353794
CCL_1077                       -11.576702  ...              -31.679962
CCL_1078                       -10.355489  ...              -26.232325

[897 rows x 186 columns]


Testing combat_batch_effect_removal
Gene expression data of 60 NCI60 cell lines and 1018 CCLE cell lines with 17741 genes.
                     NCI60.786-0|CCL_1  ...  CCLE.ZR7530|CCL_1078
entrezID geneSymbol                     ...                      
1        A1BG                     0.00  ...                  3.04
29974    A1CF                     0.00  ...                  0.00
2        A2M                      0.00  ...                  1.20
144568   A2ML1                    0.00  ...                  1.09
127550   A3GALT2                  0.00  ...                  0.03
...                                ...  ...                   ...
440590   ZYG11A                   0.01  ...                  2.44
79699    ZYG11B                   3.37  ...                  3.68
7791     ZYX                      7.05  ...                  5.97
23140    ZZEF1                    4.05  ...                  3.54
26009    ZZZ3                     4.10  ...                  3.60

[17741 rows x 1078 columns]
Before removal of batch effect between NCI60 and CCLE datasets
Average third quartile of NCI60 cell lines is 4.0
Average median of NCI60 cell lines is 1.71
Average first quartile of NCI60 cell lines is 0.01
Average third quartile of CCLE cell lines is 4.88
Average median of CCLE cell lines is 2.75
Average first quartile of CCLE cell lines is 0.14
Adjusting data
After removal of batch effect between NCI60 and CCLE datasets
Average third quartile of NCI60 cell lines is 4.81
Average median of NCI60 cell lines is 2.65
Average first quartile of NCI60 cell lines is 0.23
Average third quartile of CCLE cell lines is 4.83
Average median of CCLE cell lines is 2.72
Average first quartile of CCLE cell lines is 0.13
```