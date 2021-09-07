import sys
import numpy as np
import numpy.linalg as la
import pandas as pd
import patsy
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

from feature_selection_utils import select_features_by_variation


# Auxiliary functions of COXEN start here ####################
def calculate_concordance_correlation_coefficient(u, v):
    '''
    This function calculates the concordance correlation coefficient between two input 1-D numpy arrays.

    Parameters:
    -----------
    u: 1-D numpy array of a variable
    v: 1-D numpy array of a variable

    Returns:
    --------
    ccc: a numeric value of concordance correlation coefficient between the two input variables.
    '''
    a = 2 * np.mean((u - np.mean(u)) * (v - np.mean(v)))
    b = np.mean(np.square(u - np.mean(u))) + np.mean(np.square(v - np.mean(v))) + np.square(np.mean(u) - np.mean(v))
    ccc = a / b
    return ccc


def generalization_feature_selection(data1, data2, measure, cutoff):
    '''
    This function uses the Pearson correlation coefficient to select the features that are generalizable
    between data1 and data2.

    Parameters:
    -----------
    data1: 2D numpy array of the first dataset with a size of (n_samples_1, n_features)
    data2: 2D numpy array of the second dataset with a size of (n_samples_2, n_features)
    measure: string. 'pearson' indicates the Pearson correlation coefficient;
        'ccc' indicates the concordance correlation coefficient. Default is 'pearson'.
    cutoff: a positive number for selecting generalizable features. If cutoff < 1, this function selects
        the features with a correlation coefficient >= cutoff. If cutoff >= 1, it must be an
        integer indicating the number of features to be selected based on correlation coefficient.

    Returns:
    --------
    fid: 1-D numpy array containing the indices of selected features.
    '''
    cor1 = np.corrcoef(np.transpose(data1))
    cor2 = np.corrcoef(np.transpose(data2))
    num = data1.shape[1]
    cor = []
    if measure == 'pearson':
        for i in range(num):
            cor.append(np.corrcoef(np.vstack((list(cor1[:i, i]) + list(cor1[(i + 1):, i]),
                       list(cor2[:i, i]) + list(cor2[(i + 1):, i]))))[0, 1])
    elif measure == 'ccc':
        for i in range(num):
            cor.append(calculate_concordance_correlation_coefficient(np.array(list(cor1[:i, i]) + list(cor1[(i + 1):, i])),
                       np.array(list(cor2[:i, i]) + list(cor2[(i + 1):, i]))))
    cor = np.array(cor)
    fid = np.argsort(-cor)[:int(cutoff)]
    return fid


# Auxiliary functions of COXEN end here ####################
def coxen_single_drug_gene_selection(
    source_data, target_data, drug_response_data, drug_response_col, tumor_col,
    prediction_power_measure='pearson', num_predictive_gene=100, generalization_power_measure='ccc',
        num_generalizable_gene=50, multi_drug_mode=False):
    '''
    This function selects genes for drug response prediction using the COXEN approach. The COXEN approach is
    designed for selecting genes to predict the response of tumor cells to a specific drug. This function
    assumes no missing data exist.

    Parameters:
    -----------
    source_data: pandas data frame of gene expressions of tumors, for which drug response is known. Its size is
        [n_source_samples, n_features].
    target_data: pandas data frame of gene expressions of tumors, for which drug response needs to be predicted.
        Its size is [n_target_samples, n_features]. source_data and target_data have the same set
        of features and the orders of features must match.
    drug_response_data: pandas data frame of drug response values for a drug. It must include a column of drug
        response values and a column of tumor IDs.
    drug_response_col: non-negative integer or string. If integer, it is the column index of drug response in
        drug_response_data. If string, it is the column name of drug response.
    tumor_col: non-negative integer or string. If integer, it is the column index of tumor IDs in drug_response_data.
        If string, it is the column name of tumor IDs.
    prediction_power_measure: string. 'pearson' uses the absolute value of Pearson correlation coefficient to
        measure prediction power of gene; 'mutual_info' uses the mutual information to measure prediction power
        of gene. Default is 'pearson'.
    num_predictive_gene: positive integer indicating the number of predictive genes to be selected.
    generalization_power_measure: string. 'pearson' indicates the Pearson correlation coefficient;
        'ccc' indicates the concordance correlation coefficient. Default is 'ccc'.
    num_generalizable_gene: positive integer indicating the number of generalizable genes to be selected.
    multi_drug_mode: boolean, indicating whether the function runs as an auxiliary function of COXEN
        gene selection for multiple drugs. Default is False.

    Returns:
    --------
    indices: 1-D numpy array containing the indices of selected genes, if multi_drug_mode is False;
    1-D numpy array of indices of sorting all genes according to their prediction power, if multi_drug_mode is True.
    '''

    if isinstance(drug_response_col, str):
        drug_response_col = np.where(drug_response_data.columns == drug_response_col)[0][0]

    if isinstance(tumor_col, str):
        tumor_col = np.where(drug_response_data.columns == tumor_col)[0][0]

    drug_response_data = drug_response_data.copy()
    drug_response_data = drug_response_data.iloc[np.where(np.isin(drug_response_data.iloc[:, tumor_col],
                                                                  source_data.index))[0], :]

    source_data = source_data.copy()
    source_data = source_data.iloc[np.where(np.isin(source_data.index, drug_response_data.iloc[:, tumor_col]))[0], :]

    source_std_id = select_features_by_variation(source_data, variation_measure='std', threshold=0.00000001)
    target_std_id = select_features_by_variation(target_data, variation_measure='std', threshold=0.00000001)
    std_id = np.sort(np.intersect1d(source_std_id, target_std_id))
    source_data = source_data.iloc[:, std_id]
    target_data = target_data.copy()
    target_data = target_data.iloc[:, std_id]

    # Perform the first step of COXEN approach to select predictive genes. To avoid exceeding the memory limit,
    # the prediction power of genes is calculated in batches.
    batchSize = 1000
    numBatch = int(np.ceil(source_data.shape[1] / batchSize))
    prediction_power = np.empty((source_data.shape[1], 1))
    prediction_power.fill(np.nan)
    for i in range(numBatch):
        startIndex = i * batchSize
        endIndex = min((i + 1) * batchSize, source_data.shape[1])

        if prediction_power_measure == 'pearson':
            cor_i = np.corrcoef(np.vstack(
                (np.transpose(source_data.iloc[:, startIndex:endIndex].loc[drug_response_data.iloc[:, tumor_col], :].values),
                 np.reshape(drug_response_data.iloc[:, drug_response_col].values, (1, drug_response_data.shape[0])))))
            prediction_power[startIndex:endIndex, 0] = abs(cor_i[:-1, -1])

        if prediction_power_measure == 'mutual_info':
            mi = mutual_info_regression(X=source_data.iloc[:, startIndex:endIndex].loc[drug_response_data.iloc[:, tumor_col], :].values,
                                        y=drug_response_data.iloc[:, drug_response_col].values)
            prediction_power[startIndex:endIndex, 0] = mi

    if multi_drug_mode:
        indices = np.argsort(-prediction_power[:, 0])
        return std_id[indices]

    num_predictive_gene = int(min(num_predictive_gene, source_data.shape[1]))
    gid1 = np.argsort(-prediction_power[:, 0])[:num_predictive_gene]

    # keep only predictive genes for source and target data
    source_data = source_data.iloc[:, gid1]
    target_data = target_data.iloc[:, gid1]
    num_generalizable_gene = int(min(num_generalizable_gene, len(gid1)))
    # perform the second step of COXEN approach to select generalizable genes among the predictive genes
    gid2 = generalization_feature_selection(source_data.values, target_data.values, generalization_power_measure,
                                            num_generalizable_gene)

    indices = std_id[gid1[gid2]]

    return np.sort(indices)


def coxen_multi_drug_gene_selection(
    source_data, target_data, drug_response_data, drug_response_col, tumor_col, drug_col,
    prediction_power_measure='lm', num_predictive_gene=100, generalization_power_measure='ccc',
        num_generalizable_gene=50, union_of_single_drug_selection=False):
    '''
    This function uses the COXEN approach to select genes for predicting the response of multiple drugs.
    It assumes no missing data exist. It works in three modes.
    (1) If union_of_single_drug_selection is True, prediction_power_measure must be either 'pearson' or 'mutual_info'.
    This functions runs coxen_single_drug_gene_selection for every drug with the parameter setting and takes the
    union of the selected genes of every drug as the output. The size of the selected gene set may be larger than
    num_generalizable_gene.
    (2) If union_of_single_drug_selection is False and prediction_power_measure is 'lm', this function uses a
    linear model to fit the response of multiple drugs using the expression of a gene, while the drugs are
    one-hot encoded. The p-value associated with the coefficient of gene expression is used as the prediction
    power measure, according to which num_predictive_gene genes will be selected. Then, among the predictive
    genes, num_generalizable_gene generalizable genes will be  selected.
    (3) If union_of_single_drug_selection is False and prediction_power_measure is 'pearson' or 'mutual_info',
    for each drug this functions ranks the genes according to their power of predicting the
    response of the drug. The union of an equal number of predictive genes for every drug will be generated,
    and its size must be at least num_predictive_gene. Then, num_generalizable_gene generalizable genes
    will be selected.

    Parameters:
    -----------
    source_data: pandas data frame of gene expressions of tumors, for which drug response is known. Its size is
        [n_source_samples, n_features].
    target_data: pandas data frame of gene expressions of tumors, for which drug response needs to be predicted.
        Its size is [n_target_samples, n_features]. source_data and target_data have the same set
        of features and the orders of features must match.
    drug_response_data: pandas data frame of drug response that must include a column of drug response values,
        a column of tumor IDs, and a column of drug IDs.
    drug_response_col: non-negative integer or string. If integer, it is the column index of drug response in
        drug_response_data. If string, it is the column name of drug response.
    tumor_col: non-negative integer or string. If integer, it is the column index of tumor IDs in drug_response_data.
        If string, it is the column name of tumor IDs.
    drug_col: non-negative integer or string. If integer, it is the column index of drugs in drug_response_data.
        If string, it is the column name of drugs.
    prediction_power_measure: string. 'pearson' uses the absolute value of Pearson correlation coefficient to
        measure prediction power of a gene; 'mutual_info' uses the mutual information to measure prediction power
        of a gene; 'lm' uses the linear regression model to select predictive genes for multiple drugs. Default is 'lm'.
    num_predictive_gene: positive integer indicating the number of predictive genes to be selected.
    generalization_power_measure: string. 'pearson' indicates the Pearson correlation coefficient;
        'ccc' indicates the concordance correlation coefficient. Default is 'ccc'.
    num_generalizable_gene: positive integer indicating the number of generalizable genes to be selected.
    union_of_single_drug_selection: boolean, indicating whether the final gene set should be the union of genes
        selected for every drug.

    Returns:
    --------
    indices: 1-D numpy array containing the indices of selected genes.
    '''

    if isinstance(drug_response_col, str):
        drug_response_col = np.where(drug_response_data.columns == drug_response_col)[0][0]

    if isinstance(tumor_col, str):
        tumor_col = np.where(drug_response_data.columns == tumor_col)[0][0]

    if isinstance(drug_col, str):
        drug_col = np.where(drug_response_data.columns == drug_col)[0][0]

    drug_response_data = drug_response_data.copy()
    drug_response_data = drug_response_data.iloc[np.where(np.isin(drug_response_data.iloc[:, tumor_col],
                                                                  source_data.index))[0], :]
    drugs = np.unique(drug_response_data.iloc[:, drug_col])

    source_data = source_data.copy()
    source_data = source_data.iloc[np.where(np.isin(source_data.index, drug_response_data.iloc[:, tumor_col]))[0], :]

    source_std_id = select_features_by_variation(source_data, variation_measure='std', threshold=0.00000001)
    target_std_id = select_features_by_variation(target_data, variation_measure='std', threshold=0.00000001)
    std_id = np.sort(np.intersect1d(source_std_id, target_std_id))
    source_data = source_data.iloc[:, std_id]
    target_data = target_data.copy()
    target_data = target_data.iloc[:, std_id]

    num_predictive_gene = int(min(num_predictive_gene, source_data.shape[1]))

    if union_of_single_drug_selection:
        if prediction_power_measure != 'pearson' and prediction_power_measure != 'mutual_info':
            print('pearson or mutual_info must be used as prediction_power_measure for taking the union of selected genes of every drugs')
            sys.exit(1)
        gid1 = np.array([]).astype(np.int64)
        for d in drugs:
            idd = np.where(drug_response_data.iloc[:, drug_col] == d)[0]
            response_d = drug_response_data.iloc[idd, :]
            gid2 = coxen_single_drug_gene_selection(source_data, target_data, response_d, drug_response_col, tumor_col,
                                                    prediction_power_measure, num_predictive_gene, generalization_power_measure, num_generalizable_gene)
            gid1 = np.union1d(gid1, gid2)
        return np.sort(std_id[gid1])

    if prediction_power_measure == 'lm':
        pvalue = np.empty((source_data.shape[1], 1))
        pvalue.fill(np.nan)
        drug_m = np.identity(len(drugs))
        drug_m = pd.DataFrame(drug_m, index=drugs)
        drug_sample = drug_m.loc[drug_response_data.iloc[:, drug_col], :].values
        for i in range(source_data.shape[1]):
            ge_sample = source_data.iloc[:, i].loc[drug_response_data.iloc[:, tumor_col]].values
            sample = np.hstack((np.reshape(ge_sample, (len(ge_sample), 1)), drug_sample))
            sample = sm.add_constant(sample)
            mod = sm.OLS(drug_response_data.iloc[:, drug_response_col].values, sample)
            try:
                res = mod.fit()
                pvalue[i, 0] = res.pvalues[1]
            except ValueError:
                pvalue[i, 0] = 1

        gid1 = np.argsort(pvalue[:, 0])[:num_predictive_gene]

    elif prediction_power_measure == 'pearson' or prediction_power_measure == 'mutual_info':
        gene_rank = np.empty((len(drugs), source_data.shape[1]))
        gene_rank.fill(np.nan)
        gene_rank = pd.DataFrame(gene_rank, index=drugs)
        for d in range(len(drugs)):
            idd = np.where(drug_response_data.iloc[:, drug_col] == drugs[d])[0]
            response_d = drug_response_data.iloc[idd, :]
            temp_rank = coxen_single_drug_gene_selection(
                source_data, target_data, response_d,
                drug_response_col, tumor_col, prediction_power_measure, num_predictive_gene=None,
                generalization_power_measure=None, num_generalizable_gene=None, multi_drug_mode=True)
            gene_rank.iloc[d, :len(temp_rank)] = temp_rank
        for i in range(int(np.ceil(num_predictive_gene / len(drugs))), source_data.shape[1] + 1):
            gid1 = np.unique(np.reshape(gene_rank.iloc[:, :i].values, (1, gene_rank.shape[0] * i))[0, :])
            gid1 = gid1[np.where(np.invert(np.isnan(gid1)))[0]]
            if len(gid1) >= num_predictive_gene:
                break
        gid1 = gid1.astype(np.int64)

    # keep only predictive genes for source and target data
    source_data = source_data.iloc[:, gid1]
    target_data = target_data.iloc[:, gid1]
    num_generalizable_gene = int(min(num_generalizable_gene, len(gid1)))

    # perform the second step of COXEN approach to select generalizable genes among the predictive genes
    gid2 = generalization_feature_selection(source_data.values, target_data.values, generalization_power_measure,
                                            num_generalizable_gene)

    indices = std_id[gid1[gid2]]

    return np.sort(indices)


def generate_gene_set_data(data, genes, gene_name_type='entrez', gene_set_category='c6.all', metric='mean',
                           standardize=False, data_dir='../../Data/examples/Gene_Sets/MSigDB.v7.0/'):
    '''
    This function generates genomic data summarized at the gene set level.

    Parameters:
    -----------
    data: numpy array or pandas data frame of numeric values, with a shape of [n_samples, n_features].
    genes: 1-D array or list of gene names with a length of n_features. It indicates which gene a genomic
        feature belongs to.
    gene_name_type: string, indicating the type of gene name used in genes. 'entrez' indicates Entrez gene ID and
        'symbols' indicates HGNC gene symbol. Default is 'symbols'.
    gene_set_category: string, indicating the gene sets for which data will be calculated. 'c2.cgp' indicates gene sets
        affected by chemical and genetic perturbations; 'c2.cp.biocarta' indicates BioCarta gene sets; 'c2.cp.kegg'
        indicates KEGG gene sets; 'c2.cp.pid' indicates PID gene sets; 'c2.cp.reactome' indicates Reactome gene sets;
        'c5.bp' indicates GO biological processes; 'c5.cc' indicates GO cellular components; 'c5.mf' indicates
        GO molecular functions; 'c6.all' indicates oncogenic signatures. Default is 'c6.all'.
    metric: string, indicating the way to calculate gene-set-level data. 'mean' calculates the mean of gene
        features belonging to the same gene set. 'sum' calculates the summation of gene features belonging
        to the same gene set. 'max' calculates the maximum of gene features. 'min' calculates the minimum
        of gene features. 'abs_mean' calculates the mean of absolute values. 'abs_maximum' calculates
        the maximum of absolute values. Default is 'mean'.
    standardize: boolean, indicating whether to standardize features before calculation. Standardization transforms
        each feature to have a zero mean and a unit standard deviation.

    Returns:
    --------
    gene_set_data: a data frame of calculated gene-set-level data. Column names are the gene set names.
    '''

    sample_name = None
    if isinstance(data, pd.DataFrame):
        sample_name = data.index
        data = data.values
    elif not isinstance(data, np.ndarray):
        print('Input data must be a numpy array or pandas data frame')
        sys.exit(1)

    if standardize:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    genes = [str(i) for i in genes]

    if gene_name_type == 'entrez':
        gene_set_category = gene_set_category + '.v7.0.entrez.gmt'
    if gene_name_type == 'symbols':
        gene_set_category = gene_set_category + '.v7.0.symbols.gmt'
    f = open(data_dir + gene_set_category, 'r')
    x = f.readlines()
    gene_sets = {}
    for i in range(len(x)):
        temp = x[i].split('\n')[0].split('\t')
        gene_sets[temp[0]] = temp[2:]

    gene_set_data = np.empty((data.shape[0], len(gene_sets)))
    gene_set_data.fill(np.nan)
    gene_set_names = np.array(list(gene_sets.keys()))
    for i in range(len(gene_set_names)):
        idi = np.where(np.isin(genes, gene_sets[gene_set_names[i]]))[0]
        if len(idi) > 0:
            if metric == 'sum':
                gene_set_data[:, i] = np.nansum(data[:, idi], axis=1)
            elif metric == 'max':
                gene_set_data[:, i] = np.nanmax(data[:, idi], axis=1)
            elif metric == 'min':
                gene_set_data[:, i] = np.nanmin(data[:, idi], axis=1)
            elif metric == 'abs_mean':
                gene_set_data[:, i] = np.nanmean(np.absolute(data[:, idi]), axis=1)
            elif metric == 'abs_maximum':
                gene_set_data[:, i] = np.nanmax(np.absolute(data[:, idi]), axis=1)
            else:  # 'mean'
                gene_set_data[:, i] = np.nanmean(data[:, idi], axis=1)

    if sample_name is None:
        gene_set_data = pd.DataFrame(gene_set_data, columns=gene_set_names)
    else:
        gene_set_data = pd.DataFrame(gene_set_data, columns=gene_set_names, index=sample_name)
    keep_id = np.where(np.sum(np.invert(pd.isna(gene_set_data)), axis=0) > 0)[0]
    gene_set_data = gene_set_data.iloc[:, keep_id]

    return gene_set_data


# Auxiliary functions of ComBat start here ####################
def design_mat(mod, numerical_covariates, batch_levels):
    # require levels to make sure they are in the same order as we use in the
    # rest of the script.
    design = patsy.dmatrix("~ 0 + C(batch, levels=%s)" % str(batch_levels),
                           mod, return_type="dataframe")

    mod = mod.drop(["batch"], axis=1)
    numerical_covariates = list(numerical_covariates)
    sys.stderr.write("found %i batches\n" % design.shape[1])
    other_cols = [c for i, c in enumerate(mod.columns)
                  if i not in numerical_covariates]
    factor_matrix = mod[other_cols]
    design = pd.concat((design, factor_matrix), axis=1)
    if numerical_covariates is not None:
        sys.stderr.write("found %i numerical covariates...\n" % len(numerical_covariates))
        for i, nC in enumerate(numerical_covariates):
            cname = mod.columns[nC]
            sys.stderr.write("\t{0}\n".format(cname))
            design[cname] = mod[mod.columns[nC]]
    sys.stderr.write("found %i categorical variables:" % len(other_cols))
    sys.stderr.write("\t" + ", ".join(other_cols) + '\n')
    return design


def it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=0.0001):
    n = (1 - np.isnan(sdat)).sum(axis=1)
    g_old = g_hat.copy()
    d_old = d_hat.copy()

    change = 1
    count = 0
    while change > conv:
        # print g_hat.shape, g_bar.shape, t2.shape
        g_new = postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = ((sdat - np.dot(g_new.values.reshape((g_new.shape[0], 1)), np.ones((1, sdat.shape[1])))) ** 2).sum(
            axis=1)
        d_new = postvar(sum2, n, a, b)

        change = max((abs(g_new - g_old) / g_old).max(), (abs(d_new - d_old) / d_old).max())
        g_old = g_new  # .copy()
        d_old = d_new  # .copy()
        count = count + 1
    adjust = (g_new, d_new)
    return adjust


def aprior(gamma_hat):
    m = gamma_hat.mean()
    s2 = gamma_hat.var()
    return (2 * s2 + m ** 2) / s2


def bprior(gamma_hat):
    m = gamma_hat.mean()
    s2 = gamma_hat.var()
    return (m * s2 + m ** 3) / s2


def postmean(g_hat, g_bar, n, d_star, t2):
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)


# Auxiliary functions of ComBat end here ####################
def combat_batch_effect_removal(data, batch_labels, model=None, numerical_covariates=None):
    '''
    This function corrects for batch effect in data.

    Parameters:
    -----------
    data: pandas data frame of numeric values, with a size of (n_features, n_samples)
    batch_labels: pandas series, with a length of n_samples. It should provide the batch labels of samples.
        Its indices are the same as the column names (sample names) in "data".
    model: an object of patsy.design_info.DesignMatrix. It is a design matrix describing the covariate
        information on the samples that could cause batch effects. If not provided, this function
        will attempt to coarsely correct just based on the information provided in "batch".
    numerical_covariates: a list of the names of covariates in "model" that are numerical rather than
        categorical.

    Returns:
    --------
    corrected : pandas data frame of numeric values, with a size of (n_features, n_samples). It is
        the data with batch effects corrected.
    '''

    if isinstance(numerical_covariates, str):
        numerical_covariates = [numerical_covariates]
    if numerical_covariates is None:
        numerical_covariates = []

    if model is not None and isinstance(model, pd.DataFrame):
        model["batch"] = list(batch_labels)
    else:
        model = pd.DataFrame({'batch': batch_labels})

    batch_items = model.groupby("batch").groups.items()
    batch_levels = [k for k, v in batch_items]
    batch_info = [v for k, v in batch_items]
    n_batch = len(batch_info)
    n_batches = np.array([len(v) for v in batch_info])
    n_array = float(sum(n_batches))

    # drop intercept
    drop_cols = [cname for cname, inter in ((model == 1).all()).iteritems() if inter == True]
    drop_idxs = [list(model.columns).index(cdrop) for cdrop in drop_cols]
    model = model[[c for c in model.columns if c not in drop_cols]]
    numerical_covariates = [list(model.columns).index(c) if isinstance(c, str) else c
                            for c in numerical_covariates if c not in drop_cols]

    design = design_mat(model, numerical_covariates, batch_levels)

    sys.stderr.write("Standardizing Data across genes.\n")
    B_hat = np.dot(np.dot(la.inv(np.dot(design.T, design)), design.T), data.T)
    grand_mean = np.dot((n_batches / n_array).T, B_hat[:n_batch, :])
    var_pooled = np.dot(((data - np.dot(design, B_hat).T) ** 2), np.ones((int(n_array), 1)) / int(n_array))

    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, int(n_array))))
    tmp = np.array(design.copy())
    tmp[:, :n_batch] = 0
    stand_mean += np.dot(tmp, B_hat).T

    s_data = ((data - stand_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, int(n_array)))))

    sys.stderr.write("Fitting L/S model and finding priors\n")
    batch_design = design[design.columns[:n_batch]]
    gamma_hat = np.dot(np.dot(la.inv(np.dot(batch_design.T, batch_design)), batch_design.T), s_data.T)

    delta_hat = []

    for i, batch_idxs in enumerate(batch_info):
        delta_hat.append(s_data[batch_idxs].var(axis=1))

    gamma_bar = gamma_hat.mean(axis=1)
    t2 = gamma_hat.var(axis=1)

    a_prior = list(map(aprior, delta_hat))
    b_prior = list(map(bprior, delta_hat))

    sys.stderr.write("Finding parametric adjustments\n")
    gamma_star, delta_star = [], []
    for i, batch_idxs in enumerate(batch_info):
        temp = it_sol(s_data[batch_idxs], gamma_hat[i],
                      delta_hat[i], gamma_bar[i], t2[i], a_prior[i], b_prior[i])

        gamma_star.append(temp[0])
        delta_star.append(temp[1])

    sys.stdout.write("Adjusting data\n")
    bayesdata = s_data
    gamma_star = np.array(gamma_star)
    delta_star = np.array(delta_star)

    for j, batch_idxs in enumerate(batch_info):
        dsq = np.sqrt(delta_star[j, :])
        dsq = dsq.reshape((len(dsq), 1))
        denom = np.dot(dsq, np.ones((1, n_batches[j])))
        numer = np.array(bayesdata[batch_idxs] - np.dot(batch_design.loc[batch_idxs], gamma_star).T)

        bayesdata[batch_idxs] = numer / denom

    vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))
    bayesdata = bayesdata * np.dot(vpsq, np.ones((1, int(n_array)))) + stand_mean

    return bayesdata
