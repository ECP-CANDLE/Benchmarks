import os
import sys
import logging
import pandas as pd
import numpy as np
from functools import partial

class DataLoader:
    def __init__(self, data_path, args):
        self.data_path = data_path
        self.cache_path = args.cache
        self.args = args
        self.cl_metadata = None
        self.rnaseq = None
        self.snps = None
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

    def get_key(self, file):
        return file.split(".")[0].split("_")[-1]

    def cache(self, data, file):
        fn = self.cache_path + file
        data.to_hdf(fn, key=self.get_key(fn))
        logging.debug("Cached data to %s", fn)

    def load_hdf(self, file, key=None):
        if key is None:
            key = self.get_key(file)
        df = pd.read_hdf(self.data_path + file, key=key)
        logging.debug("Loaded %s with shape (%i, %i)", file, df.shape[0], df.shape[1])
        return df

    def load_cell_metadata(self, file="combined_cl_metadata"):
        if self.cl_metadata is not None:
            return self.cl_metadata
        if os.path.exists(self.data_path + file + ".hdf"):
            self.cl_metadata =  self.load_hdf(self.data_path + file + ".hdf")
            return self.cl_metadata

        df = pd.read_table(self.data_path + file)
        df = df.rename(
            {'sample_name': "Sample", 'simplified_tumor_site': 'tumor_site', 'simplified_tumor_type': 'tumor_type'},
            axis=1)
        df = df[['Sample', 'tumor_site', 'tumor_type']]
        self.cache(df, file + ".hdf")
        self.cl_metadata = df
        return df

    def load_snp_data(self, file="combo_snp", name_mapping="ensembl2genes"):
        if self.snps is not None:
            return self.snps
        if os.path.exists(self.data_path + file + ".hdf"):
            self.snps = self.load_hdf(self.data_path + file + ".hdf")
            return self.snps

        df = pd.read_table(self.data_path + file).astype(np.int, errors='ignore')
        to_numeric_ignore = partial(pd.to_numeric, errors="ignore")
        df = df.apply(to_numeric_ignore, axis=1)
        if name_mapping is not None:
            ensembl_dict = self.load_ensembl_dict(name_mapping)
            snp_names = df.columns.to_series()
            snp_names_cut_off = snp_names.apply(lambda x: x.split(":")[0])
            snp_colon_delim = snp_names.apply(lambda x : x.split(":")[-1])
            snp_to_ensembl = snp_names_cut_off.replace(ensembl_dict)
            snp_to_ensembl = snp_to_ensembl + ":" + snp_colon_delim
            snp_to_ensembl = snp_to_ensembl.to_dict()
            df = df.rename(snp_to_ensembl, axis=1)
        self.cache(df, file + ".hdf")
        return df

    #
    # reduces snps by a given function along ensembl id
    #
    def reduce_snps_by_ensembl(self, snps, reduce_func = np.sum):
        df = snps.transpose()
        df = df.reset_index()
        df['index'] = [this.split(":")[0] for this in df['index']]
        samples = df.iloc[0]
        df = df.drop(0, axis=0)
        did = True

        for i in df.columns.to_series():
            if did and str(i) == "index":
                did = False
                df[i] = df[i].astype(str)
            else:
                df[i] = pd.to_numeric(df[i])

        gb = df.groupby("index", sort=False).sum().transpose()
        gb['Sample'] = samples
        return gb


    def load_ensembl_dict(self, file="ensembl2genes"):
        ensembl2gene = pd.read_table(self.data_path + file, names=["ensembl", 'gene', 'type'])
        ensembl_dict = ensembl2gene.set_index("gene")['ensembl'].to_dict()
        return ensembl_dict

    #
    # Loads RNASeq from file and maps names to ensembl id
    #
    def load_rnaseq_data(self, file="combined_rnaseq_data", name_mapping="ensembl2genes"):
        if self.rnaseq is not None:
            return self.rnaseq
        if os.path.exists(self.data_path + file + ".hdf"):
            self.rnaseq = self.load_hdf(self.data_path + file + ".hdf")
            return self.rnaseq

        df = pd.read_table(self.data_path + file)
        if name_mapping is not None:
            ensembl_dict = self.load_ensembl_dict(name_mapping)
            df = df.rename(ensembl_dict, axis=1)

        self.cache(df, file)
        self.rnaseq = df
        return df

    def load_aligned_snps_rnaseq(self, file_rnaseq=None, file_snp=None, cached_file=("combined_aligned_rnaseq_snp.hdf", "aligned_snp", "aligned_rnaseq"), name_mapping="ensembl2genes", use_reduced=False, align_by='name'):

        if os.path.exists(self.data_path + cached_file[0]):
            snps =  self.load_hdf(self.data_path + cached_file[0], key=cached_file[1])
            rnaseq = self.load_hdf(self.data_path + cached_file[0], key=cached_file[2])
            return snps, rnaseq

        if self.args.pooled_snps is not None:
            snps = self.load_hdf(self.args.pooled_snps)
        else:
            snps = self.load_snp_data()
            if use_reduced:
                snps = self.reduce_snps_by_ensembl(snps)
        rnaseq = self.load_rnaseq_data()
        ensembl_dict = self.load_ensembl_dict(name_mapping)

        #now I have snps aligned to ensembl id's..... I want to align them and get the stride based on position on the chromosome
        # aligned by chromosome position here.
        if align_by == 'pos':
            print "Position aligned by not support yet"
            exit(1)
        elif align_by == 'name':
            snp_feats = set(snps.columns.to_series())
            rna_feats = set(rnaseq.columns.to_series())
            intersect = snp_feats.intersection(rna_feats)
        elif align_by == 'genemania':
            print "Genemania not supported yet."
            exit(1)
        else:
            print "Please select name, pos, or genemania"
            exit(1)

        snps.to_hdf(self.cache_path + cached_file[0], key=cached_file[1])
        rnaseq.to_hdf(self.cache_path + cached_file[0], key=cached_file[2])

        return snps, rnaseq



    def load_table(self, file):
        df = pd.read_table(self.data_path + file)
        logging.debug("Loaded %s with shape (%i, %i)", file, df.shape[0], df.shape[1])
        return df

    def get_training_data(self):
        self.load_aligned_snps_rnaseq()