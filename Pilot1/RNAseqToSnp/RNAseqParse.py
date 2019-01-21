from __future__ import print_function

import os
import sys
import logging
import pandas as pd
import numpy as np
from functools import partial
from multiprocessing import cpu_count, Pool
import mygene

cores = cpu_count()  # Number of CPU cores on your system
partitions = cores  # Define as many partitions as you want


def parallelize(data, func):
    data_split = np.array_split(data, partitions)
    pool = Pool(cores)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data

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
        df = pd.read_hdf(self.cache_path + file, key=key)
        logging.debug("Loaded %s with shape (%i, %i)", file, df.shape[0], df.shape[1])
        return df

    def load_cell_metadata(self, file="combined_cl_metadata"):
        if self.cl_metadata is not None:
            return self.cl_metadata
        if os.path.exists(self.cache_path + file + ".hdf"):
            self.cl_metadata =  self.load_hdf(file + ".hdf")
            return self.cl_metadata

        df = pd.read_table(self.data_path + file)
        df = df.rename(
            {'sample_name': "Sample", 'simplified_tumor_site': 'tumor_site', 'simplified_tumor_type': 'tumor_type'},
            axis=1)
        df = df[['Sample', 'tumor_site', 'tumor_type']]
        self.cache(df, file + ".hdf")
        self.cl_metadata = df
        return df

    def load_oncogenes_(self, file="oncogenes"):
        dic = self.load_ensembl_dict()
        oncogenes = pd.read_table(self.data_path + file, names=['oncogenes'])
        oncogenes.replace(dic)
        return oncogenes

    def load_snp_data(self, file="combo_snp", name_mapping="ensembl2genes"):
        if self.snps is not None:
            return self.snps
        if os.path.exists(self.cache_path + file + ".hdf"):
            print("loading from cache...")
            self.snps = self.load_hdf(file + ".hdf")
            return self.snps
        print("Reading snp data....could take awhile.")
        df = pd.read_table(self.data_path + file).astype(np.int, errors='ignore')
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
        logging.debug("Logging reduce snps by ensbl.")
        df = snps
        df = df.set_index('Sample:Sample')
        df = df.apply(pd.to_numeric, axis=1)
        df = df.transpose()
        df = df.apply(lambda x : x.astype(int), axis=1)
        df = df.reset_index()
        df['index'] = [this.split(":")[0] for this in df['index']]
        logging.debug("Reworked snp numerics.")

        gb = df.groupby("index", sort=False).sum().transpose()
        logging.debug("Loaded reduced snps.")
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
        if os.path.exists(self.cache_path + file + ".hdf"):
            print("loading from cache...")
            self.rnaseq = self.load_hdf( file + ".hdf")
            return self.rnaseq

        df = pd.read_table(self.data_path + file)
        if name_mapping is not None:
            ensembl_dict = self.load_ensembl_dict(name_mapping)
            df = df.rename(ensembl_dict, axis=1)

        self.cache(df, file)
        self.rnaseq = df
        return df

    def load_aligned_snps_rnaseq(self, file_rnaseq=None, file_snp=None, cached_file=("combined_aligned_rnaseq_snp.hdf", "aligned_snp", "aligned_rnaseq"), name_mapping="ensembl2genes", use_reduced=False, align_by=['name']):
        logging.info("Loading aligned snp rna seq.")
        if len(align_by) == 0:
            align_by = ['None']
        if os.path.exists(self.cache_path + "_".join(align_by) + "_" + cached_file[0]):
            snps =  self.load_hdf("_".join(align_by) + "_" + cached_file[0], key=cached_file[1])
            rnaseq = self.load_hdf("_".join(align_by) + "_" + cached_file[0], key=cached_file[2])
            if "genemania" in align_by:
                adj = pd.read_hdf(self.cache_path + "_".join(align_by) + "_" + cached_file[0], key="adj")
                self.adj = adj
            return snps, rnaseq

        logging.debug("Loading data from scratch.")
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
        logging.debug("Loaded all files. Aligning by %s", align_by)
        if  'pos' in align_by:
            print("Position aligned by not support yet")
            snp_feats = set(snps.columns)
            rna_feats = set(rnaseq.columns)
            common_feats = snp_feats.intersection(rna_feats)
            snps = snps.drop(snp_feats.difference(common_feats), axis=1)
            rnaseq = rnaseq.drop(rna_feats.difference(common_feats), axis=1)
            print("intersectyion has %i" % len(common_feats))
            gene_names = common_feats
            mg = mygene.MyGeneInfo()
            qu = mg.querymany(gene_names, fields='genomic_pos', scopes='ensembl.gene', as_dataframe=True, df_index=True)
            qu = qu[['genomic_pos.chr', 'genomic_pos.start']].dropna(axis=0)
            qu['genomic_pos.chr'] = pd.to_numeric(qu['genomic_pos.chr'], coerce)
            qu['genomic_pos.start'] = pd.to_numeric(qu['genomic_pos.start'], coerce)
            qu = qu.dropna(axis=0)
            ordered_rna_seq = qu.sort_values(["genomic_pos.chr", "genomic_pos.start"]).index.to_series()
            rnaseq = rnaseq[ordered_rna_seq]
            snps = snps[ordered_rna_seq]
        elif "genemania" in align_by:
            adj = pd.read_hdf(self.data_path + "GeneMania_adj.hdf", key="adj")
            adj_feats = set(adj.columns)
            rna_feats = set(rnaseq.columns)
            snp_feats = set(snps.columns)
            common_feats = set(adj.columns).intersection(rna_feats).intersection(snp_feats)
            adj = adj.drop(adj_feats.difference(common_feats), axis=1).drop(adj_feats.difference(common_feats), axis=0)
            snps = snps.drop(snp_feats.difference(common_feats), axis=1)
            rnaseq = rnaseq.drop(rna_feats.difference(common_feats), axis=1)
            print(adj.shape, snps.shape, rnaseq.shape)
            snps = snps.sort_index(axis=1)
            rnaseq = rnaseq.sort_index(axis=1)
            adj = adj.sort_index(axis=0).sort_index(axis=1)
            self.adj = adj
            print("Caching adj....")
            adj.to_hdf(self.cache_path + "_".join(align_by) + "_" + cached_file[0], key='adj')
            print("Success!")

        else:
            print("Please select name, pos, or genemania")
            exit(1)
        logging.debug("Aligned files. Final shapes: " + str(snps.shape) + str(rnaseq.shape))

        snps.to_hdf(self.cache_path + "_".join(align_by) + "_" + cached_file[0], key=cached_file[1])
        rnaseq.to_hdf(self.cache_path + "_".join(align_by) + "_" + cached_file[0], key=cached_file[2])
        logging.debug("Cached files.")
        return snps, rnaseq

    def get_adj(self):
        return self.adj

    def load_table(self, file):
        df = pd.read_table(self.data_path + file)
        logging.debug("Loaded %s with shape (%i, %i)", file, df.shape[0], df.shape[1])
        return df

    def get_training_data(self):
        self.load_aligned_snps_rnaseq()