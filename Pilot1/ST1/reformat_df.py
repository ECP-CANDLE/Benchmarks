import sys
import os
import sklearn.model_selection
import pandas as pd

data_path = sys.argv[1]
base = os.path.basename(data_path)

# classification is in cls column, regression is in reg
outfile = '{}.xform-smiles.csv.reg'.format(base)

df = pd.read_parquet(data_path)

df2=pd.DataFrame()

# cls or reg for ml.RDRP_7BV2_A_3_F.Orderable_zinc_db_enaHLL.sorted.4col.dd.parquet
df2[['type','smiles']]=df[['reg','SMILES']]
x_train, x_val = sklearn.model_selection.train_test_split(df2, test_size=.2, train_size=.8)

x_train.to_csv('{}.train'.format(outfile), index=None)
x_val.to_csv('{}.val'.format(outfile), index=None)
df2.to_csv('{}'.format(outfile),index=None)

