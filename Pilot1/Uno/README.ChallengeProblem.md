# Using Uno for Challenge Problem

## Generate Plan file

From the (master dataframe generation process)[https://github.com/ECP-CANDLE/topN_generator/blob/master/build_master.ipynb], the unique list of cell line and drug will be generated (top21_cell.txt and top21_drug.txt respectively)

```
$ python plangen.py --fs_name cell drug --fs_paths top21_cell.txt top21_drug.txt --fs_parts 4 1 --out_dir . --overwrite
...
plangen_cell694-p4_drug1492-p1.json JSON file written
```

## Node specific dataframe generation

This part is already integrated in the Challenge Problem workflow, but for the testing purpose, you can run this command to generate a dataframe for node 1.1

```
$ python topN_to_uno.py --dataframe_from top21.parquet --plan plangen_cell694-p4_drug1492-p1.json --incremental --cell_feature_selection lincs1000 --node 1.1 --output TopN_1.1_uno.h5
```

## Running Uno with new dataframe

```
# set CANDLE_DATA_DIR to point direcotry containing topN_1.1_uno.h5
$ python uno_baseline_keras2.py --conf uno_auc_model.txt --use_exported_data topN_1.1_uno.h5
```
