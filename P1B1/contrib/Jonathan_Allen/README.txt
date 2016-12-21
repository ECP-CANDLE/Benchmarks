## Create 5-fold cross validation partions 
## to support persistant/stable held out test data


### change to local data location of GDC
cd /p/lscratchf/allen99/anlftp/public/datasets/GDC/data_frames/BySite/partitions

## full run
## create and enter a temporary working subdirectory
## generate partitions, output will be: 
## gdc_rand5.train.fea.X, gdc_rand5.test.fea.X, gdc_rand5.train.lab.X, gdc_rand5.test.lab.X
## for partition X, "fea" files store the data matrix, and "lab" retains labels 
python2.7 /p/lscratchf/allen99/lbexp/r_partition2.py --features ../X  --labels ../y --partitions 5 --outname gdc_rand5

## 
## conduct a simple model topology parameter sweep
##
## change to working directory
## run parameter sweep
## output is reconstruction error and stored in lot files of the form: ae_log.Y.out.\*.hp.X
## where Y is a unique process ID and X is the cross-validation partition
python lbannae_sweep.py 16 5 gdc_rand5 /p/lscratchf/allen99/anlftp/public/datasets/GDC/data_frames/BySite >& run_log.txt

## parses LBANN output to report reconstruction error output to generate summary report
## generate average reconstruction error per epoch, with standard deviation
find . -name ae_log.20468.out.\*.hp.4 | python parse_lbann_ae.py e_log.20468.out.hp.4
find . -name ae_log.20468.out.\*.hp.3 | python parse_lbann_ae.py e_log.20468.out.hp.3
find . -name ae_log.20468.out.\*.hp.2 | python parse_lbann_ae.py e_log.20468.out.hp.2
find . -name ae_log.20468.out.\*.hp.0 | python parse_lbann_ae.py e_log.20468.out.hp.0
find . -name ae_log.20468.out.\*.hp.1 | python parse_lbann_ae.py e_log.20468.out.hp.1

## Merge files to do a direct compare and plot with matplotlib
## plots reconstruction error (y-axis) over epoch (x-axis)
python3 plot_recon_err.py "e_log.20468.out.hp.0,400x300x100 e_log.20468.out.hp.1,500x100 e_log.20468.out.hp.2,1000x500 e_log.20468.out.hp.3,1000x500x250x100" ex3.png
