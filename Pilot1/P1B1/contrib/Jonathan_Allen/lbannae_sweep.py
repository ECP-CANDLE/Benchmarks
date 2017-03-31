###
###
###  Python script to launch a simple sweep of hyper paramters
###
###
import os,sys
from subprocess import call

nodes=int(sys.argv[1])   ## number of compute nodes to use
partitions=int(sys.argv[2])  ## expected number of cross validation partitions
## name of the train/testing cross validation files (format is: "filebn".train.fea.X or "filebn".test.fea.X)
## where 
filebn=sys.argv[3] 
ddir=sys.argv[4]

## original
ddir="/p/lscratchf/allen99/anlftp/public/datasets/GDC/data_frames/BySite"

aecmd="/p/lscratchf/allen99/lbexp/run_lbann_ae.sh"

##source code command line references
##LearnRateMethod = Input("--learning-rate-method", "1 - Adagrad, 2 - RMSprop, 3 - Adam", LearnRateMethod);
##ActivationType = static_cast<activation_type>(Input("--activation-type", "1 - Sigmoid, 2 - Tanh, 3 - reLU, 4 - id", static_cast<int>(ActivationType)));

## -f -> data location
## -e -> epoch
## -b -> mini-match
## -a -> activiation type 
## -r -> learning rate
## -j -> learning rate decay
## -k -> fraction of training data to use for training
## -g -> dropout rate
## -q -> learning rate method
## -n -> network topology : specify number of nodes in each hidden layer
## original parameters
param_lst=[]
params="-e 100 -g 0.1 -b 50 -a 3 -r 0.0001 -j 0.5 -g -1 -q 1 -n 100 -k 0.75 -n 400,300,100" + " -f "+ddir
param_lst.append(params)
params="-e 100 -g 0.1 -b 50 -a 3 -r 0.0001 -j 0.5 -g -1 -q 1 -n 100 -k 0.75 -n 500,100" + " -f "+ddir
param_lst.append(params)
params="-e 100 -g 0.1 -b 50 -a 3 -r 0.0001 -j 0.5 -g -1 -q 1 -n 100 -k 0.75 -n 1000,500" + " -f "+ddir
param_lst.append(params)
params="-e 100 -g 0.1 -b 50 -a 3 -r 0.0001 -j 0.5 -g -1 -q 1 -n 100 -k 0.75 -n 1000,500,250,100" + " -f "+ddir
param_lst.append(params)
params="-e 100 -g 0.1 -b 50 -a 3 -r 0.0001 -j 0.5 -g -1 -q 1 -n 100 -k 0.75 -n 100" + " -f "+ddir
param_lst.append(params)

for hpi in range(len(param_lst)) :
   for parti in range(partitions) :
      tr_file=filebn+".train.fea."+str(parti)
      ts_file=filebn+".test.fea."+str(parti)
      out_name="ae_log."+str(os.getpid())+".out."+str(parti) + ".hp."+str(hpi)
      run_cmd="sbatch -N"+str(nodes) + " -t 1440 --clear-ssd --msr-safe --output="+out_name+" "+aecmd+" -x "+tr_file+" -y " +ts_file+" "+param_lst[hpi]
      print run_cmd
      call(run_cmd,shell=True)

#sbatch -N$nodes  -t 1440 --clear-ssd --msr-safe --output="slurm-lbann-nci_ae_tst-%j.v2.out" $bdir/run_lbann_ae.sh -x gdc_rand5.train.fea.0 -y gdc_rand5.test.fea.0 $params
