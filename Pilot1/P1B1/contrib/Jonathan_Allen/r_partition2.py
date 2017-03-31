########################################################
##
## input is M x N matrix (--features) with M=examples, N=features
## a corresonding set of labels for each example is specified in a separate file
## an optional hold out set can be specified, in the form of a two column text file
## the first column specifies the label to be held out and second column
## specifies the fraction of the examples for the label to be held out
## examples:
## 1 1.0  -> All examples labeled 1 will be put in the held out set
## 2 0.5  -> 1/2 of the examples labeled 2 will be placed in the hod out set
## after puting examples in the held out set, the remaining data 
## will be partitioned into train and test sets.
##
## example useage:
## python2.7 /p/lscratchf/allen99/lbexp/r_partition2.py --features ../X  --labels ../y --partitions 5 --outname gdc_rh5 --holdout check.txt
##
########################################################
import sys
import numpy as np
import argparse

def main () :
   parser = argparse.ArgumentParser(description='Basic autoencoder')
   parser.add_argument('--features',required=True, help= "feature file (X)")
   parser.add_argument('--labels',required=True, help="label file (Y)")
   parser.add_argument('--partitions',required=True, help="number of partitions")
   parser.add_argument('--outname',required=True, help="prefix for output filename")
   parser.add_argument('--holdout',required=False, help="file with list of labels to hold out and fractio of the labels to hold out")

   args=parser.parse_args()

   ## initialize key parameters and input variables
   feat_file=args.features
   label_file=args.labels
   ofname=args.outname
   pnum=int(args.partitions)
   feat_mat =np.loadtxt(feat_file)
   lab_mat =np.loadtxt(label_file)

   if args.holdout != None : 
      print "orighap",feat_mat.shape,lab_mat.shape
      feat_mat,lab_mat=makeHoldout(args.holdout,feat_mat,lab_mat)
      print "newshap",feat_mat.shape,lab_mat.shape

   row,col = feat_mat.shape
   row1 = lab_mat.shape[0]

   iarr = np.arange( row )

   np.random.shuffle( iarr )

   binsize = row / pnum

   train_feat,test_feat,train_lab,test_lab = [],[],[],[]

   for part in range(pnum) :
      top_test = (part*binsize) + binsize
      bot_test = part*binsize

      first_test=True
      first_train=True
      for iter_raw in range(row) :
         iter = iarr[iter_raw]
         if iter_raw >= bot_test and iter_raw < top_test :
            if first_test : 
               test_feat.append( np.array(feat_mat[iter]) )
               test_lab.append( np.array(lab_mat[iter]) )
               first_test=False
            else : 
               #print "debug1",lab_mat[iter],iter,part
               test_feat[part] = np.vstack((test_feat[part],feat_mat[iter]))
               test_lab[part] = np.vstack((test_lab[part],lab_mat[iter]))
         else :
            if first_train :
               train_feat.append( np.array(feat_mat[iter]) )
               train_lab.append( np.array(lab_mat[iter]) )
               first_train=False
            else :
               train_feat[part] = np.vstack((train_feat[part],feat_mat[iter]))
               train_lab[part] = np.vstack((train_lab[part],lab_mat[iter]))

   print "train_shape",train_feat[0].shape,train_lab[0].shape
   print "test_shape",test_feat[0].shape,test_lab[0].shape

   for part in range(pnum) :
      ltest = ofname + ".test.lab." + str(part)
      ltrain = ofname + ".train.lab." + str(part)
      ftest = ofname + ".test.fea." + str(part)
      ftrain = ofname + ".train.fea." + str(part)

      np.savetxt(ltest,test_lab[part],fmt="%d",delimiter="\t") 
      np.savetxt(ltrain,train_lab[part],fmt="%d",delimiter="\t") 
      np.savetxt(ftest,test_feat[part],delimiter="\t") 
      np.savetxt(ftrain,train_feat[part],delimiter="\t") 


def makeHoldout(fname,feat,labl) :
   fh=open(fname)
   save={}
   for line in fh :
      line=line.rstrip()
      vals=line.split(' ')
      lval=float(vals[0])
      frac=float(vals[1])
      save.setdefault(lval,frac)
   print "huh",save.keys()
   rows=feat.shape[0]
   ctrack = {}
   lab_tot={}
   for row_i in range(rows) : 
      lab = labl[row_i]
      ctrack.setdefault(lab,0)
      if lab_tot.has_key(lab) :
         lab_tot[lab] += 1
      else :
         lab_tot.setdefault(lab,1)

   isFirstHold=True
   isFirstTrain=True
   hold_feat,hold_labl=None,None
   new_feat,new_labl=None,None
   for row_i in range(rows) : 
      lab = labl[row_i]
      if save.has_key(lab) :
         pcnt = save[lab]
         cpcnt = float(ctrack[lab]) / float(lab_tot[lab])
         if cpcnt < pcnt :
            if isFirstHold :
               hold_feat=np.array(feat[row_i])
               hold_labl=np.array(labl[row_i])
               isFirstHold = False
            else :
               hold_feat=np.vstack((hold_feat,feat[row_i]))
               hold_labl=np.vstack((hold_labl,labl[row_i]))

            ctrack[lab] += 1
         else :
            if isFirstTrain :
               new_feat=np.array(feat[row_i])
               new_labl=np.array(labl[row_i])
               isFirstTrain = False
            else :
               new_feat=np.vstack((new_feat,feat[row_i]))
               new_labl=np.vstack((new_labl,labl[row_i]))

   print "hold_dim",hold_feat.shape,hold_labl.shape
   print "new_dim",new_feat.shape,new_labl.shape
   np.savetxt(fname+".feat",hold_feat,delimiter="\t") 
   np.savetxt(fname+".label",hold_labl,delimiter="\t") 

   return new_feat,new_labl
if __name__ == '__main__' :
   main()
