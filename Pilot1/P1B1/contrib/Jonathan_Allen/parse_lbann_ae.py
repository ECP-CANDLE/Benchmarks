#######################################################################################
## Parse reconstruction error from LBANN output
##
## program is designed to take X LBANN output files from X-fold cross validation
##
## and compute the average reconstruction error for each Epoch with standard deviation
##
#######################################################################################
import sys,math

def getParams(fn) :
   epochs,nlayers=-1,-1
   fh = open(fn)
   for line in fh :
      line=line.rstrip()
      kw="--network "
      kw_len = len(kw)
      sidx = line.find(kw)
      if sidx != -1 :
         eidx = line.find(" ",sidx+kw_len)
         nstr=line[sidx+kw_len:eidx]
         vals=nstr.split(',')
         nlayers=len(vals)
      kw="--num-epochs "
      kw_len = len(kw)
      sidx = line.find(kw)
      if sidx != -1 :
         eidx = line.find(" ",sidx+kw_len)
         nstr=line[sidx+kw_len:eidx]
         epochs=int(nstr)
         print "chk",sidx,eidx,nstr,epochs
         break
   
   assert nlayers != -1 and epochs != -1
   return nlayers,epochs

def getCost(fn,num_layers) :
   cost_val={}
   fh = open(fn)
   epoch_val=-1
   for line in fh :
      line=line.rstrip()
      kw="Phase [" + str(num_layers-1)+"] Epoch ["
      kw_len = len(kw)
      sidx = line.find(kw)
      if sidx != -1 :
         eidx = line.find("]",sidx+kw_len)
         nstr=line[sidx+kw_len:eidx]
         epoch_val=int(nstr)
         #print "save e",epoch_val
      kw="Testing model 0 average reconstruction cost: "
      kw_len = len(kw)
      sidx = line.find(kw)
      if sidx != -1 and epoch_val != -1 :
         nstr=line[sidx+kw_len:]
         recon_val=float(nstr)
         assert epoch_val != -1 
         #print "save",epoch_val,recon_val
         cost_val.setdefault(epoch_val,recon_val)

   assert cost_val != {}
   return cost_val


flst=[]
for fn in sys.stdin :
   fn=fn.rstrip()
   flst.append(fn)


cost_save=[]
max_epoch=-1
for fn in flst :
   num_layers,epochs=getParams(fn)
   cost = getCost(fn,num_layers)
   cost_save.append(cost)
   if epochs > max_epoch :
      max_epoch = epochs
   
oname=sys.argv[1]
oh=open(oname,"w")
for epi in range(max_epoch) :
   cnt,sval=0,0
   for val in cost_save :
      if val.has_key(epi) :
         sval += val[epi]
         cnt += 1
   avg = sval / cnt
   sval = 0
   for val in cost_save :
      if val.has_key(epi) :
         sval += (val[epi]-avg)*(val[epi]-avg)
   stdev=0
   if cnt > 1 :
      stdev = math.sqrt(sval / (cnt-1))    

   res = str(epi)+" "+str(avg)+" "+str(stdev)
   oh.write(res + "\n")
