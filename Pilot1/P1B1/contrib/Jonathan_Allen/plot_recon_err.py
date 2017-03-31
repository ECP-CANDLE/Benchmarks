###
### Simple matplotlib routine to plot reconstruction error (y-axis) with standard deviation over epochs (x-axis) 
### 
###
### example useage for 3 different autoencoder models:
### python3 ../plot_recon_err.py "e_log.20468.out.hp.0,400x300x100 e_log.20468.out.hp.1,500x100 e_log.20468.out.hp.2,1000x500" ex1.png
import sys
import matplotlib.pyplot as plt


flst=sys.argv[1]
output=sys.argv[2]
files=[]
if flst.find(" ") != -1 :
      files=flst.split(" ")
else :
      files.append(flst)

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
lstyle='dotted'
cval=["red","blue","green","purple","orange","black"]
cnt=0
for file_all in files :
   file,desc=file_all.split(',')
   fh=open(file)
   xval,yval,std=[],[],[]
   for line in fh :
      line=line.rstrip()
      vals=line.split(" ")
      xval.append(int(vals[0]))
      yval.append(float(vals[1]))
      std.append(float(vals[2]))

   plt.errorbar(xval,yval,yerr=std,ls=lstyle,color=cval[cnt],label=desc)
   cnt+=1

plt.xlabel('Epoch')
plt.ylabel('Reconstruction Error')
plt.legend()
plt.savefig(output,dpi=100)
plt.show()
