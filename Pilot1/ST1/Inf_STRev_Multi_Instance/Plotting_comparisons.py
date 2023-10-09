import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
font = {'family' : 'normal',
    'weight' : 'normal',
    'size'   : 18}

matplotlib.rc('font', **font)
cmap="RdBu_r"

data_PWD = '.'
ranks_ST_T1 = [12, 24, 48, 96]#[i * 4 for i in range (1,5)]
preproc_ST_T1 = []
inf_ST_T1 = []
smi_sec_ST_T1 = []

for r in ranks_ST_T1:
    data =  np.loadtxt(f'{data_PWD}/time_info_ranks{r}.csv', delimiter=",")
    data_preproc = data[:,1]
    data_infer = data[:,2]
    total_time = data[:,3]
    
    preproc_ST_T1.append(np.mean(data_preproc))
    inf_ST_T1.append(np.mean(data_infer))
    smi_sec_ST_T1.append(10000*162/(np.mean(total_time)))


ranks_ST_OLD = [12, 24, 48]
dat_rd_ST_OLD = []
preproc_ST_OLD = []
inf_ST_OLD = []
smi_sec_ST_OLD = []

data_PWD = '/lus/gila/projects/candle_aesp_CNDA/avasan/Inference_ST_All_Recepts/ST_Original'

for r in ranks_ST_OLD:
    data =  np.loadtxt(f'{data_PWD}/time_info_ranks{r}.csv', delimiter=",")
    data_read = data[:,1]
    data_preproc = data[:,2]
    data_infer = data[:,3]
    total_time = data[:,4]
    
    dat_rd_ST_OLD.append(np.mean(data_read))
    preproc_ST_OLD.append(np.mean(data_preproc))
    inf_ST_OLD.append(np.mean(data_infer))
    smi_sec_ST_OLD.append(10000*162/(np.mean(total_time)))

ranks_reg_T3 = [12, 24, 48, 96]
dat_rd_reg_T3 = []
preproc_reg_T3 = []
inf_reg_T3 = []

smi_sec_reg_T3 = []
#smi_sec_reg_T3_mord = []

data_PWD = '/lus/gila/projects/candle_aesp_CNDA/avasan/Reg_GO/Inference_Multi_Instance'
#mordred = np.loadtxt(f'{data_PWD}/time_info_ranks12_mordred.sunspot.csv', delimiter=",")
#print(np.mean(mordred[:,1]))
for r in ranks_reg_T3:
    data = np.loadtxt(f'{data_PWD}/time_info_ranks{r}.csv', delimiter=",")
    data_read = data[:,1]
    #data_preproc = data[:,2]
    data_infer = data[:,2]
    total_time = data[:,3]
    
    dat_rd_reg_T3.append(np.mean(data_read))
    preproc_reg_T3.append(np.mean(data_preproc))
    inf_reg_T3.append(np.mean(data_infer))
    smi_sec_reg_T3.append(10000*160/(np.mean(total_time)))
#    smi_sec_reg_T3_mord.append(10000*160/(np.mean(total_time)+162*np.mean(mordred[:,1])))


#plt.scatter(ranks_ST_T1, preproc_ST_T1, c='red')
plt.semilogx(ranks_ST_T1, preproc_ST_T1, base=2, marker='.', color='red')
plt.semilogx(ranks_ST_OLD, preproc_ST_OLD, base=2, marker='.',  color='blue')
#plt.semilogx(ranks_reg_T3, preproc_reg_T3, base=2, marker='.',  color='green')

plt.xlim(3,16)
plt.xticks([4,8,12,16])
plt.xlabel('Ranks/node')
plt.ylabel('Preprocessing time')
plt.legend()
plt.savefig('Images/Preprocessing_time.png', bbox_inches='tight', dpi=300)
plt.close()

plt.semilogx(ranks_ST_T1, inf_ST_T1, base=2, marker='.', color='red')
plt.semilogx(ranks_ST_OLD, inf_ST_OLD, base=2, marker='.',  color='blue')
#plt.scatter(ranks_ST_T1, inf_ST_T1, c='red')
plt.xlim(3,16)
plt.xticks([4,8,12,16])
plt.xlabel('Ranks/node')
plt.ylabel('Inference time')
plt.legend()
plt.savefig('Images/Inference_time.png', bbox_inches='tight', dpi=300)
plt.close()
#

fig, ax = plt.subplots()

ax.plot(ranks_ST_T1, smi_sec_ST_T1, marker='.', color='red')
ax.plot(ranks_ST_OLD, smi_sec_ST_OLD, marker='.',  color='blue')
ax.plot(ranks_reg_T3, smi_sec_reg_T3, marker='.',  color='green')
#ax.plot(ranks_reg_T3, smi_sec_reg_T3_mord,  marker='.',  color='orange')
ax.set_xscale('log', base=2)
ax.set_yscale('log', base=10)
plt.ylim(85,150000)
plt.xlim(11,100)
plt.xticks([12,24,48,96])
plt.xlabel('Ranks/node')
plt.ylabel('SMILES/second')
plt.legend()
plt.savefig('Images/SMILESpersec.png', bbox_inches='tight', dpi=300)
plt.close()





#print(ranks_ST_T1)
#print(smi_sec_ST_T1)
#plt.scatter(ranks_ST_T1, smi_sec_ST_T1, c='red')
#plt.xlim(3,33)
#plt.xticks([4,8,12,16,32])
##plt.ylim(5000,8000)
#plt.xlabel('Ranks/node')
#plt.ylabel('SMILES/second')
#plt.savefig('Images/SMILESpersec.png', bbox_inches='tight', dpi=300)
#plt.close()




