import pickle
import matplotlib.pyplot as plt
import numpy as np

l = pickle.load(open('losses.pkl','rb'))
plt.plot(l[0])
plt.plot(l[1])
plt.xlabel('epoch')
plt.legend(['training loss','testing loss'])

seqs = pickle.load(open('seqs1.pkl','rb'))
crime = pickle.load(open('../../data/crime_mat_sequence.pkl','rb'))

plt.figure()
plt.subplots_adjust(wspace=0.4, hspace=0.6)
mse = np.zeros((5,5))
dse = np.zeros((5,5))
c = 0
for i in range(5):
    for j in range(5):
        c += 1
        if seqs[i][j]:
            orig_seq = np.array([c[i,j] for c in crime][int(0.9*4384):][-20:])
            pred_seq = np.round(seqs[i][j][-20:])
            mse[i,j] = np.mean((orig_seq-pred_seq)**2)
            dse[i,j] = np.std((orig_seq-pred_seq)**2)
            plt.subplot(5,5,c)
            plt.plot(orig_seq)
            plt.plot(pred_seq)
            ax = plt.gca()
            ax.tick_params(axis = 'both', which = 'major', labelsize = 6)
            ax.tick_params(axis = 'both', which = 'minor', labelsize = 6)
        
maxx = np.zeros((5,5))
for i in range(5):
    for j in range(5):
        maxx[i,j] = np.mean([c[i,j] for c in crime])
s = ''
i=0
for i in range(5):
    for  j in range(4):
        s+='$'+str(round(mse[i][j],3))+'\pm '+str(round(dse[i][j],3))+'$'+'&'
    s+='$'+str(round(mse[i][-1],3))+'\pm '+str(round(dse[i][-1],3))+'$'
    s+='\\\\\n\hline\n'
print (s)