import pickle
import matplotlib.pyplot as plt
import numpy as np

l = pickle.load(open('losses_pv.pkl', 'rb'))
plt.plot(l[0][1:])
plt.plot(l[1][2:])

seqs = pickle.load(open('seqs_pv_1.pkl','rb'))
print(seqs)
parking = pickle.load(open(r'C:\Users\deept\PycharmProjects\UC_project\parking_grid_sequence.pkl', 'rb'))
print(len(parking))
print(len(seqs))
plt.subplots_adjust(wspace=0.4, hspace=0.6)
mse = np.zeros((5, 5))
dse = np.zeros((5, 5))
c = 550
for i in range(5):
    for j in range(5):
        c += 1
        if seqs[i][j]:
            orig_seq = np.array([c[i, j] for c in parking][int(0.9*1940):])
            pred_seq = np.round(seqs[i][j])
            mse[i, j] = np.mean((orig_seq - pred_seq) ** 2)
            dse[i, j] = np.std((orig_seq - pred_seq) ** 2)
            plt.subplot(c)
            plt.plot(orig_seq)
            plt.plot(pred_seq)
plt.show()
