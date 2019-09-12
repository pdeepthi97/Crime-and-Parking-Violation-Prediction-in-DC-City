import matplotlib.pyplot as plt
import pickle
import numpy as np
two_arrays = pickle.load(open('prediction.pkl', 'rb'))
true_op = two_arrays[0]
pred_op = np.round(two_arrays[1])
plt.plot(true_op[120:220])
plt.plot(pred_op[120:220])
plt.legend(['true', 'predicted'])
plt.ylabel('crime count')
plt.xlabel('time interval')
print ()