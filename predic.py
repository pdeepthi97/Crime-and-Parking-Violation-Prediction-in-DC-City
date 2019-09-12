import matplotlib.pyplot as plt
import pickle

two_arrays = pickle.load(open('prediction.pkl', 'rb'))
true_op = two_arrays[0]
pred_op = two_arrays[1]
plt.plot(true_op)
plt.plot(pred_op)
plt.legend(['actual sequence', 'predicted sequence'])
plt.ylabel('parking_violation count')
plt.xlabel('time interval')
plt.show()