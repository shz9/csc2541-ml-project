import numpy as np
import scipy.signal as sp
from pyESN.pyESN import ESN
from matplotlib import pyplot as plt
import csv

# lake erie
# 70,5,45
with open('./data/monthly-lake-erie-levels-1921-19.csv', 'rt') as f:
    reader = csv.reader(f)
    lake = list(reader)

y = []
for i in range(1, len(lake)):
    y.append(float(lake[i][1]))

y = np.array(y)
trainlen = 400
future = 200

# solar
# 120, 5, 4
with open('./data/solar.csv', 'rt') as f:
    reader = csv.reader(f)
    solar = list(reader)

y = []
for i in range(1, len(solar)):
    y.append(float(solar[i][2]))

y = np.array(y)
trainlen = 300
future = 102

# co2
# 160, 32, 93
# detrend 80, 18, 42
from sklearn.datasets import fetch_mldata

data = fetch_mldata('mauna-loa-atmospheric-co2').data
y = data[:, 0]

Y = []
for i in range(len(y)):
    Y.append(y[i])

y = np.array(Y)
y = sp.detrend(y)
trainlen = 300
future = 168

# airline
# 150, 15, 84
# detrend 30, 46, 12
with open('./data/airline.csv', 'rt') as f:
    reader = csv.reader(f)
    airline = list(reader)

y = []
for i in range(1, len(airline)):
    y.append(float(airline[i][2]))

y = np.array(y)
y = sp.detrend(y)
trainlen = 72
future = 72

""" Grid Search
best_par = [0, 0, 0]
min_mse = float("inf")
for res in range(10, 60, 10):
    for rad in range(5, 50):
        for rs in range(1, 100):
            esn = ESN(n_inputs = 1,
                      n_outputs = 1,
                      n_reservoir = res,
                      spectral_radius = rad*0.1,
                      random_state= rs)

            pred_training = esn.fit(np.ones(trainlen),y[:trainlen])
            prediction = esn.predict(np.ones(future))
            mse = np.sqrt(np.mean((prediction.flatten() - y[trainlen:trainlen+future])**2))
            if mse < min_mse:
                best_par = [res, rad, rs]
                min_mse = mse
"""
# run ESN over different spectral radii with the best n_reservoir & random_state for the data
rads = 10
pred = np.zeros(shape=(rads, future))
for i in range(1, rads+1):
    esn = ESN(n_inputs = 1,
              n_outputs = 1,
              n_reservoir = 120,
              spectral_radius = i*0.1,
              random_state = 4)

    pred_training = esn.fit(np.ones(trainlen),y[:trainlen])
    prediction = esn.predict(np.ones(future))
    pred_list = [p[0] for p in prediction]
    pred[i-1] = pred_list
# get mean & std over predictions
stats = list(map(lambda i: (np.mean(pred[:, i]), np.std(pred[:, i])), range(future)))
mean = np.array([[tup[0]] for tup in stats])
std = np.array([[tup[1]] for tup in stats])
lb = [mean[i][0]-2*std[i][0] for i in range(future)]
ub = [mean[i][0]+2*std[i][0] for i in range(future)]
# plot time-series
plt.figure(figsize=(11,5))
plt.plot(range(0,trainlen+future),y[0:trainlen+future],'k',label="Target")
plt.plot(range(trainlen,trainlen+future), mean,'r', label="ESN 70-45")
plt.fill_between(range(trainlen,trainlen+future), lb, ub, facecolor='red', alpha=0.3)
lo,hi = plt.ylim()
plt.plot([trainlen,trainlen],[lo+np.spacing(1),hi-np.spacing(1)],'k:')
plt.legend(loc=(0.61,1),fontsize='x-small')
plt.show()

"""
esn = ESN(n_inputs = 1,
          n_outputs = 1,
          n_reservoir = 160,
          spectral_radius = 3.2,
          random_state = 93)

pred_training = esn.fit(np.ones(trainlen),y[:trainlen])
prediction = esn.predict(np.ones(future))
mse = np.sqrt(np.mean((prediction.flatten() - y[trainlen:trainlen+future])**2))

print("test error: \n"+str(np.sqrt(np.mean((prediction.flatten() - y[trainlen:trainlen+future])**2))))

plt.figure(figsize=(11,5))
plt.plot(range(0,trainlen+future),y[0:trainlen+future],'k',label="target system")
plt.plot(range(trainlen,trainlen+future),prediction,'r', label="free running ESN")
lo,hi = plt.ylim()
plt.plot([trainlen,trainlen],[lo+np.spacing(1),hi-np.spacing(1)],'k:')
plt.legend(loc=(0.61,1),fontsize='x-small')
plt.show()
"""