"""
Author: Shadi Zabad
Date: November 2017
-------------------

"""

from keras.models import load_model
from keras import backend as K
import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from matplotlib import gridspec

model = load_model("./models/13.kmodel")
input_layer = model.get_layer(index=0)
recurrent_layer = model.get_layer(index=1)
output_layer = model.get_layer(index=2)

# ----------------------------------------

seq_func = K.function([input_layer.input],
                      [recurrent_layer.output,
                       recurrent_layer.recurrent_kernel_i,
                       recurrent_layer.recurrent_kernel_o,
                       recurrent_layer.recurrent_kernel_f,
                       recurrent_layer.recurrent_kernel_c,
                       output_layer.output])

# ----------------------------------------

ex_dataset = pd.read_csv('../../data/mauna-loa-atmospheric-co2.csv',
                         header=None)
ex_dataset.columns = ['CO2Concentration', 'Time']
diff_values = ex_dataset['CO2Concentration'].diff()

pred_vals = []

for idx in range(len(ex_dataset['CO2Concentration'])):

    if idx > 0:
        rec_output, w_i, w_o, w_f, w_c, pv = seq_func([np.array([diff_values[idx]]).reshape(-1, 1, 1)])
        pred_vals.append(pred_vals[-1] + pv)

    else:
        pred_vals.append(ex_dataset['CO2Concentration'][0])
        rec_output = np.zeros((1, 50))
        w_i, w_o, w_f, w_c = np.zeros((50, 50)), np.zeros((50, 50)), np.zeros((50, 50)), np.zeros((50, 50))

    plt.figure(1, figsize=(8, 8))

    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 4])

    """plt.subplot(311)
    plt.imshow(w_f, cmap='GnBu', interpolation='nearest', aspect='auto')
    plt.axis('off')
    plt.title("Forget Weights")
    # --------------
    plt.subplot(621)
    plt.imshow(w_o, cmap='GnBu', interpolation='nearest')
    plt.axis('off')
    plt.title("O Weights")
    # --------------
    plt.subplot(621)
    plt.imshow(w_i, cmap='GnBu', interpolation='nearest')
    plt.axis('off')
    plt.title("F Weights")
    # --------------
    plt.subplot(621)
    plt.imshow(w_c, cmap='GnBu', interpolation='nearest')
    plt.axis('off')
    plt.title("C Weights")"""
    # --------------
    plt.subplot(gs[0])
    plt.imshow(rec_output, cmap='GnBu', interpolation='nearest', aspect='auto')
    plt.axis('off')
    plt.title("Output of the Recurrent Layer")
    # --------------
    plt.subplot(gs[1])
    plt.plot(pred_vals, color="blue")
    plt.plot(ex_dataset['CO2Concentration'][:idx + 1], color="red")
    plt.title("Ground Truth vs. Predictions")
    plt.ylim([310, 380])
    plt.xlim([0, len(ex_dataset['CO2Concentration'])])

    plt.savefig("./model_viz/img" + str(idx) + ".png")

    plt.close()
