import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np


loss_data = np.loadtxt("loss.csv", delimiter=',')

draw_all = False
num_to_draw = 100
num_to_draw = num_to_draw if (not draw_all) else len(loss_data)
x_value = np.arange(num_to_draw)

num_iter = loss_data.shape[0]

rpn_loss_cls = loss_data[:num_to_draw, 0]
rpn_loss_box = loss_data[:num_to_draw, 1]
loss_cls = loss_data[:num_to_draw, 2]
loss_box = loss_data[:num_to_draw, 3]
total_loss = loss_data[:num_to_draw, 4]

# draw all type loss
plt.figure(1) 
plt.plot(x_value, rpn_loss_cls, 'lime', label = 'rpn loss cls')
plt.plot(x_value, rpn_loss_box, 'orange', label = 'rpn loss box')
plt.plot(x_value, loss_cls, 'gold', label = 'loss cls')
plt.plot(x_value, loss_box, 'm', label = 'loss box')
plt.plot(x_value, total_loss, 'deepskyblue', label = 'total loss')
plt.legend()

# draw only total loss
plt.figure(2)
plt.plot(x_value, total_loss, 'deepskyblue', label = 'total loss')
plt.legend()
plt.show()


