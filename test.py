""" Basic example showing samples from the dataset.

    Chose the set you want to see and run the script:
    > python view_samples.py

    Close the figure to see the next sample.
"""
from __future__ import print_function, unicode_literals

import pickle
import os
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# chose between training and evaluation set
# set = 'training'
set = 'evaluation'

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))

image = plt.imread('../../00000.png')
prediction = np.array([[-0.00488, -0.05724, 0.7018], [-0.01406, 0.03035, 0.5829], [-0.02056, 0.01356, 0.6067],
                       [-0.02833, - 0.01342, 0.6344], [-0.02037, - 0.0345, 0.6679], [0.01981, 0.0278, 0.5292],
                       [0.01434, 0.01526, 0.5475], [0.007658, - 0.003428, 0.5729], [-0.001645, - 0.02029, 0.6021],
                       [0.02355, 0.06246, 0.5643], [0.02565, 0.03507, 0.5711], [0.02679, 0.008805, 0.5783],
                       [0.02054, - 0.01729, 0.6071], [0.0355, 0.05862, 0.5791], [0.03781, 0.03081, 0.5893],
                       [0.03916, 0.007767, 0.5961], [0.03463, - 0.01549, 0.6197], [0.0599, 0.04777, 0.5969],
                       [0.06011, 0.03241, 0.6036], [0.05921, 0.01552, 0.6101], [0.04836, - 0.01112, 0.639, ],
                       [-0.4399, -0.01899, 0.7171], [-0.3519, 0.047, 0.6143], [-0.3639, 0.03898, 0.6388],
                       [-0.3827, 0.0153, 0.666, ], [-0.4046, -0.004739, 0.6943], [-0.3583, 0.07382, 0.572, ],
                       [-0.3676, 0.05699, 0.5837], [-0.3823, 0.03344, 0.6008], [-0.3934, 0.01277, 0.627, ],
                       [-0.3869, 0.0789, 0.5519], [-0.3948, 0.06257, 0.573], [-0.4044, 0.04395, 0.5911],
                       [-0.4144, 0.02466, 0.6247], [-0.4096, 0.08036, 0.5609], [-0.4172, 0.06595, 0.5847],
                       [-0.4241, 0.0505, 0.604], [-0.4317, 0.03282, 0.6324], [-0.4322, 0.0789, 0.5857],
                       [-0.4378, 0.06792, 0.5996], [-0.4429, 0.05694, 0.611], [-0.45, 0.04281, 0.6466]])

# Visualize data
fig = plt.figure(1)
ax1 = fig.add_subplot(221)
ax3 = fig.add_subplot(223, projection='3d')
ax4 = fig.add_subplot(224, projection='3d')

ax1.imshow(image)

# show scattered skeleton points
print(prediction.shape)

kp_visible = np.ones(42)
kp_visible = kp_visible == kp_visible

ax3.scatter(prediction[kp_visible, 0], prediction[kp_visible, 1], prediction[kp_visible, 2])

ax3.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z')

# plot the skeleton in 3d space

kp_visible = np.ones(5)
kp_visible = kp_visible == kp_visible

# Keypoints available:
# 0: left wrist,
# 1-4: left thumb [tip to palm],
finger = np.flip(np.vstack((prediction[1:5], prediction[0])))
ax4.plot(finger[kp_visible, 0], finger[kp_visible, 1], finger[kp_visible, 2])
# 5-8: left index,
finger = np.flip(np.vstack((prediction[5:9], prediction[0])))
ax4.plot(finger[kp_visible, 0], finger[kp_visible, 1], finger[kp_visible, 2])
# 9-12: 3 finger,
finger = np.flip(np.vstack((prediction[9:13], prediction[0])))
ax4.plot(finger[kp_visible, 0], finger[kp_visible, 1], finger[kp_visible, 2])
# 13-16 4 finger
finger = np.flip(np.vstack((prediction[13:17], prediction[0])))
ax4.plot(finger[kp_visible, 0], finger[kp_visible, 1], finger[kp_visible, 2])
# 17-20: left pinky,
finger = np.flip(np.vstack((prediction[17:21], prediction[0])))
ax4.plot(finger[kp_visible, 0], finger[kp_visible, 1], finger[kp_visible, 2])

# 21: right wrist,
# 22-25: right thumb,
finger = np.flip(np.vstack((prediction[22:26], prediction[21])))
ax4.plot(finger[kp_visible, 0], finger[kp_visible, 1], finger[kp_visible, 2])
# 26-29: 2 finger
finger = np.flip(np.vstack((prediction[26:30], prediction[21])))
ax4.plot(finger[kp_visible, 0], finger[kp_visible, 1], finger[kp_visible, 2])
# 30-33: 3 finger
finger = np.flip(np.vstack((prediction[30:34], prediction[21])))
ax4.plot(finger[kp_visible, 0], finger[kp_visible, 1], finger[kp_visible, 2])
# 34-37: 4 finger
finger = np.flip(np.vstack((prediction[34:38], prediction[21])))
ax4.plot(finger[kp_visible, 0], finger[kp_visible, 1], finger[kp_visible, 2])
# 38-41: right pinky
finger = np.flip(np.vstack((prediction[38:42], prediction[21])))
ax4.plot(finger[kp_visible, 0], finger[kp_visible, 1], finger[kp_visible, 2])

ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_zlabel('z')

plt.show()
