""" Basic example showing samples from the dataset.

    Chose the set you want to see and run the script:
    > python view_samples.py

    Close the figure to see the next sample.
"""
from __future__ import print_function, unicode_literals

import json
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

image = plt.imread('data/hiu_dmtl_data/valid/abpkifzhye.png')
with open('data/hiu_dmtl_data/valid/abpkifzhye.json', 'r') as jsonfile:
    label = json.load(jsonfile)
prediction = np.array(label['pts3d_2hand'])

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
