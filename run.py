from os.path import exists
import torch
import models.cnn
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.models as torch_models
import tensorflow as tf

model = None

device = "cuda" if torch.cuda.is_available() else "cpu"


def forward(trained_model, image):
    # TODO: evaluate the image using given model, you may want to do some post-processing

    image_tensor = tf.convert_to_tensor(trained_model)
    result = trained_model(image_tensor)

    return result


def visualize(image, prediction):
    # TODO: project the vectors into a 3d space, visualize it side by side with the original image.

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

    return 0


def main(model_type='CNN', model_param_path='./params/param.pt'):
    global model
    if model_type == 'CNN':
        model = models.cnn.CNN(320, 320).to(device)

    if exists(model_param_path):
        model = torch_models.vgg16()
        model.load_state_dict(torch.load(model_param_path))

    # TODO: implementing eval and visualize functions, use these two function to visualize skeleton of input hand image.
    set_id = 'evaluation'
    sample_id = 1

    image = plt.imread(os.path.join('data/RHD_published_v2', set_id, 'color', '%.5d.png' % sample_id))
    prediction = forward(model, image)
    visualize(image, prediction)

    return 0


if __name__ == "__main__":
    main()
