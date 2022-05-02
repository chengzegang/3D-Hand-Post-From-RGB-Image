from os.path import exists
import torch
from models.layers import Swin
import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.models as torch_models
import tensorflow as tf
import torchvision.transforms as transforms
import PIL.Image as Image
from torch.autograd import Variable

model = None

device = "cuda" if torch.cuda.is_available() else "cpu"

loader = transforms.Compose([transforms.ToTensor()])


def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
    return image.cuda()  # assumes that you're using GPU


def forward(trained_model, imagename):
    # TODO: evaluate the image using given model, you may want to do some post-processing

    image = image_loader(imagename)

    result = trained_model(image)

    # torch tensor to np array

    return result.data.cpu().numpy()


def visualize(image, prediction):
    prediction = prediction[0]

    # TODO: project the vectors into a 3d space, visualize it side by side with the original image.

    # Visualize data
    fig = plt.figure(1)
    ax1 = fig.add_subplot(221)
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='3d')

    ax1.imshow(image)

    # show scattered skeleton points

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
    finger = np.flip(np.vstack((prediction[0], prediction[1:5])))
    ax4.plot(finger[kp_visible, 0], finger[kp_visible, 1], finger[kp_visible, 2])
    # 5-8: left index,
    finger = np.flip(np.vstack((prediction[0], prediction[5:9])))
    ax4.plot(finger[kp_visible, 0], finger[kp_visible, 1], finger[kp_visible, 2])
    # 9-12: 3 finger,
    finger = np.flip(np.vstack((prediction[0], prediction[9:13])))
    ax4.plot(finger[kp_visible, 0], finger[kp_visible, 1], finger[kp_visible, 2])
    # 13-16 4 finger
    finger = np.flip(np.vstack((prediction[0], prediction[13:17])))
    ax4.plot(finger[kp_visible, 0], finger[kp_visible, 1], finger[kp_visible, 2])
    # 17-20: left pinky,
    finger = np.flip(np.vstack((prediction[0], prediction[17:21])))
    ax4.plot(finger[kp_visible, 0], finger[kp_visible, 1], finger[kp_visible, 2])

    # 21: right wrist,
    # 22-25: right thumb,
    finger = np.flip(np.vstack((prediction[21], prediction[22:26])))
    ax4.plot(finger[kp_visible, 0], finger[kp_visible, 1], finger[kp_visible, 2])
    # 26-29: 2 finger
    finger = np.flip(np.vstack((prediction[21], prediction[26:30])))
    ax4.plot(finger[kp_visible, 0], finger[kp_visible, 1], finger[kp_visible, 2])
    # 30-33: 3 finger
    finger = np.flip(np.vstack((prediction[21], prediction[30:34])))
    ax4.plot(finger[kp_visible, 0], finger[kp_visible, 1], finger[kp_visible, 2])
    # 34-37: 4 finger
    finger = np.flip(np.vstack((prediction[21], prediction[34:38])))
    ax4.plot(finger[kp_visible, 0], finger[kp_visible, 1], finger[kp_visible, 2])
    # 38-41: right pinky
    finger = np.flip(np.vstack((prediction[21], prediction[38:42])))
    ax4.plot(finger[kp_visible, 0], finger[kp_visible, 1], finger[kp_visible, 2])

    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('z')

    plt.show()

    return 0


def main(model_type='CNN', model_param_path='./params/param.pt'):
    global model
    if model_type == 'CNN':
        model = Swin().to(device)

    if exists(model_param_path):
        model.load_state_dict(torch.load(model_param_path))
        model.to(device)

    # TODO: implementing eval and visualize functions, use these two function to visualize skeleton of input hand image.
    set_id = 'evaluation'
    sample_id = 1

    image_dir = os.path.join('data', 'RHD_published_v2', 'evaluation', 'color', '00003.png')
    image = plt.imread(image_dir)
    prediction = forward(model, image_dir)
    visualize(image, prediction)

    return 0


if __name__ == "__main__":
    main()
