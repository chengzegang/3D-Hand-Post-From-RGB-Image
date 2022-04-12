from os.path import exists

import torch

import models.cnn

model = None

def eval(model, image):
    # TODO: evaluate the image using given model, you may want to do some post-processing
    return 0


def visualize(image, prediction):
    # TODO: project the vectors into a 3d space, visualize it side by side with the original image.
    return 0



def main(model_type='CNN', model_param_path='./params/param.pt'):
    global model
    if model_type == 'CNN':
        model = models.cnn.CNN

    if exists(model_param_path):
        model.load_state_dict(torch.load(model_param_path))
    
    # TODO: implementing eval and visualize functions, use these two function to visualize skeleton of input hand image.
    return 0

if __name__ == "__main__":
    main()
