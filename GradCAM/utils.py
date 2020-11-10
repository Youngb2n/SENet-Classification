import cv2
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import os
from PIL import Image

def find_resnet_layer(arch, target_layer_name):
    '''
        target_layer_name = 'conv1'
        target_layer_name = 'block1'
        target_layer_name = 'block2'
        target_layer_name = 'block3'
        target_layer_name = 'block4'
        target_layer_name = 'avgpool'
        target_layer_name = 'linear'
    '''
    if 'block' in target_layer_name:
        layer_num = int(target_layer_name.lstrip('block'))
        if layer_num == 1:
            target_layer = arch.block1
        elif layer_num == 2:
            target_layer = arch.block2
        elif layer_num == 3:
            target_layer = arch.block3
        elif layer_num == 4:
            target_layer = arch.block4
    else:
        target_layer = arch._modules[target_layer_name]

    return target_layer

def find_vgg_layer(arch, target_layer_name):
    '''
        target_layer_name = 'features'
        target_layer_name = 'classifier'
    '''
    if 'features' == target_layer_name:
        target_layer = arch.features
    elif 'classifier' == target_layer_name:
        target_layer = arch.classifier
    else:
        target_layer = arch._modules[target_layer_name]

    return target_layer

def find_densenet_layer(arch, target_layer_name):
    '''
        target_layer_name = 'conv1'
        target_layer_name = 'block1'
        target_layer_name = 'block2'
        target_layer_name = 'block3'
        target_layer_name = 'block4'
        target_layer_name = 'avgpool'
        target_layer_name = 'linear'
    '''
    if 'block' in target_layer_name:
        layer_num = int(target_layer_name.lstrip('block'))
        if layer_num == 1:
            target_layer = arch.block1
        elif layer_num == 2:
            target_layer = arch.block2
        elif layer_num == 3:
            target_layer = arch.block3
        elif layer_num == 4:
            target_layer = arch.block4
    else:
        target_layer = arch._modules[target_layer_name]

    return target_layer

def image_loader(image_path):
    image = Image.open(image_path)

    loader = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()]) 

    image = loader(image).unsqueeze(0)
    return image.to(torch.float)


def imshow(img, model):
    unloader = transforms.ToPILImage()
    pred = model(img.unsqueeze(0)).squeeze()
    pred = F.softmax(pred, dim=0).detach().numpy()
    if pred[0] >pred[1]:
        pred = pred[0]
        name ='cat '
    else:
        pred = pred[1]
        name ='dog '
    image = img.cpu().clone() 
    image = image.squeeze(0)    
    image = unloader(image)
    plt.title('{} {:.2f}%'.format(name,pred*100))
    plt.imshow(image)
    try:
        if not os.path.exists('/content/ex/workspace'):
            os.makedirs('/content/ex/workspace')
    except OSError:
        print('Error Creating director')
    plt.savefig('/content/ex/workspace/gradcam.png', dpi=100)
    print('saved gradcam image...')


def visualize_cam(mask, img):

    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    
    result = heatmap+img.cpu()
    result = result.div(result.max()).squeeze()
    
    return heatmap, result
