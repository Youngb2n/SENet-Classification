from GradCAM import GradCAM
from GradCAM import utils
import argparse
import modellist
import torch

modellist = modellist.Modellist()

parser = argparse.ArgumentParser(description='show GradCAM')
parser.add_argument('path',type=str, help='image path')
parser.add_argument('modelnum',type=int, help='Select your model number')
parser.add_argument('state_dict_path',type=str, help='Select your model number')
parser.add_argument("-se", help="Put the selayer in the model.",
                    action="store_true")

args = parser.parse_args()

#load image
img = utils.image_loader(args.path)

#load model & state_dict
model = modellist(args.modelnum, seon = args.se)
model.load_state_dict(torch.load(args.state_dict_path))

if  args.modelnum in range(1,5):
    model_type ='vgg'
elif args.modelnum in range(5,10):
    model_type ='resnet'
    layer_name = 'block4'
else:
    model_type = 'densenet'

model.eval()
model_dict = dict(model_type=model_type, arch=model, layer_name=layer_name, input_size=(224,224))
cam =GradCAM.GradCAM(model_dict)
mask = cam(img)
_, result = utils.visualize_cam(mask, img)
utils.imshow(result,model)
