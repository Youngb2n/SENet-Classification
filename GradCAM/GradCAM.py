import torch
import torch.nn.functional as F
import numpy as np
from GradCAM import utils

class GradCAM(object):
    def __init__(self,model_dict):
        #model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        model_type = model_dict['model_type']
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        
        def forward_hook(module,input,output):
            self.activations['value'] = output
            return None

        if 'vgg' in model_type.lower():
            target_layer = utils.find_vgg_layer(self.model_arch,layer_name)
        if 'resnet' in model_type.lower():
            target_layer = utils.find_resnet_layer(self.model_arch,layer_name)
        if 'densenet' in model_type.lower():
            target_layer = utils.find_densenet_layer(self.model_arch,layer_name)

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def forward(self, inputs, class_idx = None, retain_graph=False):
        b,c,h,w =inputs.size()
        logit =self.model_arch(inputs)
        if class_idx is None:
            score =logit[:,logit.max(1)[-1].squeeze()]
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)

        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()
        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        saliency_map = (weights*(activations)).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
        
        return saliency_map

    def __call__(self,inputs,class_idx=None,retain_graph=False):
        return self.forward(inputs,class_idx, retain_graph)
