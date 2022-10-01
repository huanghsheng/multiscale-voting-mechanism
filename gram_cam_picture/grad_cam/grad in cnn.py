import os
import numpy as np
import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image
from model import EfficientNet
import cv2 #opencv读取的格式是BGR

def gram_incnn(net,input,labels):
    n,c,h,w = input.shape
    for i in range (n) :
        cam = GradCAM(model=net, target_layers=input[i], use_cuda=True)
        target_category = labels[i]
        grayscale_cam = cam(input_tensor=input[i], target_category=target_category)
        grayscale_cam = grayscale_cam[0, :]
        # print(grayscale_cam.shape)
        grayscale_cam = grayscale_cam * 255
        grayscale_cam = grayscale_cam.astype("uint16")
        ret, thresh3 = cv2.threshold(grayscale_cam, 140, 255, cv2.THRESH_TOZERO)
        thresh3 = np.float32(thresh3) / 255
        thresh3 = torch.tensor(thresh3)
        end = torch.unsqueeze(thresh3,0)
        end = end.repeat(c,1,1)
        end = end.cuda()
        input[i] = end
    return input

# a = torch.Tensor([[[1,2],[3,4]],[[5,6],[7,8]],[[7,1],[8,2]],[[9,3],[4,6]]])
# print(a.shape)
# end = torch.unsqueeze(a,0)
# end = end.repeat(4,1,1,1)
# print(end.shape)
# print(end)
# # b=a.repeat(2,1,1)
# # print(b.shape)
# # print(b)