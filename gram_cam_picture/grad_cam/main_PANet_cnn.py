import os
import numpy as np
import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image
from model_PANet import EfficientNet

def main():
    # model = models.mobilenet_v3_large(pretrained=True)
    # target_layers = [model.features[-1]]
    # print(target_layers)

    model = EfficientNet.from_name("efficientnet-b0")
    model._fc1 = nn.Linear(50, 5)
    # pthfile = './b0_disease_model_FPN_cat_zeyou_vote_1.pth'
    pthfile = './b0_disease_model_PANet_400-300-2_1000~1200epo.pth'
    # pthfile = './b0_disease_model_FPN_cat_zeyou_vote_3.pth'
    model.load_state_dict(torch.load(pthfile))
    model.eval()  # 8
    # target_layers = [model._blocks[10]]
    # target_layers = [model.FPNlayers]
    target_layers = []
    target_layers.append(model.lastsmooth3)
    target_layers.append(model.smooth3)
    target_layers.append(model.lastsmooth1)
    target_layers.append(model.lastsmooth2)

    print(target_layers)

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]
    # print(target_layers)

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ])
    # load image
    img_path = "./end/2-2.jpg"
    # img_path = "both.png"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = 2  # tabby, tabby cat
    # target_category = 281  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    main()
