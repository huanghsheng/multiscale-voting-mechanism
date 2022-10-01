# import torchvision.models as models
# from flops_counter import get_model_complexity_info
# net=models.inception_v3()
# flops.params = get_model_complexity_info(net,(300,400),as_strings=True,print_per_layer_stat=True)
# print('Flops: {}'.format(flops))
# print("Params: " + params)



import torchvision.models as models
import torch
from ptflops import get_model_complexity_info

with torch.cuda.device(0):
  # net = models.resnet50()
  # net = models.inception_v3()
  net = models.densenet121()
  macs, params = get_model_complexity_info(net, (3, 300, 400), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))