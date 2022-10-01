# from efficientnet_pytorch import EfficientNet
from model_Fip import EfficientNet
import os
from PIL import Image
import json
import torchvision
from torchvision import transforms
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import nn
from prettytable import PrettyTable
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class MyDataset(torch.utils.data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, is_train, root):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()
        fh = open(root, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []  # 创建一个名为img的空列表，一会儿用来装东西
        for line in fh:  # 按行循环txt文本中的内容
            line = line.rstrip()  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            imgs.append((words[0], int(words[1])))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            # word[0]是文件名, word[1]为标签
        self.imgs = imgs
        self.is_train = is_train

        if self.is_train:
            self.train_tsf = torchvision.transforms.Compose([
                # torchvision.transforms.CenterCrop(size=224),
                torchvision.transforms.Resize((300,400)),
                torchvision.transforms.ToTensor()
            ])
        else:
            self.test_tsf = torchvision.transforms.Compose([
                # torchvision.transforms.CenterCrop(size=224),
                torchvision.transforms.Resize((300,400)),
                torchvision.transforms.ToTensor()])

        # 归一化
        # if self.is_train:
        #     self.train_tsf = torchvision.transforms.Compose([
        #         torchvision.transforms.RandomResizedCrop(224, scale=(0.1, 1), ratio=(0.5, 2)),
        #         torchvision.transforms.ToTensor()
        #     ])
        # else:
        #     self.test_tsf = torchvision.transforms.Compose([
        #         torchvision.transforms.Resize(size=224),
        #         torchvision.transforms.CenterCrop(size=224),
        #         torchvision.transforms.ToTensor(),
        #         torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        feature, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        feature = Image.open(feature).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片
        if self.is_train:
            feature = self.train_tsf(feature)
        else:
            feature = self.test_tsf(feature)
        return feature, label

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "F1-score"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            F1score = round(2 * Precision * Recall / (Precision + Recall), 3) if Precision + Recall != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, F1score])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # train = r'E:\\dataset\\MTL\\dataset1\\train.txt'
    test = r'F:\EfficientNet-Pytorch-master\efficientnet_realsize\400-300-2_test.txt'
    # train_data = MTLmodel2.MyDataset(is_train=False, root=train)
    test_data = MyDataset(is_train=False, root=test)
    # train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=12, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0)
    print("training on ", device)
    net =EfficientNet.from_name("efficientnet-b0")
    net._fc1 = nn.Linear(100, 5)

    pretext_model = torch.load('./b0_disease_model_400-300-2_Fip.pth')
    net_dict = net.state_dict()
    state_dict = {k: v for k, v in pretext_model.items() if k in net_dict.keys()}
    net_dict.update(state_dict)
    net.load_state_dict(net_dict)

    # model_weight_path = "./MTL_disease_model_24.pth"
    # assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.to(device)

    # read class_indict
    json_label_path = './class_indices_5.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=5, labels=labels)
    net.eval()

    # 下面5行是为了输出识别正确与错误的样本做的准备工作
    c_w=1 # 统计测试集中识别错误的数量
    c_r=1 # 统计测试集中识别正确的数量
    # img_wrong_path=r'F:\EfficientNet-Pytorch-master\efficientnet_chidu\pythoncodes\11.5\3_fc_add_0+3+4\analysis\wrong\wrong1'
    # img_right_path=r'F:\EfficientNet-Pytorch-master\efficientnet_chidu\pythoncodes\11.5\3_fc_add_0+3+4\analysis\right\right1'
    img_wrong_path = r'F:\EfficientNet-Pytorch-master\analysis\wrong\wrong1'
    img_right_path = r'F:\EfficientNet-Pytorch-master\analysis\right\right1'
    batch_size=4

    with torch.no_grad():
        for train_data in tqdm(test_loader):
            test_images, test_labels = train_data
            outputs = net(test_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)

            # 输出识别正确与错误的样本
            labels=test_labels.to(device) #这一步是将测试集标签格式转成tensor
            is_correct=(labels==outputs).squeeze().to(device)
            k=0

            for k in range(batch_size):
                # 下面3步是将图像的tensor格式转为numpy格式，以便cv2输出
                img = test_images[k].mul(255).byte()
                # img= test_images[k]*255
                img=img.numpy().squeeze().transpose((1, 2, 0))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if is_correct[k]:
                    filename = img_right_path + 'image_%d_%d.jpg' %(c_r,labels[k])
                    cv2.imwrite(filename, img)  #使用cv2实现图片与numpy数组的相互转化
                    c_r+=1
                else:
                    filename = img_wrong_path + 'image_%d_%d_%d.jpg' %(c_w,labels[k],outputs[k])
                    cv2.imwrite(filename, img)
                    c_w+=1


            confusion.update(outputs.to("cpu").numpy(), test_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()

# print(net)