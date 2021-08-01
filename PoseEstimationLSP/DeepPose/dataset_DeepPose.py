#适用于DeepPose模型的数据处理

import torch
import numpy as np
import os
import scipy.io
from PIL import Image
from torchvision.transforms import *


def standardize_label(label, orim):  #保证了原图像的坐标的准确
    label_std = []
    for idx, _ in enumerate(label):
        labelX = label[idx][0] / orim.size[0]  #x的值除于原始图像的宽
        labelY = label[idx][1] / orim.size[1]  #y的值除于原始图像的高
        label_std.append([labelX, labelY])
    label_std = np.array(label_std)
    # print(label_std)
    return label_std

class PoseImageDataset(torch.utils.data.Dataset):
    def __init__(self, transforms, imagespath='', labelsfilepath=''):

        imgs_list = sorted(os.listdir(os.path.join(imagespath)))  # 获得文件夹内的图片的名称列表
        self.filenames = imgs_list

        #将注释文件加载到矩阵中
        self.annotationmat = scipy.io.loadmat(labelsfilepath)
        # print(self.annotationmat) #加载.mat文件的数据

        joints = self.annotationmat['joints']
        # print(joints) # 只加载'joints'键的数据
        # print(joints.shape) # (3, 14, 2000)

        joints = np.swapaxes(joints, 2, 0)
        """
        print(joints)
        将0轴和2轴转换，使[ 29.74645941 143.34544031   0.        ]就为第一张图片的x轴和y轴以及二进制的值
        [[[ 29.74645941 143.34544031   0.        ]
          [ 30.5501068  117.22690013   0.        ]
          [ 28.94281202  84.67918082   0.        ]
          ...
        """

        labels = []
        images = []
        origin_image_size = []

        for file_idx, file_name in enumerate(imgs_list):
            fn = imgs_list[file_idx]
            orim = Image.open(os.path.join(imagespath,fn))
            origin_image_size.append(orim.size)
            # print(orim)   # Image.open根据拼接的路径获取图像信息

            # print(self.transforms)
            image1 = transforms(orim)  #将图像信息归一化
            # print(image1.shape)

            label = joints[file_idx]
            # print(label)

            #standardizing标准化
            label = standardize_label(label, orim)
            # print(label)

            label = torch.from_numpy(label)  # torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。
            label1 = label.type(torch.FloatTensor)

            images.append(image1)
            labels.append(label1)

        self.images = images
        self.labels = labels
        self.orim_size = origin_image_size

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.orim_size[idx]

    def __len__(self):
        return len(self.filenames)

