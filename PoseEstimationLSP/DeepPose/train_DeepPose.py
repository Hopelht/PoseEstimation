#运行主入口

from dataset_DeepPose import PoseImageDataset
from torchvision.transforms import *
from DeepPose import DeepPose
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import Accuracy
import scipy.io
import Visualization
from torch.autograd import Variable


from PIL import Image


from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先使用显卡

print(f"using {DEVICE}")
image_path = '../images'
labels_file_path = '../joints.mat'

batchsize = 20
EPOCHS = 1
image_size = 196


transforms = Compose([
    Resize((image_size,image_size)),
    ToTensor(),  #张量化
    #ToTensor()能够把灰度范围从0-255变换到0-1之间，
    # 而transform.Normalize()则把0-1变换到(-1,1).具体地说，对每个通道而言
    Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])



#加载自己的数据集
dataset = PoseImageDataset(transforms, image_path, labels_file_path)
# print(dataset.images.shape)  #(2000, 14, 3)


#数据集划分
total = len(dataset)
train_size, test_size = int(total * 0.8), int(total * 0.2)
lengths = [train_size, test_size]
train_dataset, test_dataset = torch.utils.data.dataset.random_split(dataset, lengths)


train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True)


#模型
model = DeepPose()

#迁移学习
if os.path.exists('./chpcurrent_HourGlass.chp'):
    d = torch.load('./chpcurrent_HourGlass.chp')
    model.load_state_dict(d)  #加载参数



parameters = model.parameters()
# print(parameters) #返回参数的迭代器

lossFunction = torch.nn.MSELoss(reduction='sum')  #均方差损失函数
optimizer = torch.optim.Adam(parameters, lr=0.00001)  # 优化器


def train():
    model.train()
    loss_value = 0
    for tmp, (images, labels, _) in enumerate(train_dataloader):
        # print(labels.shape)
        # print(images.shape)

        # print(images)
        images, labels = images.to(DEVICE), labels.to(DEVICE)  #方便用于gpu


        predict = model(images)  # 把图像数据放进model里
        # print(predict.shape)
        # print(labels.shape)
        # print(predict.shape)

        predict =predict.view(labels.shape)
        # print(predict)

        loss_value = lossFunction(predict, labels)  # 将标签和图像数据同时放入

        # 反向传播
        optimizer.zero_grad()  # 梯度置零，也就是把loss关于weight的导数变成0.
        loss_value.backward()
        optimizer.step()


        if tmp > 0:
            if (tmp % 10) == 0:
                print('checkpoint guardado' + str(tmp))
                torch.save(model.state_dict(), './chpcurrent_HourGlass.chp')

    return loss_value.item()

def test():
    model.eval()
    test_loss_total = 0

    images_accuracy = []

    for images, labels, orim_size in test_dataloader:
        # print(labels.shape)  # torch.Size([32, 14, 3])
        # print(images.shape)  # torch.Size([32, 14, 3])
        # print(orim_size[0])

        # print(images)
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        # print(labels)


        # print(images.shape)  # torch.Size([32, 3, 64, 64])
        # print(images.shape)
        predict = model(images)  # 把图像数据放进model里
        # print(predict.shape)  # torch.Size([32, 42])
        # print(predict)

        predict =predict.view(labels.shape)  #将预测的关节点坐标整理成原始坐标的形状

        for idx, _ in enumerate(labels):
            image_accuracy = Accuracy.accuracy(predict[idx], labels[idx], image_size)
            images_accuracy.append(image_accuracy)

        loss_value = lossFunction(predict, labels)  # 将标签和图像数据同时放入

        test_loss_total += loss_value


    accuracy = sum(images_accuracy) / len(images_accuracy)  #batch的准确率
    test_loss_total /= len(test_dataset)

    # print("准确率：", accuracy)
    return test_loss_total, accuracy



test_loss, accuracy = [], []


# 批训练EPOCHS
for epoch in range(1, EPOCHS+1):

    train_loss = train()  #训练

    test_loss_, accuracy_ = test()   #测试
    print(f"第{epoch}/{EPOCHS}次训练，训练损失值为{train_loss}, 测试损失值为{test_loss_},测试准确率为{accuracy_}")

    test_loss1 = test_loss_.detach().numpy()
    test_loss.append(test_loss1)
    accuracy.append(accuracy_)

Visualization.chart(test_loss, accuracy)  #损失值以及准确率的可视化




#挑选一张照片进行验证
sample_idx = 1  # 第几张照片
model.eval()

imgs_list = os.listdir(os.path.join(image_path))  # 获得文件夹内的图片的名称列表
orim = Image.open(os.path.join(image_path, imgs_list[sample_idx - 1]))  #原始图像

annotationmat = scipy.io.loadmat(labels_file_path)
joints1 = annotationmat['joints']
joints = np.swapaxes(joints1, 2, 0)

label = joints[sample_idx - 1]
label_ = []

for idx, _ in enumerate(label):   #标签处理
    labelX = label[idx][0]
    labelY = label[idx][1]
    label_.append([labelX, labelY])

label_std = np.array(label_)
label2 = torch.from_numpy(label_std)  # torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。
labels_ = label2.type(torch.FloatTensor)

image1 = transforms(orim)  # 将图像信息归一化
image1 = Variable(torch.unsqueeze(image1, dim=0).float(), requires_grad=False)  #整理成模型批训练的维度

image, labels = image1.to(DEVICE), labels_.to(DEVICE)
predict = model(image)  # 把图像数据放进model里

predict_ = predict.view(label_std.shape)

predict_list = []
for idx, _ in enumerate(predict_):
    predictX = predict_[idx][0].detach().numpy() * orim.size[0]
    predictY = predict_[idx][1].detach().numpy() * orim.size[1]
    predict_list.append([predictX, predictY])

predicts = np.array(predict_list)
# print(labels)

Visualization.print_sample(predicts, labels, orim)  #原始joints和预测joints可视化


