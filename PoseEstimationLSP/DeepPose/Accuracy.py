#准确率的计算：预测的关节点和原始的关节点用距离公式判断准确率，即在圆内的为准确
#其中距离的标准自定义

imagespath='../images'
deviation_distance = 5     #标准定为5


def accuracy(predict, labels, img_size):
    #处理一个图片的准确率
    joints = 14
    correct = 0  #正确的个数

    for idx, _ in enumerate(labels):
        # print(predict[idx] * img_size, labels[idx] * img_size)
        deviation = ((predict[idx] - labels[idx]) * img_size) ** 2  #使用距离公式
        deviation = sum(deviation)
        # print(deviation)
        # print(deviation_distance ** 2)

        if deviation <= deviation_distance ** 2:
            correct += 1

    accuary = correct / joints

    return accuary