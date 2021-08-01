#可视化
import matplotlib.pyplot as plt

import torch
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先使用显卡



# loss和accuracy数据可视化
def chart(lossList, accuracyList):
    plt.figure(figsize=(13, 6))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    ax1.set(xlabel='Epoch', ylabel='Loss')
    ax1.plot(range(1, len(lossList)), lossList, label='Test Loss', color='r')
    ax1.legend(loc='upper right', borderpad=0.4)

    ax2.set(xlabel='Epoch', ylabel='Accuracy')
    ax2.plot(range(1, len(accuracyList)), accuracyList, label='Test Accuracy')
    ax2.legend(loc='upper right', borderpad=0.4)

    plt.show()



#原始joints和预测joints可视化
def print_sample(predict, labels, orim):

    original_img = orim
    normalized_img = orim

    # fig, (ax1, ax2) = plt.subplots(1, 2)  # figsize=(14,14)
    ax1 = plt.subplot(121)  # figsize=(14,14)
    ax2 = plt.subplot(122)

    ax1.title.set_text('original joints')
    ax1.imshow(original_img)
    for idx in range(14):
        ax1.plot(labels[idx][0], labels[idx][1], '.', color='b')

    ax2.title.set_text('predict joints')
    ax2.imshow(normalized_img)
    for idx in range(14):
        ax2.plot(predict[idx][0], predict[idx][1], '.', color='b')

    # fig.tight_layout()
    plt.show()

