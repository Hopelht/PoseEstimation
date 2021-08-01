# PoseEstimation AI姿态估计



一部分内容在个人博客（包括参考的博客和GitHub代码）：

https://blog.csdn.net/weixin_51107732/article/details/119304737





## 1.文件介绍

+ DeepPose文件夹

  + Accuracy文件：准确率计算

  + DeepPose文件：DeepPose模型搭建

  + Visualization文件：可视化

  + dataset_DeepPose文件：适用于DeepPose的数据集处理

  + train_DeepPose文件：程序主入口

    

+ HourGlass文件夹

  + Accuracy文件：准确率计算
  + HourGlass文件：hourglass模型搭建
  + Visualization文件：可视化
  + dataset_hourglass文件：适用于hourglass的数据集处理
  + train_hourglass文件：程序主入口





## 2.运行

+ 可以在这里下面网址获取数据集

  https://blog.csdn.net/weixin_42216109/article/details/115269420?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522162782717116780271519197%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=162782717116780271519197&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-3-115269420.first_rank_v2_pc_rank_v29&utm_term=LSP%E6%95%B0%E6%8D%AE%E9%9B%86&spm=1018.2226.3001.4187



+ 下载完后将里面的images文件夹和joints.mat文件放到PoseEstimationLSP目录下

  即如下图：
![图片](https://user-images.githubusercontent.com/75004553/127774787-20e0babf-d071-46b2-8b18-1182069256b1.png)
  

  执行程序主入口文件即可运行


PS：该模型还没做优化



