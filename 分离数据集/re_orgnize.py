import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import shutil
import time
import pandas as pd
import random

# 设置随机数种子
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


def mkdir_if_not_exist(path):
    # 若目录path不存在，则创建目录
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))
        
def reorg_dog_data(data_dir, label_file, train_dir, test_dir, new_data_dir, valid_ratio):
    # 读取训练数据标签
    labels = pd.read_csv(os.path.join(data_dir, label_file))
    id2label = {Id: label for Id, label in labels.values}  # (key: value): (id: label)

    # 随机打乱训练数据
    train_files = os.listdir(os.path.join(data_dir, train_dir))
    random.shuffle(train_files)    

    # 原训练集
    valid_ds_size = int(len(train_files) * valid_ratio)  # 验证集大小
    for i, file in enumerate(train_files):
        img_id = file.split('.')[0]  # file是形式为id.jpg的字符串
        img_label = id2label[img_id]
        if i < valid_ds_size:
            mkdir_if_not_exist([new_data_dir, 'valid', img_label])
            shutil.copy(os.path.join(data_dir, train_dir, file),
                        os.path.join(new_data_dir, 'valid', img_label))
        else:
            mkdir_if_not_exist([new_data_dir, 'train', img_label])
            shutil.copy(os.path.join(data_dir, train_dir, file),
                        os.path.join(new_data_dir, 'train', img_label))
        mkdir_if_not_exist([new_data_dir, 'train_valid', img_label])
        shutil.copy(os.path.join(data_dir, train_dir, file),
                    os.path.join(new_data_dir, 'train_valid', img_label))

    # 测试集
    mkdir_if_not_exist([new_data_dir, 'test', 'unknown'])
    for test_file in os.listdir(os.path.join(data_dir, test_dir)):
        shutil.copy(os.path.join(data_dir, test_dir, test_file),
                    os.path.join(new_data_dir, 'test', 'unknown'))




data_dir = '/home/kesci/input/Kaggle_Dog6357/dog-breed-identification'  # 数据集目录
label_file, train_dir, test_dir = 'labels.csv', 'train', 'test'  # data_dir中的文件夹、文件
new_data_dir = './train_valid_test'  # 整理之后的数据存放的目录
valid_ratio = 0.1  # 验证集所占比例
reorg_dog_data(data_dir, label_file, train_dir, test_dir, new_data_dir, valid_ratio)
