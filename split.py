
#info：将训练集中的数据按照一定比例划分为训练集、验证机以及测试集
# -*- coding: utf-8 -*-
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transformer_ImageNet = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transformer_ImageNet = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


class MyDataset(Dataset):
    def __init__(self, filenames, labels, transform):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):  # 因为漏了这行代码，花了一个多小时解决问题
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.labels[idx]


def fetch_dataloaders(data_dir, ratio, batchsize=64):
    """ the sum of ratio must equal to 1"""
    dataset = ImageFolder(data_dir)
    character = [[] for i in range(len(dataset.classes))]
    for x, y in dataset.samples:  # 将数据按类标存放
        character[y].append(x)

    train_inputs, val_inputs, test_inputs = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    for i, data in enumerate(character):
        num_sample_train = int(len(data) * ratio[0])
        num_sample_val = int(len(data) * ratio[1])
        num_val_index = num_sample_train + num_sample_val

        for x in data[:num_sample_train]:
            train_inputs.append(str(x))
            train_labels.append(i)
        for x in data[num_sample_train:num_val_index]:
            val_inputs.append(str(x))
            val_labels.append(i)
        for x in data[num_val_index:]:
            test_inputs.append(str(x))
            test_labels.append(i)

    train_dataloader = DataLoader(MyDataset(train_inputs, train_labels, train_transformer_ImageNet), batch_size=batchsize, drop_last=True, shuffle=True)
    val_dataloader = DataLoader(MyDataset(val_inputs, val_labels, val_transformer_ImageNet), batch_size=batchsize, drop_last=True, shuffle=True)
    test_dataloader = DataLoader(MyDataset(test_inputs, test_labels, val_transformer_ImageNet), batch_size=batchsize, shuffle=False)

    loader = {}
    loader['train'] = train_dataloader
    loader['val'] = val_dataloader
    loader['test'] = test_dataloader

    return loader


if __name__ == '__main__':
    data_dir = './data'
    """ 每一类图片有1300张，其中780张用于训练，260张用于测试，260张用于测试"""
    loader = fetch_dataloaders(data_dir, [0.6, 0.2, 0.2], batchsize=64)
    k=0
    for x, y in loader['train']:
        cv2.imwrite("img"+k+".jpg",x)
        k=k+1
