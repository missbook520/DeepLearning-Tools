import torch
import torch.nn as nn

# convolutional neural network (2 convolutional layers)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNet().to(device)

#计算模型整体参数量
#torch.numel() 返回一个tensor变量内所有元素个数，可以理解为矩阵内元素的个数
num_parameters = sum(torch.numel(parameter) for parameter in model.parameters())

print("模型整体参数量为：",num_parameters)

# 计算模型所占内存大小
print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

#查看网络中的参数
#可以通过model.state_dict()或者model.named_parameters()函数查看现在的全部可训练参数（包括通过继承得到的父类中的参数）
params = list(model.named_parameters())
(name, param) = params[0]
print(name)
print(param.grad)
print('-------------------------------------------------')
(name2, param2) = params[1]
print(name2)
print(param2.grad)
print('----------------------------------------------------')
(name1, param1) = params[2]
print(name1)
print(param1.grad)
