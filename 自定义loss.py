import torch
import torch.nn as nn
#继承torch.nn.Module类写自己的loss。

class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, x, y):
        loss = torch.mean((x - y) ** 2)
        return loss

criterial=MyLoss()
a=torch.tensor([0.,0.])
b=torch.tensor(([1.,1.]))

loss=criterial(b,a)
print(loss)