import torch
import numpy as np
import PIL
import torchvision
#张量的基本信息

tensor_1=torch.randn(3,4,5)
print(tensor_1.type())
print(tensor_1.size())
print(tensor_1.device)
print(tensor_1.requires_grad)
print(tensor_1.dim())
print(tensor_1)

# 在PyTorch 1.3之前，需要使用注释
# Tensor[N, C, H, W]
images = torch.randn(32, 3, 56, 56)
images.sum(dim=1)
images.select(dim=1, index=0)

# PyTorch 1.3之后
#NCHW = [‘N’, ‘C’, ‘H’, ‘W’]
#images = torch.randn(32, 3, 56, 56, names=NCHW)
#images.sum('C')
#images.select('C', index=0)
# 也可以这么设置
#tensor = torch.rand(3,4,1,2,names=('C', 'N', 'H', 'W'))
# 使用align_to可以对维度方便地排序
#tensor = tensor.align_to('N', 'C', 'H', 'W')

#数据类型转换
# 设置默认类型，pytorch中的FloatTensor远远快于DoubleTensor
torch.set_default_tensor_type(torch.FloatTensor)

# 类型转换
tensor = tensor_1.cuda()
tensor = tensor_1.cpu()
tensor = tensor_1.float()
tensor = tensor_1.long()

#torch.Tensor与np.ndarray转换
ndarray = tensor.cpu().numpy()

tensor = torch.from_numpy(ndarray).float()

tensor = torch.from_numpy(ndarray.copy()).float() # If ndarray has negative stride.

#Torch.tensor与PIL.Image转换
# pytorch中的张量默认采用[N, C, H, W]的顺序，并且数据范围在[0,1]，需要进行转置和规范化

# torch.Tensor -> PIL.Image
image = PIL.Image.fromarray(torch.clamp(tensor*255, min=0, max=255).byte().permute(1,2,0).cpu().numpy())
image = torchvision.transforms.functional.to_pil_image(tensor)  # Equivalently way

# PIL.Image -> torch.Tensor
path = r'./girl.jpg'
tensor = torch.from_numpy(np.asarray(PIL.Image.open(path))).permute(2,0,1).float() / 255
tensor = torchvision.transforms.functional.to_tensor(PIL.Image.open(path)) # Equivalently way


#将张量打乱数据
tensor = tensor[torch.randperm(tensor_1.size(0))]  # 打乱第一个维度

#水平翻转
# pytorch不支持tensor[::-1]这样的负步长操作，水平翻转可以通过张量索引实现
# 假设张量的维度为[N, D, H, W].
#tensor = tensor[:,:,:,torch.arange(tensor.size(3) - 1, -1, -1).long()]

#计算两组数据间的欧式距离

#利用广播机制
#dist = torch.sqrt(torch.sum((X1[:,None,:] - X2) ** 2, dim=2))
a=torch.tensor([1.,1.,1.])
b=torch.tensor([2.,2.,2.])
#print(torch.sum((b-a)**2))
dist=torch.sqrt(torch.sum((b-a)**2))
print(dist)