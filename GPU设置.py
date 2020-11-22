import torch
import os
#如果只需要一张显卡
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

#如果需要多卡训练
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
