#PyTorch可以使用tensorboard来可视化训练过程。

#安装和运行TensorBoard。

# pip install tensorboard
# tensorboard --logdir=runs

#使用SummaryWriter类来收集和可视化相应的数据，放了方便查看，可以使用不同的文件夹，比如'Loss/train'和'Loss/test'。

# from torch.utils.tensorboard import SummaryWriter
# import numpy as np
#
# writer = SummaryWriter()
#
# for n_iter in range(100):
#     writer.add_scalar('Loss/train', np.random.random(), n_iter)
#     writer.add_scalar('Loss/test', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/test', np.random.random(), n_iter)