'''所需数据文件索引文件train.txt,内容示例如下：
	其中0和1代表图像的label
	/home/data/images/1.JPEG 0
	/home/data/images/2.JPEG 0
	/home/data/images/3.JPEG 1
	/home/data/images/4.JPEG 1
	
'''
'''优化方法：
	创建两个队列：

输入图像队列：用于存放多线程并行加载和预处理得到的图像。
CUDA图像队列：将输入图像从“输入图像队列”传输到GPU内存。
在给定的示例中，采用4个并行的线程(workers)来进行输入图像的加载和预处理，处理后的结果会push到共享的输入图像队列，此外还必须保证图像生成器的线程安全。采用1个线程来实现输入图像传输到GPU内存。由此可以看出，输入图像的加载和预处理，CUDA传输以及GPU训练都在同时进行，大大减少了空闲等待时间，提高了资源利用率。
from:https://blog.csdn.net/winycg/article/details/92443146
'''






import threading
import random
import torch
import time
import argparse
from queue import Empty, Queue
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image

parser = argparse.ArgumentParser(description='More quick data loading for Pytorch')
parser.add_argument('-b', '--batch-size', default=8, type=int)
parser.add_argument('-epochs', '--num-epoches', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('-pj', '--preprocess-workers', default=4, type=int,
                    help='number of works for preprocessing data')
parser.add_argument('-cj', '--cuda-workers', default=1, type=int,
                    help='number of works for transfering tensors from CPU memory to CUDA memory')
parser.add_argument('-tm', '--train-batches-queue-maxsize', default=12, type=int,
                    help='maxsize of train batches queue')
parser.add_argument('-cm', '--cuda-batches-queue-maxsize', default=1, type=int,
                    help='maxsize of cuda batches queue')

args = parser.parse_args()
print(args)

class threadsafe_iter(object):
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def get_path_i(paths_count):
    """Cyclic generator of paths indice
    """
    current_path_id = 0
    while True:
        yield current_path_id
        current_path_id = (current_path_id + 1) % paths_count



class InputGen:
    def __init__(self, paths, batch_size):
        self.paths = paths
        self.index = 0
        self.batch_size = batch_size
        self.init_count = 0
        self.lock = threading.Lock()  # mutex for input path
        self.yield_lock = threading.Lock()  # mutex for generator yielding of batch
        self.path_id_generator = threadsafe_iter(get_path_i(len(self.paths)))
        self.images = []
        self.labels = []

    def pre_process_input(self, im):
        """ Do your pre-processing here
                Need to be thread-safe function"""
        transformer = transforms.Compose([
            transforms.RandomCrop(32),
            transforms.ToTensor()])
        im = transformer(im)
        return im

    def __next__(self):
        return self.__iter__()

    def __iter__(self):
        while True:
            # In the start of each epoch we shuffle the data paths
            with self.lock:
                if self.init_count == 0:
                    random.shuffle(self.paths)
                    self.images, self.labels = [], []
                    self.init_count = 1
            # Iterates through the input paths in a thread-safe manner
            for path_id in self.path_id_generator:
                try:
                    img, label = self.paths[path_id].split(' ')
                except ValueError:
                    continue  # ['\n']错误
                img = Image.open(img, "r")
                img = self.pre_process_input(img)
                # Concurrent access by multiple threads to the lists below
                with self.yield_lock:
                    self.images.append(img)
                    self.labels.append(float(label))
                    if len(self.images) % self.batch_size == 0:
                        yield torch.stack(self.images, dim=0), torch.tensor(self.labels, dtype=torch.long)
                        self.images, self.labels = [], []
            # At the end of an epoch we re-init data-structures
            with self.lock:
                self.init_count = 0

    def __call__(self):
        return self.__iter__()

class thread_killer(object):
    """Boolean object for signaling a worker thread to terminate
    """

    def __init__(self):
        self.to_kill = False

    def __call__(self):
        return self.to_kill

    def set_tokill(self, tokill):
        self.to_kill = tokill


def threaded_batches_feeder(tokill, batches_queue, dataset_generator):
    """Threaded worker for pre-processing input data.
    tokill is a thread_killer object that indicates whether a thread should be terminated
    dataset_generator is the training/validation dataset generator
    batches_queue is a limited size thread-safe Queue instance.
    """
    while tokill() == False:
        for batch, (batch_images, batch_labels) in enumerate(dataset_generator):
            # We fill the queue with new fetched batch until we reach the max       size.
            batches_queue.put((batch, (batch_images, batch_labels)), block=True)
            if tokill() == True:
                return


def threaded_cuda_batches(tokill, cuda_batches_queue, batches_queue):
    """Thread worker for transferring pytorch tensors into
    GPU. batches_queue is the queue that fetches numpy cpu tensors.
    cuda_batches_queue receives numpy cpu tensors and transfers them to GPU space.
    """
    while tokill() == False:
        batch, (batch_images, batch_labels) = batches_queue.get(block=True)
        batch_images = batch_images.cuda()
        batch_labels = batch_labels.cuda()
        cuda_batches_queue.put((batch, (batch_images, batch_labels)), block=True)
        if tokill() == True:
            return


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


if __name__ == '__main__':
    model = LeNet()
    model.cuda()
    model.train()
    # Training set list suppose to be a list of full-paths for all the training images
    with open('train.txt') as f:
        training_set_list = f.readlines()
    batches_per_epoch = len(training_set_list) // args.batch_size
    # Once the queue is filled the queue is locked.
    train_batches_queue = Queue(maxsize=args.train_batches_queue_maxsize)
    # Our torch tensor batches cuda transferer queue.
    # Once the queue is filled the queue is locked
    cuda_batches_queue = Queue(maxsize=args.cuda_batches_queue_maxsize)

    training_set_generator = InputGen(training_set_list, args.batch_size)
    train_thread_killer = thread_killer()
    train_thread_killer.set_tokill(False)

    # We launch 4 threads to do load &amp;&amp; pre-process the input images
    for _ in range(args.preprocess_workers):
        t =threading.Thread(target=threaded_batches_feeder,
                            args=(train_thread_killer, train_batches_queue, training_set_generator))
        t.start()
    cuda_transfers_thread_killer = thread_killer()
    cuda_transfers_thread_killer.set_tokill(False)

    for _ in range(args.cuda_workers):
        cudathread = threading.Thread(target=threaded_cuda_batches,
                                      args=(cuda_transfers_thread_killer, cuda_batches_queue, train_batches_queue))
        cudathread.start()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True)
    for epoch in range(args.num_epoches):
        print('epochs:', epoch)
        for batch in range(batches_per_epoch):
            # We fetch a GPU batch in 0's due to the queue mechanism
            _, (batch_images, batch_labels) = cuda_batches_queue.get(block=True)

            # train batch is the method for your training step.
            # no need to pin_memory due to diminished cuda transfers using queues.
            def train_batch(batch_images, batch_labels):
                optimizer.zero_grad()
                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                return loss.item()
            loss = train_batch(batch_images, batch_labels)
            print('batch %d, loss: %.4f' % (batch, loss))


    train_thread_killer.set_tokill(True)
    cuda_transfers_thread_killer.set_tokill(True)
    for _ in range(args.preprocess_workers):
        try:
            # Enforcing thread shutdown
            train_batches_queue.get(block=True, timeout=1)
            cuda_batches_queue.get(block=True, timeout=1)
        except Empty:
            pass
    print('training done!')
