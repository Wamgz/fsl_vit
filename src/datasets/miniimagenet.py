import numpy as np
from PIL import Image
import pickle as pkl
import os
import glob
import csv
import torch.utils.data as data
from torchvision import transforms as transforms
import sys
if '..' not in sys.path:
    sys.path.append('..')

from src.utils.parser_util import get_parser
from src.utils.logger_utils import logger

import torch

class MiniImageNet(data.Dataset):
    '''
    类别数量：100 个类别，每个类别 600 张图片，共计 60,000 张图片。
    数据内容：RGB 图片，.jpg 格式，分辨率 84x84。
    数据切分：训练集 64 个类，验证集 16 个类，测试集 20 个类。
    '''
    def __init__(self, mode, opt, transform=None, target_transform=None):
        self.im_width, self.im_height, self.channels = 3, 84, 84
        self.split = mode
        self.root = opt.dataset_root
        self.x = []
        self.y = []
        if transform is None: # 使用默认的transform
            tsfm = []
            tsfm.append(transforms.ToPILImage())

            if opt.height > 0 and opt.width > 0:
                self.im_width, self.im_height, self.channels = opt.width, opt.height, opt.channel
                tsfm.append(transforms.Resize((self.im_height, self.im_width)))

            transforms.RandomCrop(self.im_width, padding=4),  # 先四周填充0，在把图像随机裁剪成32*32
            transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.RandomRotation((-45, 45)),  # 随机旋转
            tsfm.append(transforms.ToTensor())

            # tsfm.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225)))
            self.transform = transforms.Compose(tsfm)

        #1、读取pkl文件
        pkl_name = '{}/miniImagenet/data/mini-imagenet-cache-{}.pkl'.format(self.root, self.split)
        logger.info('Loading pkl data: {} '.format(pkl_name))

        try:
            with open(pkl_name, "rb") as f:
                data = pkl.load(f, encoding='bytes')
                image_data = data[b'image_data']
                class_dict = data[b'class_dict']
        except:
            with open(pkl_name, "rb") as f:
                data = pkl.load(f)
                image_data = data['image_data']  # (38400, 84, 84, 3)
                class_dict = data['class_dict']  # dict, key_num = 64, {'n01532829': [0, 1, 2···599]}
        self.x = image_data
        self.y = np.arange(len(class_dict.keys())).reshape(1, -1).repeat(600, 1).squeeze(0)
        print()

    def __getitem__(self, idx):
        x = self.x[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

    def __len__(self):
        return len(self.x)

    def total_classes(self):
        return 100

class FastCollate:
    def __init__(self, nw, ns, nq, bs):
        self.nw, self.ns, self.nq = nw, ns, nq
        self.bs = bs

    def __call__(self, batch):
        # batch [(img, label), (img, label), (img, label), ···] 共batch
        imgs = [img[0] for img in batch]
        targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)

        w = imgs[0].shape[1]
        h = imgs[0].shape[2]
        episodic_targets = torch.from_numpy(
            np.repeat(np.repeat(range(self.nw), self.ns + self.nq), self.bs)
        ).reshape(-1, self.bs).permute(1, 0).reshape(-1)
        # print(targets, episodic_targets)
        return tensor, targets, episodic_targets


if __name__ == '__main__':
    options = get_parser().parse_args()
    dataset = MiniImageNet(mode='train', opt=options)
    classes_per_it = 20
    num_samples = 20
    from prototypical_batch_sampler import PrototypicalBatchSampler
    sampler = PrototypicalBatchSampler(labels=dataset.y,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=100)
    from tqdm import tqdm


    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler,
                                              collate_fn=FastCollate(20, 5, 15, 1))
    data_loader = iter(data_loader)
    for batch in tqdm(data_loader):
        tensor, labels, episode_labels = batch
        print()