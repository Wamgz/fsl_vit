import numpy as np
from PIL import Image
import pickle as pkl
import os
import glob
import csv
import torch.utils.data as data
from torchvision import transforms as transforms
import scipy.io
import logging
import sys
import pickle
if '..' not in sys.path:
    sys.path.append('..')

from src.utils.logger_utils import logger

class StanfordCars(data.Dataset):
    '''
    一共包含16185张不同型号的汽车图片，其中8144张为训练集，8041张为测试集
    196个类，train和test每个class的sample数量不低于60
    '''
    def __init__(self, mode, opt, transform=None, target_transform=None):
        self.im_width, self.im_height, self.channels = opt.width, opt.height, opt.channel
        self.split = mode
        self.root = opt.dataset_root
        self.x = []
        self.y = []
        if transform is None: # 使用默认的transform
            tsfm = []
            if opt.height > 0 and opt.width > 0:
                self.im_width, self.im_height, self.channels = opt.width, opt.height, opt.channel
                tsfm.append(transforms.Resize((self.im_height, self.im_width)))

            tsfm.append(transforms.ToTensor())
            tsfm.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.229, 0.224, 0.225)))
            self.transform = transforms.Compose(tsfm)

        dir = os.path.join(self.root, opt.dataset_name)
        data_dir = os.path.join(dir, 'data')
        mode2image = {'train': [], 'val': [], 'test': []} ## TODO 如何划分
        image2label = {}
        logger.info('Loading data_dir data: {}, mode: {} '.format(data_dir, mode))
        with open(os.path.join(dir, 'mat2txt.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                split = line.strip().split(' ')
                cur_mode = 'train' if split[2] == '0' else 'test'
                img_path = os.path.join(os.path.join(data_dir, split[0]))
                mode2image[cur_mode].append(img_path)
                image2label[img_path] = int(split[1]) - 1
            mode2image['val'] = mode2image['test']

        ## 直接读取pkl文件
        with open(os.path.join(dir, 'stanfordCars' + '-' + mode + '.pkl'), 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            self.x = data['image_data']
            self.y = data['class_dict']

    def __getitem__(self, idx):
        x = self.x[idx]
        x = Image.fromarray(x)
        if len(x.split()) < 3:
            x = x.convert('RGB')
        if self.transform:
            x = self.transform(x)
        return x, self.y[idx]

    def __len__(self):
        return len(self.x)

## 将批量的文件打包为一个pkl文件
def write_pkl(mode):
    dir = os.path.join('/Users/wangzi/PycharmProjects/few-shot-learning/Prototypical-Networks-for-Few-shot-Learning-PyTorch/data', 'stanfordCars')
    data_dir = os.path.join(dir, 'data')
    mode2image = {'train': [], 'val': [], 'test': []}  ## TODO 如何划分
    image2label = {}
    print('Loading data_dir data: {}, mode: {} '.format(data_dir, mode))
    with open(os.path.join(dir, 'mat2txt.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            split = line.strip().split(' ')
            cur_mode = 'train' if split[2] == '0' else 'test'
            img_path = os.path.join(os.path.join(data_dir, split[0]))
            mode2image[cur_mode].append(img_path)
            image2label[img_path] = int(split[1]) - 1
        mode2image['val'] = mode2image['test']

    img_paths = mode2image[mode]
    img_ndarray = [np.array(Image.open(path).resize((128, 128)), dtype=np.uint8) for path in img_paths]
    y = [image2label[path] for path in img_paths]
    data = {'image_data': img_ndarray, 'class_dict': y}
    with open(os.path.join(dir, 'stanfordCars' + '-' + mode + '.pkl'), 'wb') as f:
        pickle.dump(data, f, True)


if __name__ == '__main__':
    write_pkl('train')
    write_pkl('test')