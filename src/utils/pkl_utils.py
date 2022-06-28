import os
import pickle
import numpy as np
from PIL import Image

def write_pkl(img_paths, labels, to_path, mode):
    '''
        img_paths: ['a/b/c', 'a/b/d'] 文件路径list
        labels: [1, 1, 1, 2, 2]
        to_path: 输出路径
    '''
    img_ndarray = [np.array(Image.open(path).resize((128, 128)), dtype=np.uint8) for path in img_paths]
    y = labels
    data = {'image_data': img_ndarray, 'class_dict': y}
    with open(os.path.join(to_path + '-' + mode + '.pkl'), 'wb') as f:
        pickle.dump(data, f, True)

def read_pkl(path, mode):
    with open(os.path.join(path, 'stanfordCars' + '-' + mode + '.pkl'), 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        x = data['image_data']
        y = data['class_dict']

    return x, y

