import cv2
import os
import random
import numpy as np

class DataLoader(object):
    def __init__(self, batchsize=6):
        self.batchsize = batchsize
        self.img_path = './data/train/img/'
        self.label_path = './data/train/mask/'
        self.img_list = self.read()

    def __call__(self):
        index_list = list(range(len(self.img_list)))
        random.shuffle(index_list)
        imgs_list = []
        masks_list = []
        for i in index_list:
            img = cv2.imread(self.img_path+self.img_list[i])
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR).transpose([2, 0, 1])
            mask = cv2.imread(self.label_path+self.img_list[i].split('.')[0]+'.png', cv2.IMREAD_GRAYSCALE)
            imgs_list.append(img.astype('float32'))
            masks_list.append(mask.astype('int64'))
            if len(imgs_list) == self.batchsize:
                yield np.array(imgs_list), np.array(masks_list)
                imgs_list = []
                masks_list = []
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(masks_list)

    def read(self):
        img_list = os.listdir(self.img_path)
        return img_list