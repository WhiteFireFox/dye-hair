import cv2
import glob
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    path_list = glob.glob('CelebAMask-HQ/train_label/*.png')
    pbar = tqdm(path_list)
    for path in pbar:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # label 13 is the hair
        label = np.ones(img.shape) * 13
        mask = np.array(img == label).astype(int)
        cv2.imwrite('./dataset/train/mask/'+path.split('/')[-1], mask)