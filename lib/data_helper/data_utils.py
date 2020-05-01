from __future__ import print_function, absolute_import

# Import
import cv2
import torch
import numpy as np
from lib.config import cfg  # from ..config import cfg


# Load Image
def load_image(img_path, resize=None, is_norm=True, toTesor=False):
    # Load Image [RGB (w*h*c)]
    img = cv2.imread(img_path, 0)

    # norm with / 255
    img = img / 255 if is_norm and img.max() > 1 else img

    # Resize
    img = cv2.resize(img, (resize[1], resize[0])) if resize is not None else img

    # to Tensor
    img = torch.from_numpy(img) if toTesor else img

    return img

def label_process(label):
    label = cv2.resize(label, (int(cfg.DATASET.IMG_RESIZE[0] / 8),
                               int(cfg.DATASET.IMG_RESIZE[1] / 8)))
    label_pixel = imgBinarization(label)
    label = 1 if label.sum() > 0 else 0

    return label_pixel, label

def imgBinarization(img, threshold=0.5):
    img = np.asarray(img)
    image = np.where(img > threshold, 1, 0)
    return image


def mean_std(self, cfg, is_train):

    if is_train:
        image_names = self.data_list['img']
        data_len = len(image_names)

        mean = torch.zeros(1)
        std = torch.zeros(1)
        for path in image_names:
            # Load
            img = load_image(path, is_norm=True, toTesor=True)

            # Compute
            mean += img.mean()
            std += img.std()
        mean /= data_len
        std /= data_len

        # Save
        meanstd = {'mean': mean, 'std': std}
        torch.save(meanstd, cfg.DATASET.ROOT + 'meanstd.pth')
        print('Down!! ', meanstd)

    else:
        meanstd = torch.load(cfg.DATASET.ROOT + 'meanstd.pth')

    return meanstd['mean'], meanstd['std']

