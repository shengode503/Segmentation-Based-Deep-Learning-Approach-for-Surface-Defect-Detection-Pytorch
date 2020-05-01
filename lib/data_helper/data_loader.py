import os
import random
import pickle
from lib.data_helper.data_utils import *  # from .data_utils import *
import torchvision.transforms as transforms


class Data_loader(torch.utils.data.Dataset):
    def __init__(self, cfg, is_train):

        self.is_train = is_train

        # load split List
        list_name = 'defect_split_list.pkl' \
            if cfg.TRAIN.TRAIN_MODEL == 'SegNet' \
            else 'split_list.pkl'

        with open(os.path.join(cfg.DATASET.ROOT, list_name), 'rb') as f:
            train_list, val_list = pickle.load(f)
        self.data_list = train_list if is_train else val_list

        # mean_std
        self.mean, self.std = mean_std(self, cfg, is_train)

    def __getitem__(self, index):
        image_path = self.data_list['img'][index]
        label_path = self.data_list['label'][index]

        # Load data
        w, h = cfg.DATASET.IMG_RESIZE[0], cfg.DATASET.IMG_RESIZE[1]
        image = load_image(image_path, resize=[h, w])
        target = load_image(label_path, is_norm=False, resize=[h, w])

        if self.is_train:
            # Flip
            if cfg.TRAIN.IS_FLIP and random.random() > cfg.TRAIN.FLIP_PROB:
                td_flip_prob = random.random()
                image = image[::-1, :] if td_flip_prob > .5 else image[:, ::-1]
                target = target[::-1, :] if td_flip_prob > .5 else target[:, ::-1]

            # Dilation
            if cfg.TRAIN.IS_DILATE and random.random() > cfg.TRAIN.DILATE_PROB:
                ks = random.randint(3, 15)
                target = cv2.dilate(target, kernel=np.ones((ks, ks), np.uint8))

        # Lable_process
        target_pixel, label = label_process(target)
        target_pixel = target_pixel.squeeze()
        label = np.asarray([label])

        # dtype
        image, target_pixel, label = np.float32(image), np.float32(target_pixel), np.float32(label)

        # transform
        normalize = transforms.Normalize(mean=self.mean.numpy().tolist(),
                                         std=self.std.numpy().tolist())
        trans = transforms.Compose([transforms.ToTensor(), normalize])
        image = trans(image)

        return image, target_pixel, label, [image_path, label_path]

    def __len__(self):
        return len(self.data_list['img'])


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        Data_loader(cfg, is_train=True),
        batch_size=5, shuffle=False,
        num_workers=0, pin_memory=False)


    for i, (image, label_pixel, label, info) in enumerate(train_loader):

        label_pixel = label_pixel.detach().to('cpu').numpy()
        image = image.detach().to('cpu').numpy()

        if i == 0:
            break

    # Plot
    plt.figure(0)
    plt.imshow(label_pixel[0])
    plt.figure(1)
    plt.imshow(image[0][0])
