import numpy as np
import torch.utils.data as data
import os
import os.path
from imageio import imread
import numpy as np


def load_flo(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert(202021.25 == magic),'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    data2D = np.resize(data, (w, h, 2))
    return data2D


def default_loader(root, path_imgs, path_flo):
    imgs = [os.path.join(root, path) for path in path_imgs]
    flo = os.path.join(root, path_flo)
    return [imread(img).astype(np.float32) for img in imgs],load_flo(flo)

#splits the data in test-train samples
def split2list(images, split, default_split=0.9):
    print('inside spliting dataset to test-train...')
    if isinstance(split, str):
        with open(split) as f:
            split_values = [x.strip() == '1' for x in f.readlines()]
        assert(len(images) == len(split_values))
    elif split is None:
        split_values = np.random.uniform(0,1,len(images)) < default_split
    else:
        try:
            split = float(split)
        except TypeError:
            print("Invalid Split value, it must be either a filepath or a float")
            raise
        split_values = np.random.uniform(0,1,len(images)) < split
    train_samples = [sample for sample, split in zip(images, split_values) if split]
    test_samples = [sample for sample, split in zip(images, split_values) if not split]
    return train_samples, test_samples


class ListDataset(data.Dataset):
    def __init__(self, path_list, transform, co_transform, target_transform):

        # self.root = root
        self.path_list = path_list
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        # self.loader = loader

    def __getitem__(self, index):
        inputs, target = self.path_list[index]

        # inputs, target = self.loader(self.root, inputs, target)
        if self.transform is not None:
            inputs[0] = self.transform(inputs[0])
            inputs[1] = self.transform(inputs[1])
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.co_transform is not None:
            inputs[0] = self.co_transform(inputs[0])
            inputs[1] = self.co_transform(inputs[1])
            target = self.co_transform(target)
            # inputs, target = self.co_transform(inputs, target)
        return inputs, target

    def __len__(self):
        return len(self.path_list)