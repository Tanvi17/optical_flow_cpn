import numpy as np
import torchvision
from skimage import io, exposure
import os
from glob import glob
import matplotlib.pyplot as plt
from utils.input_utils import read_gen
import torch
import torch.utils.data as data
from torchvision.transforms import CenterCrop, RandomCrop, RandomEqualize
from matplotlib import transforms
from PIL import Image
from utils import dataset_utils


def make_dataset(dir, split=None):
    '''Will search for triplets that go by the pattern 
    '[name]_img1.ppm  [name]_img2.ppm    [name]_flow.flo' '''
    print('inside creating dataset...')
    images = []
    flow_list = sorted(glob(os.path.join(dir, 'data', '*.flo')))
    for flow_map in flow_list:
        flow_map_full = flow_map
        flow_map = os.path.basename(flow_map) #00001_flow.flo
        root_filename = flow_map[:-9] #00001

        img1 = os.path.join(dir, 'data', root_filename+'_img1.ppm')
        img2 = os.path.join(dir, 'data', root_filename+'_img2.ppm')
        if not (os.path.isfile(os.path.join(dir, 'data', img1)) and os.path.isfile(os.path.join(dir, 'data', img2))):
            print('here')
            continue

        images.append([[img1,img2],flow_map_full])
    print('len: ', len(images))
    #splits the data in test-train samples
    return dataset_utils.split2list(images, split, default_split=0.97)


def flying_chairs(root, transform, target_transform, co_transform, split=None):
    print('inside flying chairs..')
    train_list, test_list = make_dataset(root, split)
    train_dataset = dataset_utils.ListDataset(train_list, transform, co_transform, target_transform)
    test_dataset = dataset_utils.ListDataset(test_list, transform, co_transform=None, target_transform=target_transform)
    print("train-test shape: ", len(train_dataset), len(test_dataset))
    return train_dataset, test_dataset

########################################################
class FlyingChairs(data.Dataset):
    def __init__(self, root="/media/common/datasets/scene_flow_datasets/FlyingChairs_release/"):
        # self.args = args
        self.is_cropped = True
        # self.crop_size = args.crop_size
        # self.render_size = args.inference_size
        self.replicates = 1
        self.image_list = []

        images = sorted(glob(os.path.join(root, "data", '*.ppm')))

        self.flow_list = sorted(glob(os.path.join(root, "data", '*.flo')))

        assert (len(images)//2 == len(self.flow_list))

        for i in range(len(self.flow_list)):
            im1 = images[2*i]
            im2 = images[2*i + 1]
            self.image_list += [[im1, im2]]

        assert len(self.image_list) == len(self.flow_list)

        self.size = len(self.image_list)

        self.frame_size = read_gen(self.image_list[0][0]).shape

        # if (self.render_size[0] < 0) or (self.render_size[1] < 0) or (self.frame_size[0]%64) or (self.frame_size[1]%64):
        #     self.render_size[0] = ( (self.frame_size[0])//64 ) * 64
        #     self.render_size[1] = ( (self.frame_size[1])//64 ) * 64

        # args.inference_size = self.render_size

        for i in range(1):
            img1 = self.image_list[0][0]
            img2 = self.image_list[0][-1]
            img = Image.open(img1)
            img.save('my1.png')
            img.show()
            print("before== ", io.imread(img1).shape)

            #histogram equalize image
            rimg = RandomEqualize(p=1).forward(img)
            rimg.save('randomEqualize.png')


    def __getitem__(self, index):
        index = index % self.size

        img1 = read_gen(self.image_list[index][0])
        img2 = read_gen(self.image_list[index][1])
        flow = read_gen(self.flow_list[index])

        images = [img1, img2]
        image_size = img1.shape[:2]
        
        if self.is_cropped:
            # cropper = StaticRandomCrop(image_size, self.crop_size)
            cropper = RandomCrop(image_size)
        else:
            # cropper = StaticCenterCrop(image_size, self.render_size)
            cropper = CenterCrop(image_size)

        images = list(map(cropper, images))
        flow = cropper(flow)

        images = np.array(images).transpose(3,0,1,2)
        flow = flow.transpose(2,0,1)

        images = torch.from_numpy(images.astype(np.float32))
        flow = torch.from_numpy(flow.astype(np.float32))

        return [images], [flow]

    def __len__(self):
        return self.size * self.replicates
