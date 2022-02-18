import numpy as np
import torchvision
from skimage import io, exposure
import pdb
import os
import random
from glob import glob   # matches path names with same pattern in the folder
import matplotlib.pyplot as plt
from utils.input_utils import read_gen
import torch
import torch.utils.data as data
from torchvision.transforms import CenterCrop, RandomCrop, RandomEqualize
from matplotlib import transforms
from PIL import Image


# dataset path
# path = "/media/common/datasets/scene_flow_datasets/FlyingChairs_release/"
# flyingchairs_data = torchvision.datasets.FlyingChairs(  root=path, 
#                                                         split='train', 
#                                                         transforms=None)
# data_loader = torch.utils.data.DataLoader(flyingchairs_data)
# print(data_loader)

class StaticRandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        h, w = image_size
        self.h1 = random.randint(0, h - self.th)
        self.w1 = random.randint(0, w - self.tw)

    def __call__(self, img):
        return img[self.h1:(self.h1+self.th), self.w1:(self.w1+self.tw),:]

class StaticCenterCrop(object):
    def __init__(self, image_size, crop_size):
        self.th, self.tw = crop_size
        self.h, self.w = image_size
    def __call__(self, img):
        return img[(self.h-self.th)//2:(self.h+self.th)//2, (self.w-self.tw)//2:(self.w+self.tw)//2,:]

class HistogramEqualizer(object):

    # img_obj = HistogramEqualizer(image)
    # img_obj()

    def __init__(self, img_ip):
        self.img_ip = img_ip.astype(np.float32)

    def __call__(self):
        channels = 3
        for i in range(channels):
            pixels_per_channel = self.img_ip[:,:,i]

            # created cummulative frequency of pixels against the instensity value
            cdf, bin_centers = exposure.cumulative_distribution(pixels_per_channel, nbins=256)

            img_interpolated = np.interp(pixels_per_channel.flat, bin_centers, cdf)
            
            pixels_per_channel = img_interpolated.reshape(pixels_per_channel.shape)
            
            # equalize the image pixels
            xmax = np.amax(pixels_per_channel)
            xmin = np.amin(pixels_per_channel)
            dx = xmax - xmin
            img_eq = (pixels_per_channel - xmin)/dx
            
            self.img_ip[:,:,i] = img_eq

        return self.img_ip


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
        
        # if self.is_cropped:
        #     # cropper = StaticRandomCrop(image_size, self.crop_size)
        #     cropper = RandomCrop(image_size)
        # else:
        #     # cropper = StaticCenterCrop(image_size, self.render_size)
        #     cropper = CenterCrop(image_size)

        # images = list(map(cropper, images))
        # flow = cropper(flow)

        images = np.array(images).transpose(3,0,1,2)
        flow = flow.transpose(2,0,1)

        images = torch.from_numpy(images.astype(np.float32))
        flow = torch.from_numpy(flow.astype(np.float32))

        return [images], [flow]

    def __len__(self):
        return self.size * self.replicates

dataset = FlyingChairs()
dataset[0]