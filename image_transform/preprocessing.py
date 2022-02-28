from torchvision.transforms import CenterCrop, RandomCrop, RandomEqualize
from PIL import Image
import numpy as np

# 1. (Preprocessing) a histogram equalization -- I think this is the Pytorch function, 
# it should take in a PIL Image
# 2. (Preprocessing) Adjust the range from [0,255] to [0,1] -- torchvision.transforms.ToTensor
# 3. (Preprocessing) subtract 0.5 from all pixels so that range becomes [-0.5,0.5]

'''
NORMALIZATION:
- converts the PIL image with a pixel range of [0, 255] to a PyTorch FloatTensor of shape (C, H, W) with a range [0.0, 1.0]
- transforming the images into such values that the mean and standard deviation of the image become 0.0 and 1.0 respectively. 
- To do this first the channel mean is subtracted from each input channel and then the result is divided by the channel standard deviation:

output[channel] = (input[channel] - mean[channel]) / std[channel]

- Normalization helps get data within a range and reduces the skewness which helps learn faster and better. 
- Normalization can also tackle the diminishing and exploding gradients problems.
'''

class HistogramEqualization():
    def __init__(self):
        return
        # self.image_path = image_path

    def __call__(self, image_path):
        self.image = Image.open(image_path)
        # self.image = RandomEqualize(p=1).forward(self.image)
        # self.image.save('randomEqualize.png')
        return RandomEqualize(p=1).forward(self.image) #returns PIL image

class ReadFromFile():
    def __init__(self):
        return
        # self.image_path = image_path

    def __call__(self, image_path):
        """ Read .flo file in Middlebury format"""
        # Code adapted from:
        # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

        # WARNING: this will work on little-endian architectures (eg Intel x86) only!
        with open(image_path, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                print('Magic number incorrect. Invalid .flo file')
                return None
            else:
                w = np.fromfile(f, np.int32, count=1)
                h = np.fromfile(f, np.int32, count=1)
                # print 'Reading %d x %d flo file\n' % (w, h)
                data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
                # Reshape data into 3D array (columns, rows, bands)
                # The reshape here is for visualization, the original code is (w,h,2)
                return np.resize(data, (int(h), int(w), 2))

class CustomRange():
    def __call__(self, array):
        try:
            if array is not None:
                return array - 0.5
            else:
                return None
        except TypeError:
            return np.array(array) - 0.5


if __name__ == '__main__':
    HistogramEqualization()
