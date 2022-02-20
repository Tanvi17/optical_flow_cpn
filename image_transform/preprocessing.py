from torchvision.transforms import CenterCrop, RandomCrop, RandomEqualize
from PIL import Image

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

class Process():
    def __init__(self, image):
        self.image = image

    def histogram_equalization(self):
        self.image = Image.open(self.image)
        # self.image = RandomEqualize(p=1).forward(self.image)
        # self.image.save('randomEqualize.png')
        return RandomEqualize(p=1).forward(self.image)

    def preprocessing(self):
        pass

if __name__ == '__main__':
    Process()
