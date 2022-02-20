import torch
import torch.nn as nn
import torch.nn.functional as F


#construct conditional prior network

'''
imH = 384
imW = 448
oriH= 384
oriW= 512
takes only img1; not img2
cropped size image; random crop to height=384 x width=448
inputs:
    img1_ph, flow5_up, name=Prior_1_name='Mani_1', orisize=[imH,imW], 
    reuse=False, f=Prior_1_factor=0.25,1.0, training=Prior_1_train=False
     
'''

class ConditionalPriorNet(nn.Module):
    def __init__(self):
      super(ConditionalPriorNet, self).__init__()
      self.conv1 = nn.Conv2d(1, 32, 3, 1)
      self.conv2 = nn.Conv2d(32, 64, 3, 1)
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout2d(0.5)
      self.fc1 = nn.Linear(9216, 128)
      self.fc2 = nn.Linear(128, 10)

    # x represents our data
    def forward(self, x):
      # Pass data through conv1
      x = self.conv1(x)
      # Use the rectified-linear activation function over x
      x = F.relu(x)

      x = self.conv2(x)
      x = F.relu(x)

      # Run max pooling over x
      x = F.max_pool2d(x, 2)
      # Pass data through dropout1
      x = self.dropout1(x)
      # Flatten x with start_dim=1
      x = torch.flatten(x, 1)
      # Pass data through fc1
      x = self.fc1(x)
      x = F.relu(x)
      x = self.dropout2(x)
      x = self.fc2(x)

      # Apply softmax to x
      output = F.log_softmax(x, dim=1)
      return output