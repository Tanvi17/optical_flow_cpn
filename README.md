# optical_flow_cpn


1. todo -> 
    - takes path to the dataset +DONE
    - loads data +DONE
    - splits train-test data +DONE
    - performs image data preprcessing and aug +DONE
    - data aug on flo imgs +DONE
    - create model - flows
    - create cpn model - images
    - feeds to the model
    - initialize loss function and optimizer
    - train the data
2. Code References:
    - https://pytorch.org/vision/main/auto_examples/plot_optical_flow.html#sphx-glr-auto-examples-plot-optical-flow-py



On the RGB images, the data transform should be:
1. (Preprocessing) a histogram equalization -- I think this is the Pytorch function, it should take in a PIL Image +DONE
2. (Preprocessing) Adjust the range from [0,255] to [0,1] -- torchvision.transforms.ToTensor +DONE
3. (Preprocessing) subtract 0.5 from all pixels so that range becomes [-0.5,0.5] +DONE
4. (Data Augmentation) random crop to height=384 x width=448. (original image size is height=384 x width=512). When a network takes (img1, flow) or (img1, img2), both inputs need to have the same random crop. (Pytorch function, but you should know this one.) +DONE
5. (Data Augmentation) random affine transformation (scale + rotate + translate). Parameters are on this line. Pytorch has this built-in too. +DONE
6. (Data Augmentation) random flip. This is a vertical flip, horizontal flip, or 180 degree rotation with equal probability. +DONE
