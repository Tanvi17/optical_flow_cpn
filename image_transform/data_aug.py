# 4. (Data Augmentation) random crop to height=384 x width=448. (original image size is height=384 x width=512). 
# When a network takes (img1, flow) or (img1, img2), both inputs need to have the same random crop. 
# (Pytorch function, but you should know this one.)
# 5. (Data Augmentation) random affine transformation (scale + rotate + translate). 
#     Parameters are on this line. Pytorch has this built-in too.
# 6. (Data Augmentation) random flip. 
#     This is a vertical flip, horizontal flip, or 180 degree rotation with equal probability.

