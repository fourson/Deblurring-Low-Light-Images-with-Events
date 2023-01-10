import os
import cv2
import numpy as np

for f in os.listdir(os.getcwd()):
    if f.endswith('.npy'):
        name = f.split('.')[0]
        patch_wise_kernel = np.load(f)
        cv2.imwrite(name + '.png', patch_wise_kernel / patch_wise_kernel.max() * 255)
