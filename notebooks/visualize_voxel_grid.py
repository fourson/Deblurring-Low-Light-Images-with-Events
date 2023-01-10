import os
import cv2
import numpy as np

for f in os.listdir(os.getcwd()):
    if f.endswith('.npy'):
        name = f.split('.')[0]
        voxel_grid = np.load(f)
        single_frame = np.sum(voxel_grid, axis=2)
        H, W = single_frame.shape
        visual_image = np.zeros((H, W, 3), dtype=np.uint8)

        positive = np.expand_dims(np.uint8((single_frame > 0)), axis=2)
        no_polarity = np.expand_dims(np.uint8((single_frame == 0)), axis=2)
        negative = np.expand_dims(np.uint8((single_frame < 0)), axis=2)

        visual_image += positive * np.array([255, 0, 0], dtype=np.uint8) + no_polarity * np.array(
            [255, 255, 255], dtype=np.uint8) + negative * np.array([0, 0, 255], dtype=np.uint8)
        cv2.imwrite(name + '.png', visual_image)
