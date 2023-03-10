import os

import numpy as np
import cv2


def viz_flow(flow):
    # 色调H：用角度度量，取值范围为0°～360°，从红色开始按逆时针方向计算，红色为0°，绿色为120°,蓝色为240°
    # 饱和度S：取值范围为0.0～1.0
    # 亮度V：取值范围为0.0(黑色)～1.0(白色)
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # flownet是将V赋值为255, 此函数遵循flownet，饱和度S代表像素位移的大小，亮度都为最大，便于观看
    # 也有的光流可视化讲s赋值为255，亮度代表像素位移的大小，整个图片会很暗，很少这样用
    hsv[..., 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def viz_multiple_flow(multiple_flow):
    # multiple_flow: (H, W, 2*N)
    H, W, C = multiple_flow.shape
    N = C // 2
    bgrs = []
    for i in range(N):
        flow = multiple_flow[:, :, 2 * i:2 * (i + 1)]
        bgr_flow = viz_flow(flow)
        bgrs.append(bgr_flow)
    return bgrs


for f in os.listdir(os.getcwd()):
    if f.endswith('.npy'):
        name = f.split('.')[0]
        multiple_flow = np.load(f)
        bgrs = viz_multiple_flow(multiple_flow)
        for idx, bgr in enumerate(bgrs):
            cv2.imwrite('{}_flow{}.png'.format(name, idx), bgr)
