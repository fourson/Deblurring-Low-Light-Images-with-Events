import numpy as np
import cv2


def RGB_to_bayerRGGB(rgb):
    bayer_raw = np.zeros_like(rgb[:, :, 0])  # (H, W)
    r, g, b = cv2.split(rgb)
    bayer_raw[0::2, 0::2] = r[0::2, 0::2]
    bayer_raw[0::2, 1::2] = g[0::2, 1::2]
    bayer_raw[1::2, 0::2] = g[1::2, 0::2]
    bayer_raw[1::2, 1::2] = b[1::2, 1::2]
    return bayer_raw


def bayerRGGB_to_RGB(bayer_raw):
    # cv2.COLOR_BAYER_BG2RGB requires uint8
    RGB = cv2.cvtColor(np.uint8(bayer_raw * 255), cv2.COLOR_BAYER_BG2RGB)
    RGB = np.float32(RGB) / 255
    return RGB


class LowLightNoise:
    """
        simulate low-light noise
    """

    def __init__(self):
        self.sigma1 = np.random.uniform(0.01, 0.02)
        self.sigma2 = np.random.uniform(0.01, 0.02)
        self.sigma3 = np.random.uniform(0.01, 0.02)
        self.w = np.random.uniform(0.65, 0.75)

    def add_noise(self, img, rgb=True):
        if rgb:
            # add noise in bayer domain
            img_bayer = RGB_to_bayerRGGB(img)
            img_bayer_noisy = np.float32(np.random.normal(img_bayer, self.sigma1 * img_bayer))
            img_bayer_noisy = np.clip(img_bayer_noisy, a_min=0, a_max=1)  # clip for converting back to img domain
            img_noisy1 = bayerRGGB_to_RGB(img_bayer_noisy)

            # add noise in img domain
            img_noisy2 = np.float32(np.random.normal(img, self.sigma2 * img))

            # weighted sum
            img_noisy = img_noisy1 * self.w + img_noisy2 * (1 - self.w)
        else:
            # add noise in img domain
            img_noisy = np.float32(np.random.normal(img, self.sigma2 * img))

        # add quantization noise
        img_noisy = np.clip(img_noisy, a_min=0, a_max=1)
        img_noisy = np.float32(np.random.normal(img_noisy, self.sigma3 * img_noisy))
        img_noisy = np.clip(img_noisy, a_min=0, a_max=1)
        img_noisy = np.floor(img_noisy * 255) / 255

        return img_noisy
