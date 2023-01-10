import numpy as np
import matplotlib.pyplot as plt
import cv2


def triangle_fun_prod(x, y):
    return np.multiply(np.maximum(0, (1 - np.abs(x))), np.maximum(0, (1 - np.abs(y))))


class KernelGenerator:
    """
        Generate a blur kernel
        :param trajectory: the trajectory used to generate the blur kernel.
        :param canvas: size of domain where the blur kernel is defined.
    """

    def __init__(self, trajectory, canvas=16):
        self.trajectory = trajectory
        self.canvas = canvas

        self.kernel = None

    def generate(self):
        self.kernel = np.zeros((self.canvas, self.canvas), dtype=np.float32)

        for t in range(len(self.trajectory)):
            m2 = int(np.floor(self.trajectory[t].real))
            M2 = int(m2 + 1)
            m1 = int(np.floor(self.trajectory[t].imag))
            M1 = int(m1 + 1)

            self.kernel[m1, m2] += triangle_fun_prod(self.trajectory[t].real - m2, self.trajectory[t].imag - m1)
            self.kernel[m1, M2] += triangle_fun_prod(self.trajectory[t].real - M2, self.trajectory[t].imag - m1)
            self.kernel[M1, m2] += triangle_fun_prod(self.trajectory[t].real - m2, self.trajectory[t].imag - M1)
            self.kernel[M1, M2] += triangle_fun_prod(self.trajectory[t].real - M2, self.trajectory[t].imag - M1)

        # normalize (energy conservation constraint: the sum of the blur kernel should be 1)
        self.kernel /= np.sum(self.kernel)
        # flip upside to down
        self.kernel = np.flipud(self.kernel)

        return self.kernel

    def show_result(self):
        if self.kernel is None:
            raise Exception('Run self.generate() first!')
        plt.figure()
        plt.matshow(self.kernel)
        plt.show()
        plt.close()


class PatchWiseKernelGenerator:
    """
        Generate a patch-wise blur kernel
        :param patch_wise_trajectory: the trajectories used to generate blur kernels.
        :param canvas: size of domain where each blur kernel is defined.
    """

    def __init__(self, patch_wise_trajectory, canvas=16):
        self.patch_wise_trajectory = patch_wise_trajectory
        self.canvas = canvas

        self.patch_wise_kernel = None

    def generate(self):
        Patch_number_H, Patch_number_W, samples = self.patch_wise_trajectory.shape
        self.patch_wise_kernel = np.zeros((Patch_number_H * self.canvas, Patch_number_W * self.canvas),
                                          dtype=np.float32)
        for row in range(Patch_number_H):
            for col in range(Patch_number_W):
                kernel = KernelGenerator(self.patch_wise_trajectory[row, col], self.canvas).generate()
                self.patch_wise_kernel[row * self.canvas:(row + 1) * self.canvas,
                col * self.canvas:(col + 1) * self.canvas] = kernel
        return self.patch_wise_kernel

    def show_result(self):
        if self.patch_wise_kernel is None:
            raise Exception('Run self.generate() first!')
        plt.figure()
        plt.matshow(self.patch_wise_kernel)
        plt.show()
        plt.close()


def blur_image_by_kernel(image, kernel):
    # image: [0, 1+]
    blur_image = cv2.filter2D(image, cv2.CV_32F, np.flip(kernel), borderType=cv2.BORDER_REFLECT)

    return blur_image


def blur_image_by_patch_wise_kernel(image, patch_wise_kernel, patch_size):
    # image: [0, 1+]
    blur_image = np.zeros_like(image)
    Patch_number_H, Patch_number_W = patch_wise_kernel.shape[0] // patch_size, patch_wise_kernel.shape[1] // patch_size
    for row in range(Patch_number_H):
        for col in range(Patch_number_W):
            kernel = patch_wise_kernel[row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size]
            blur_patch = cv2.filter2D(image, cv2.CV_32F, np.flip(kernel), borderType=cv2.BORDER_REFLECT)[
                         row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size]
            blur_image[row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size] = blur_patch

    return blur_image


if __name__ == '__main__':
    trajectory = np.load('trajectory.npy')
    kernel_generator = KernelGenerator(trajectory)
    kernel = kernel_generator.generate()
    kernel_generator.show_result()
    np.save('kernel.npy', kernel)

    patch_wise_trajectory = np.load('patch_wise_trajectory.npy')
    patch_wise_kernel_generator = PatchWiseKernelGenerator(patch_wise_trajectory)
    patch_wise_kernel = patch_wise_kernel_generator.generate()
    patch_wise_kernel_generator.show_result()
    np.save('patch_wise_kernel.npy', patch_wise_kernel)

    image = cv2.imread('img.png', -1)
    image = cv2.resize(image, (320, 256), interpolation=cv2.INTER_CUBIC)
    image = np.float32(image) / 255.
    blur_image_uniform = blur_image_by_kernel(image, kernel)
    blur_image_spatially_variant = blur_image_by_patch_wise_kernel(image, patch_wise_kernel, kernel.shape[0])
    cv2.imwrite('blur_image_uniform.png', blur_image_uniform * 255)
    cv2.imwrite('blur_image_spatially_variant.png', blur_image_spatially_variant * 255)
