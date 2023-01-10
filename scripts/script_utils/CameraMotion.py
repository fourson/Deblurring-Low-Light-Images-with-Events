import numpy as np
import cv2
import matplotlib.pyplot as plt


class TrajectoryGenerator:
    def __init__(self, canvas=16, samples=13, arc_len=None, central_angle=None, position_angle=None, clock_wise=None,
                 downsampling=1):
        """
            Generate an image translation trajectory (inverse camera translation trajectory) in a 2D plane
            :param canvas: size of domain where the trajectory is defined.
            :param samples: number of sample points of the trajectory, include the start point.
            :param arc_len: length of the trajectory (which is an arc).
            :param central_angle: the central angle of the arc
            :param position_angle: the position angle of the arc
            :param clock_wise: the direction of the arc (default is counter clockwise)
            :param downsampling: magnitude of downsampling in the future, this will influence the sanity check
                   downsampling=1 for not downsampling
        """
        self.canvas = canvas
        self.samples = samples
        self.downsampling = downsampling

        if clock_wise is None:
            self.clock_wise = np.random.randint(0, 2)
        else:
            self.clock_wise = clock_wise
        if arc_len is None:
            self.arc_len = canvas * 0.7 + canvas * np.random.uniform(-0.05, 0.05)
        else:
            self.arc_len = arc_len
        if central_angle is None:
            self.central_angle = np.random.uniform(np.pi / 6, np.pi)
        else:
            self.central_angle = central_angle
        if position_angle is None:
            self.position_angle = np.random.uniform(0, 2 * np.pi)
        else:
            self.position_angle = position_angle
        self.radius = self.arc_len / self.central_angle

        self.trajectory = np.zeros(self.samples, dtype=complex)

        self.success = False

    def generate(self):
        # compute the start point and end point
        if not self.clock_wise:
            start_point = self.radius * np.exp(complex(0, self.position_angle))
            d_angle = self.central_angle / (self.samples - 1)
        else:
            start_point = self.radius * np.exp(complex(0, self.position_angle + self.central_angle))
            d_angle = -self.central_angle / (self.samples - 1)

        self.trajectory[0] = start_point
        for i in range(1, self.samples):
            self.trajectory[i] = start_point * np.exp(complex(0, i * d_angle))

        # center the motion
        self.trajectory -= complex(np.min(self.trajectory.real), np.min(self.trajectory.imag))
        self.trajectory += complex(1, 1)
        self.trajectory -= complex(self.trajectory[0].real % 1., self.trajectory[0].imag % 1.)
        self.trajectory += complex(np.ceil((self.canvas - np.max(self.trajectory.real)) / 2),
                                   np.ceil((self.canvas - np.max(self.trajectory.imag)) / 2))

        # sanity check
        if np.max(self.trajectory.real) >= (self.canvas - self.downsampling) or np.min(
                self.trajectory.real) <= 0 or np.max(self.trajectory.imag) >= (
                self.canvas - self.downsampling) or np.min(self.trajectory.imag) <= 0:
            return None
        else:
            self.success = True
            return self.trajectory

    def obtain_bidirectional_flow(self, H, W):
        bidirectional_flow = np.zeros((H, W, 4), dtype=np.float32)
        start = self.trajectory[0]
        mid = self.trajectory[self.samples // 2]
        end = self.trajectory[-1]
        d1 = mid - start
        d2 = end - mid
        bidirectional_flow[:, :, 0:2] += np.array([d1.real, -d1.imag], dtype=np.float32)[None, None, :]
        bidirectional_flow[:, :, 2:4] += np.array([d2.real, -d2.imag], dtype=np.float32)[None, None, :]
        return bidirectional_flow

    def show_result(self):
        if not self.success:
            raise Exception('Sanity check failed!')
        plt.figure()
        plt.plot(self.trajectory.real, self.trajectory.imag, '-', color='blue')
        plt.xlim((0, self.canvas))
        plt.ylim((0, self.canvas))
        plt.show()
        plt.close()


# class OriginalTrajectoryGenerator:
#     """
#         Generate an image translation trajectory (inverse camera translation trajectory) in a 2D plane
#         :param canvas: size of domain where the trajectory is defined.
#         :param samples: number of sample points of the trajectory, include the start point.
#         :param expl: this param helps to define probability of big shake. expl = 0 for linear motion.
#         :param max_len: maximum length of the trajectory.
#         :param downsampling: magnitude of downsampling in the future, this will influence the sanity check
#                downsampling=1 for not downsampling
#     """
#
#     def __init__(self, canvas=32, samples=13, expl=0.075, big_expl_count_max=None, max_len=None, downsampling=1):
#         self.canvas = canvas
#         self.samples = samples
#         self.downsampling = downsampling
#
#         if big_expl_count_max is None:
#             self.big_expl_count_max = np.inf
#         else:
#             self.big_expl_count_max = big_expl_count_max
#             print(self.big_expl_count_max)
#
#         if expl is None:
#             self.expl = np.random.choice([0.01, 0.009, 0.008, 0.007, 0.005, 0.003])
#         else:
#             self.expl = expl
#
#         if max_len is None:
#             self.max_len = canvas * 0.7 + canvas * np.random.uniform(-0.05, 0.05)
#         else:
#             self.max_len = max_len
#
#         self.total_length = 0
#         self.big_expl_count = 0
#         self.trajectory = np.zeros(self.samples, dtype=complex)
#
#         self.success = False
#
#     def generate(self):
#         seg_len = self.max_len / (self.samples - 1)
#         # how to be near the previous position
#         centripetal = 0.7 * np.random.uniform(0, 1)
#         # probability of big shake
#         prob_big_shake = 0.2 * np.random.uniform(0, 1)
#         # term determining, at each sample, the random component of the new direction
#         gaussian_shake = 10 * np.random.uniform(0, 1)
#         init_angle = 2 * np.pi * np.random.uniform(0, 1)
#
#         v0 = np.e ** (1j * init_angle)
#         v = v0 * seg_len
#
#         if self.expl > 0:
#             v = v0 * self.expl
#
#         for t in range(1, self.samples):
#             if np.random.uniform() < prob_big_shake * self.expl and self.big_expl_count <= self.big_expl_count_max:
#                 next_direction = 2 * v * (np.exp(complex(0, np.pi + (np.random.uniform() - 0.5))))
#                 self.big_expl_count += 1
#             else:
#                 next_direction = 0
#
#             dv = next_direction + self.expl * (
#                     gaussian_shake * complex(np.random.randn(), np.random.randn()) - centripetal * self.trajectory[
#                 t - 1]
#             ) * seg_len
#
#             v += dv
#             v = (v / np.abs(v)) * seg_len
#             self.trajectory[t] = self.trajectory[t - 1] + v
#             self.total_length += np.abs(self.trajectory[t] - self.trajectory[t - 1])
#
#         # center the motion
#         self.trajectory -= complex(np.min(self.trajectory.real), np.min(self.trajectory.imag))
#         self.trajectory += complex(1, 1)
#         self.trajectory -= complex(self.trajectory[0].real % 1., self.trajectory[0].imag % 1.)
#         self.trajectory += complex(np.ceil((self.canvas - np.max(self.trajectory.real)) / 2),
#                                    np.ceil((self.canvas - np.max(self.trajectory.imag)) / 2))
#
#         # sanity check
#         if np.max(self.trajectory.real) >= (self.canvas - self.downsampling) or np.min(
#                 self.trajectory.real) <= 0 or np.max(self.trajectory.imag) >= (
#                 self.canvas - self.downsampling) or np.min(self.trajectory.imag) <= 0:
#             return None
#         else:
#             self.success = True
#             return self.trajectory
#
#     def show_result(self):
#         if not self.success:
#             raise Exception('Sanity check failed!')
#         plt.figure()
#         plt.plot(self.trajectory.real, self.trajectory.imag, '-', color='blue')
#         plt.xlim((0, self.canvas))
#         plt.ylim((0, self.canvas))
#         plt.show()
#         plt.close()


class PatchWiseTrajectoryGenerator:
    """
        Generate multiple image translation trajectories in a 2D plane (in a patch wise manner)
        :param base_trajectory: base trajectory used for generating other trajectories.
        :param canvas: size of domain where each trajectory is defined.
        :param patch_number: a tuple (Patch_number_H, Patch_number_W) denoting the number of patches in a 2D plane.
        :param z_center: a tuple (z_center_row, z_center_col) denoting the center patch where z-axis is located.
        :param z_translation_size: magnitude of translation vector along z-axis.
        :param z_rotation_angle: angle of rotation vector along z-axis.
        :param z_rotation_size: size of rotation vector along z-axis.
        :param downsampling: magnitude of downsampling in the future, this will influence the sanity check
               downsampling=1 for not downsampling
    """

    def __init__(self, base_trajectory, canvas=16, patch_number=(16, 20), z_center=None, z_translation_size=None,
                 z_rotation_angle=None, z_rotation_size=None, downsampling=1):
        self.base_trajectory = base_trajectory
        self.canvas = canvas
        self.Patch_number_H, self.Patch_number_W = patch_number
        self.downsampling = downsampling
        self.m = self.canvas / len(self.base_trajectory)

        if z_center is None:
            self.z_center_row = np.random.randint(0, self.Patch_number_H)
            self.z_center_col = np.random.randint(0, self.Patch_number_W)
        else:
            self.z_center_row, self.z_center_col = z_center

        if z_translation_size is None:
            # self.z_translation_size = np.random.uniform(-self.m / 15, self.m / 15)
            self.z_translation_size = np.random.uniform(-self.m / 35, self.m / 35)
        else:
            self.z_translation_size = z_translation_size

        if z_rotation_angle is None:
            self.z_rotation_angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        else:
            self.z_rotation_angle = z_rotation_angle

        if z_rotation_size is None:
            # self.z_rotation_size = np.random.uniform(0, self.m / 10)
            self.z_rotation_size = np.random.uniform(0, self.m / 25)
        else:
            self.z_rotation_size = z_rotation_size

        self.patch_wise_trajectory = np.zeros((self.Patch_number_H, self.Patch_number_W, len(self.base_trajectory)),
                                              dtype=complex)

        self.success = False

    def generate(self):
        base_motion = self.base_trajectory[1:] - self.base_trajectory[:-1]
        ind_row, ind_col = np.meshgrid(np.arange(self.Patch_number_H), np.arange(self.Patch_number_W), indexing='ij')
        distance = np.sqrt((ind_row - self.z_center_row) ** 2 + (ind_col - self.z_center_col) ** 2)
        direction = np.arctan2(-(ind_row - self.z_center_row), ind_col - self.z_center_col)
        for row in range(self.Patch_number_H):
            for col in range(self.Patch_number_W):
                # for z_translation: camera forward, image outward
                direction_z_translation = direction[row, col]
                z_translation = self.z_translation_size * distance[row, col] * np.e ** (1j * direction_z_translation)
                # for z_rotation: camera clockwise, image counterclockwise
                direction_z_rotation = direction[row, col] + np.sign(self.z_rotation_angle) * np.pi / 2
                z_rotation = self.z_rotation_size * np.tan(self.z_rotation_angle) * distance[row, col] * np.e ** (
                        1j * direction_z_rotation)
                motion = base_motion + z_translation + z_rotation
                trajectory = np.copy(self.base_trajectory)
                for i in range(1, len(trajectory)):
                    trajectory[i] = trajectory[i - 1] + motion[i - 1]
                # sanity check
                if np.max(trajectory.real) >= (self.canvas - self.downsampling) or np.min(
                        trajectory.real) <= 0 or np.max(trajectory.imag) >= (self.canvas - self.downsampling) or np.min(
                    trajectory.imag) <= 0:
                    return None
                else:
                    self.patch_wise_trajectory[row, col] = trajectory
        self.success = True
        return self.patch_wise_trajectory

    def obtain_bidirectional_flow(self, H, W):
        bidirectional_flow = np.zeros((H, W, 4), dtype=np.float32)
        patch_size_H = H // self.Patch_number_H
        patch_size_W = W // self.Patch_number_W
        for row in range(self.Patch_number_H):
            for col in range(self.Patch_number_W):
                trajectory = self.patch_wise_trajectory[row, col]
                start = trajectory[0]
                mid = trajectory[trajectory.size // 2]
                end = trajectory[-1]
                d1 = mid - start
                d2 = end - mid
                bidirectional_flow[row * patch_size_H: (row + 1) * patch_size_H,
                col * patch_size_W: (col + 1) * patch_size_W, 0:2] += np.array([d1.real, -d1.imag], dtype=np.float32)[
                                                                      None, None, :]
                bidirectional_flow[row * patch_size_H: (row + 1) * patch_size_H,
                col * patch_size_W: (col + 1) * patch_size_W, 2:4] += np.array([d2.real, -d2.imag], dtype=np.float32)[
                                                                      None, None, :]
        return bidirectional_flow

    def show_result(self):
        if not self.success:
            raise Exception('Sanity check failed!')
        fig, axs = plt.subplots(self.Patch_number_H, self.Patch_number_W)
        for row in range(self.Patch_number_H):
            for col in range(self.Patch_number_W):
                trajectory = self.patch_wise_trajectory[row, col]
                axs[row, col].plot(trajectory.real, trajectory.imag, '-', color='blue')
                axs[row, col].set_xlim((0, self.canvas))
                axs[row, col].set_ylim((0, self.canvas))
        plt.show()
        plt.close()


def blur_image_by_trajectory(image, trajectory):
    # image: [0, 1+]
    H, W = image.shape[0], image.shape[1]
    samples = len(trajectory)
    blur_image = np.copy(image)
    for i in range(1, samples):
        d = trajectory[i] - trajectory[0]
        # convert complex plane into image coordinates
        dx = d.real  # axis-x keeps unchanged
        dy = -d.imag  # axis-y is flipped
        M = np.array([[1, 0, dx], [0, 1, dy]])
        current_frame = cv2.warpAffine(image, M, dsize=(W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        blur_image += current_frame
    blur_image /= samples

    return blur_image


def blur_image_by_patch_wise_trajectory(image, patch_wise_trajectory, patch_size):
    # image: [0, 1+]
    H, W = image.shape[0], image.shape[1]
    Patch_number_H, Patch_number_W, samples = patch_wise_trajectory.shape
    blur_image = np.copy(image)
    for i in range(1, samples):
        current_frame = np.zeros_like(image)
        for row in range(Patch_number_H):
            for col in range(Patch_number_W):
                trajectory = patch_wise_trajectory[row, col]
                d = trajectory[i] - trajectory[0]
                # convert complex plane into image coordinates
                dx = d.real  # axis-x keeps unchanged
                dy = -d.imag  # axis-y is flipped
                M = np.array([[1, 0, dx], [0, 1, dy]])
                current_frame_patch = cv2.warpAffine(image, M, dsize=(W, H), flags=cv2.INTER_LINEAR,
                                                     borderMode=cv2.BORDER_REFLECT)[
                                      row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size]
                current_frame[row * patch_size:(row + 1) * patch_size,
                col * patch_size:(col + 1) * patch_size] = current_frame_patch
        blur_image += current_frame
    blur_image /= samples

    return blur_image


if __name__ == '__main__':
    base_trajectory_args = {
        'canvas': 16,
        'samples': 13,
        'arc_len': None,
        'central_angle': None,
        'position_angle': None,
        'clock_wise': None,
        'downsampling': 1,
    }

    patch_wise_trajectory_args = {
        'canvas': 16,
        'patch_number': (16, 20),
        'z_center': None,
        'z_translation_size': None,
        'z_rotation_angle': None,
        'z_rotation_size': None,
        'downsampling': 1,
    }

    trajectory_generator = TrajectoryGenerator(**base_trajectory_args)
    trajectory = trajectory_generator.generate()
    success = trajectory_generator.success
    while not success:
        print('sanity check fail, regenerate a trajectory')
        trajectory_generator = TrajectoryGenerator()
        trajectory = trajectory_generator.generate()
        success = trajectory_generator.success
    trajectory_generator.show_result()
    bidirectional_flow = trajectory_generator.obtain_bidirectional_flow(256, 320)
    np.save('bidirectional_flow.npy', bidirectional_flow)

    patch_wise_trajectory_generator = PatchWiseTrajectoryGenerator(trajectory, **patch_wise_trajectory_args)
    patch_wise_trajectory = patch_wise_trajectory_generator.generate()
    success = patch_wise_trajectory_generator.success
    while not success:
        print('sanity check fail, regenerate a patch_wise_trajectory')
        patch_wise_trajectory_generator = PatchWiseTrajectoryGenerator(trajectory, **patch_wise_trajectory_args)
        patch_wise_trajectory = patch_wise_trajectory_generator.generate()
        success = patch_wise_trajectory_generator.success
    patch_wise_trajectory_generator.show_result()
    patch_wise_bidirectional_flow = patch_wise_trajectory_generator.obtain_bidirectional_flow(256, 320)
    np.save('patch_wise_bidirectional_flow.npy', patch_wise_bidirectional_flow)

    np.save('trajectory.npy', trajectory)
    np.save('patch_wise_trajectory.npy', patch_wise_trajectory)
