import os
import fnmatch
import argparse

import numpy as np
import cv2
from tqdm import tqdm

from scripts.script_utils.CameraMotion import TrajectoryGenerator, PatchWiseTrajectoryGenerator
from scripts.script_utils.Events import DVSModelSimulator, stack_events_to_voxel_grid
from scripts.script_utils.BlurKernel import KernelGenerator, PatchWiseKernelGenerator
from scripts.script_utils.LowLightNoise import LowLightNoise


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_img(path, rgb=True):
    img = cv2.imread(path, -1)
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(img) / 255.
    return img


def write_img(path, img, rgb=True):
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img * 255
    cv2.imwrite(path, img)


def augment_data(RGB, name, flag=None):
    RGB = RGB + np.float32(RGB == 1) * np.random.uniform(1.25, 2.5)  # increase dynamic range
    if flag is None:
        flag = np.random.randint(0, 4)
    if flag == 0:
        return RGB, name
    elif flag == 1:
        return np.flip(RGB, axis=0), name + '_flipud'
    elif flag == 2:
        return np.flip(RGB, axis=1), name + '_fliplr'
    elif flag == 3:
        return np.flip(RGB, axis=(0, 1)), name + '_flip'
    else:
        raise Exception('??????????')


class Processor:
    """for pre-processing the data source (with data augmentation)"""

    def __init__(self, RGB_dir, base_trajectory_args, patch_wise_trajectory_args, trajectory_num=20,
                 start_and_chunk=None, mode='train'):
        self.RGB_dir = RGB_dir

        self.base_trajectory_args = base_trajectory_args
        self.patch_wise_trajectory_args = patch_wise_trajectory_args
        self.trajectory_num = trajectory_num

        self.names = [file_name[:-4] for file_name in sorted(fnmatch.filter(os.listdir(self.RGB_dir), '*.png'))]
        if start_and_chunk is not None:
            # should be a tuple (start, chunk)
            start, chunk = start_and_chunk
            self.names = self.names[start: start + chunk]
        print('Pre-processing the following files: {}'.format(self.names))

        assert mode in ('train', 'test')
        self.mode = mode

    def _generate_random_trajectory(self):
        trajectory_fail_cnt = 0
        while (base_trajectory_RGB := TrajectoryGenerator(**self.base_trajectory_args).generate()) is None:
            print('Sanity check fail, regenerate a trajectory...')
            trajectory_fail_cnt += 1
        if trajectory_fail_cnt > 0:
            print('A trajectory is successfully generated after failing {} times.'.format(trajectory_fail_cnt))

        patch_wise_trajectory_fail_cnt = 0
        while (
                patch_wise_trajectory_RGB := PatchWiseTrajectoryGenerator(base_trajectory_RGB,
                                                                          **self.patch_wise_trajectory_args).generate()
        ) is None:
            print('Sanity check fail, regenerate a patch_wise_trajectory...')
            patch_wise_trajectory_fail_cnt += 1
        if patch_wise_trajectory_fail_cnt > 0:
            print('A patch_wise_trajectory is successfully generated after failing {} times.'.format(
                patch_wise_trajectory_fail_cnt))
        return base_trajectory_RGB, patch_wise_trajectory_RGB

    def __len__(self):
        return len(self.names) * self.trajectory_num

    def __iter__(self):
        for name in self.names:
            print('Fetching {}...'.format(name))
            RGB_file_name = os.path.join(self.RGB_dir, name + '.png')
            RGB = read_img(RGB_file_name)
            for i in range(self.trajectory_num):
                print('Generating {}-th trajectory for {}...'.format(i, name))
                base_trajectory_RGB, patch_wise_trajectory_RGB = self._generate_random_trajectory()
                if self.mode == 'train':
                    flag = i % 4
                elif self.mode == 'test':
                    flag = 0
                else:
                    raise Exception('Mode must be "train" or "test" !')
                RGB_, name_ = augment_data(RGB, name, flag=flag)
                raw_data = {'RGB': RGB_,
                            'base_trajectory_RGB': base_trajectory_RGB,
                            'patch_wise_trajectory_RGB': patch_wise_trajectory_RGB,
                            }
                yield name_ + '_{:0>3d}'.format(i), raw_data


class Maker:
    """
        output for train: RGB, RGB_blur, APS, APS_blur, flow, events, events_voxel_grid
        output for test: RGB, RGB_blur, APS, APS_blur, flow, events, events_voxel_grid,
                         base_trajectory_APS, patch_wise_trajectory_APS, base_kernel_APS, patch_wise_kernel_APS
        * all images are in [0, 1+]
        * RGB and RGB_blur is in [R, G, B] manner
    """

    def __init__(self, RGB, base_trajectory_RGB, patch_wise_trajectory_RGB, downsampling=3, total_time=0.1,
                 temporal_bins=13, **dvs_model_simulator_args):
        self.RGB = RGB
        self.RGB_blur_clean = np.zeros_like(self.RGB)
        self.RGB_blur = np.zeros_like(self.RGB)
        self.RGB_H, self.RGB_W, _ = self.RGB.shape

        self.downsampling = downsampling
        self.APS_H, self.APS_W = self.RGB_H // self.downsampling, self.RGB_W // self.downsampling
        self.APS = cv2.resize(cv2.cvtColor(self.RGB, cv2.COLOR_RGB2GRAY), (self.APS_W, self.APS_H))
        self.APS_blur_clean = np.zeros_like(self.APS)
        self.APS_blur = np.zeros_like(self.APS)

        self.base_trajectory_RGB = base_trajectory_RGB  # (Samples, )
        self.base_trajectory_APS = self.base_trajectory_RGB / self.downsampling
        self.base_kernel_APS = None
        self.patch_wise_trajectory_RGB = patch_wise_trajectory_RGB  # (Patch_number_H, Patch_number_W, Samples)
        self.patch_wise_trajectory_APS = self.patch_wise_trajectory_RGB / self.downsampling
        self.patch_wise_kernel_APS = None
        self.Patch_number_H, self.Patch_number_W, self.samples = self.patch_wise_trajectory_RGB.shape
        self.patch_size_RGB = self.RGB_H // self.Patch_number_H
        self.patch_size_APS = self.patch_size_RGB // self.downsampling

        self.flow = np.zeros((self.APS_H, self.APS_W, 4), dtype=np.float32)

        self.events = None
        self.events_voxel_grid = None

        self.total_time = total_time
        self.temporal_bins = temporal_bins
        self.dvs_model_simulator = DVSModelSimulator(**dvs_model_simulator_args)

        self.low_light_noise = LowLightNoise()

    def _make(self):
        current_timestamp = 0
        dt = self.total_time / (self.samples - 1)

        total_events = []

        # initialize the DVS Model Simulator
        self.dvs_model_simulator.initialize(self.APS * 255, current_timestamp)
        self.RGB_blur_clean += self.RGB
        self.APS_blur_clean += self.APS

        flow_done = False

        for i in range(1, self.samples):
            current_frame_RGB = np.zeros_like(self.RGB)
            current_frame_APS = np.zeros_like(self.APS)
            current_timestamp += dt

            for row in range(self.Patch_number_H):
                for col in range(self.Patch_number_W):
                    trajectory_RGB = self.patch_wise_trajectory_RGB[row, col]
                    d_RGB = trajectory_RGB[i] - trajectory_RGB[0]
                    # convert complex plane into image coordinates
                    d_RGBx = d_RGB.real  # axis-x keeps unchanged
                    d_RGBy = -d_RGB.imag  # axis-y is flipped
                    M_RGB = np.array([[1, 0, d_RGBx], [0, 1, d_RGBy]])
                    current_frame_patch_RGB = cv2.warpAffine(self.RGB, M_RGB, dsize=(self.RGB_W, self.RGB_H),
                                                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)[
                                              row * self.patch_size_RGB:(row + 1) * self.patch_size_RGB,
                                              col * self.patch_size_RGB:(col + 1) * self.patch_size_RGB]
                    current_frame_RGB[row * self.patch_size_RGB:(row + 1) * self.patch_size_RGB,
                    col * self.patch_size_RGB:(col + 1) * self.patch_size_RGB] += current_frame_patch_RGB

                    trajectory_APS = self.patch_wise_trajectory_APS[row, col]
                    d_APS = trajectory_APS[i] - trajectory_APS[0]
                    # convert complex plane into image coordinates
                    d_APSx = d_APS.real  # axis-x keeps unchanged
                    d_APSy = -d_APS.imag  # axis-y is flipped
                    M_APS = np.array([[1, 0, d_APSx], [0, 1, d_APSy]])
                    current_frame_patch_APS = cv2.warpAffine(self.APS, M_APS, dsize=(self.APS_W, self.APS_H),
                                                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)[
                                              row * self.patch_size_APS:(row + 1) * self.patch_size_APS,
                                              col * self.patch_size_APS:(col + 1) * self.patch_size_APS]
                    current_frame_APS[row * self.patch_size_APS:(row + 1) * self.patch_size_APS,
                    col * self.patch_size_APS:(col + 1) * self.patch_size_APS] += current_frame_patch_APS

                    if not flow_done:
                        start = trajectory_APS[0]
                        mid = trajectory_APS[trajectory_APS.size // 2]
                        end = trajectory_APS[-1]
                        d1 = mid - start
                        d2 = end - mid
                        self.flow[row * self.patch_size_APS: (row + 1) * self.patch_size_APS,
                        col * self.patch_size_APS: (col + 1) * self.patch_size_APS, 0:2] += np.array(
                            [d1.real, -d1.imag], dtype=np.float32)[None, None, :]
                        self.flow[row * self.patch_size_APS: (row + 1) * self.patch_size_APS,
                        col * self.patch_size_APS: (col + 1) * self.patch_size_APS, 2:4] += np.array(
                            [d2.real, -d2.imag], dtype=np.float32)[None, None, :]

            flow_done = True

            self.RGB_blur_clean += current_frame_RGB
            self.APS_blur_clean += current_frame_APS
            events = self.dvs_model_simulator.simulate(current_frame_APS * 255, current_timestamp)
            if events is not None:
                total_events.append(events)

        self.RGB_blur_clean /= self.samples
        self.RGB_blur = self.low_light_noise.add_noise(self.RGB_blur_clean, rgb=True)
        self.APS_blur_clean /= self.samples
        self.APS_blur = self.low_light_noise.add_noise(self.APS_blur_clean, rgb=False)

        self.events = np.vstack(total_events)
        self.events_voxel_grid = stack_events_to_voxel_grid(self.events, self.APS_H, self.APS_W,
                                                            temporal_bins=self.temporal_bins,
                                                            total_time=self.total_time)

    def make(self, mode):
        self._make()
        data_item = {
            'RGB': self.RGB,
            'RGB_blur_clean': self.RGB_blur_clean,
            'RGB_blur': self.RGB_blur,
            'APS': self.APS,
            'APS_blur_clean': self.APS_blur_clean,
            'APS_blur': self.APS_blur,
            'flow': self.flow,
            'events': self.events,
            'events_voxel_grid': self.events_voxel_grid,
        }
        if mode == 'train':
            return data_item
        elif mode == 'test':
            kernel_generator = KernelGenerator(self.base_trajectory_APS, canvas=self.patch_size_APS)
            self.base_kernel_APS = kernel_generator.generate()
            patch_wise_kernel_generator = PatchWiseKernelGenerator(self.patch_wise_trajectory_APS,
                                                                   canvas=self.patch_size_APS)
            self.patch_wise_kernel_APS = patch_wise_kernel_generator.generate()

            data_item.update({
                'base_trajectory_APS': self.base_trajectory_APS,
                'patch_wise_trajectory_APS': self.patch_wise_trajectory_APS,
                'base_kernel_APS': self.base_kernel_APS,
                'patch_wise_kernel_APS': self.patch_wise_kernel_APS
            })
            return data_item
        else:
            raise Exception('Mode must be "train" or "test" !')


class Saver:
    """for saving the dataset"""

    def __init__(self, data_item, out_base_dir, subdir_mapping):
        self.data_item = data_item
        self.out_base_dir = out_base_dir
        self.subdir_mapping = subdir_mapping

    def _save(self, path, data):
        saving_format = path[-4:]
        if saving_format == '.npy':
            np.save(path, data)
        elif saving_format == '.png':
            # shape must be (H, W) or (H, W, 3)
            if data.ndim == 2:
                write_img(path, data, rgb=False)
            elif data.ndim == 3:
                write_img(path, data, rgb=True)
            else:
                raise Exception('Shape must be (H, W) or (H, W, 3) for saving as .png format!')
        else:
            raise Exception('Saving as {} is not supported yet!'.format(saving_format))

    def save(self, name):
        for out_dir_name, out_subdir_names_and_saving_formats in self.subdir_mapping.items():
            for out_subdir_name, saving_format in out_subdir_names_and_saving_formats.items():
                data = self.data_item.get(out_subdir_name)
                if data is not None:
                    out_subdir = os.path.join(self.out_base_dir, out_dir_name, out_subdir_name)
                    ensure_dir(out_subdir)
                    self._save(os.path.join(out_subdir, name + saving_format), data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make_dataset')
    parser.add_argument('--start', default=None, type=int, help='start')
    parser.add_argument('--chunk', default=None, type=int, help='chunk')
    parser.add_argument('--mode', default='train', type=str, help='mode ("train" or "test")')
    args = parser.parse_args()

    start_and_chunk = None
    if args.start is not None and args.chunk is not None:
        start_and_chunk = (args.start, args.chunk)

    mode = args.mode

    processor_args = {
        'RGB_dir': 'RealBlurSourceFiles',
        'base_trajectory_args': {
            'canvas': 48,
            'samples': 25,
            'arc_len': None,
            'central_angle': None,
            'position_angle': None,
            'clock_wise': None,
            'downsampling': 3,
        },
        'patch_wise_trajectory_args': {
            'canvas': 48,
            'patch_number': (16, 20),
            'z_center': None,
            'z_translation_size': None,
            'z_rotation_angle': None,
            'z_rotation_size': None,
            'downsampling': 3,
        },
        'trajectory_num': 20,
        'start_and_chunk': start_and_chunk,
        'mode': mode,
    }

    dvs_model_simulator_args = {
        'p_th': 0.3,
        'n_th': 0.3,
        'sigma_th': 0.03,
        'cutoff_hz': 30,
        'leak_rate_hz': 0.1,
        'shot_noise_rate_hz': 0.1,
        'seed': 42
    }

    maker_args = {
        'downsampling': 3,
        'total_time': 0.1,
        'temporal_bins': 13,
        **dvs_model_simulator_args
    }

    saver_args = {
        'out_base_dir': '../data/' + mode,
        'subdir_mapping': {
            'share': {
                'APS_blur': '.png',
                'APS_blur_clean': '.png',
                'flow': '.npy',
            },
            'subnetwork1': {
                'events_voxel_grid': '.npy',
            },
            'subnetwork2': {
                'APS': '.png',
                'RGB': '.png',
                'RGB_blur': '.png',
                'RGB_blur_clean': '.png',
            },
            'others': {
                'events': '.npy',
                'base_trajectory_APS': '.npy',
                'patch_wise_trajectory_APS': '.npy',
                'base_kernel_APS': '.npy',
                'patch_wise_kernel_APS': '.npy',
            },
        }
    }

    processor = Processor(**processor_args)
    for i, name_and_raw_data in enumerate(tqdm(processor)):
        name, raw_data = name_and_raw_data
        maker = Maker(**raw_data, **maker_args)
        print('Making {}-th data_item: {}...'.format(i, name))
        data_item = maker.make(mode)
        print('Saving {}-th data_item: {}...'.format(i, name))
        saver = Saver(data_item, **saver_args)
        saver.save(name)
