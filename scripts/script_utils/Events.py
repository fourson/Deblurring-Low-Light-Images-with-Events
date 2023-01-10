import numpy as np
import cv2


def lin_log(x, th=20):
    """
        lin-log mapping
    """
    if x.dtype is not np.float32:
        x = x.astype(np.float32)
    return np.piecewise(x, [x < th, x >= th], [lambda a: a / th * np.log(th), lambda a: np.log(a)])


class DVSModelSimulator:
    """
        DVS model simulator
        :param p_th: nominal threshold of triggering positive (ON) event in log intensity
        :param n_th: nominal threshold of triggering negative (OFF) event in log intensity.
        :param sigma_th: std deviation of the threshold in log intensity.
        :param cutoff_hz: 3dB cutoff frequency in Hz of DVS photoreceptor (referring to f_{3dBmax} in Equation 4).
        :param leak_rate_hz: leak event rate per pixel in Hz, from junction leakage in reset switch.
        :param shot_noise_rate_hz: shot noise rate in Hz.
        :param seed: seed for random threshold variations, fix it to nonzero value to get same mismatch every time.
    """

    def __init__(self, p_th=0.3, n_th=0.3, sigma_th=0.03, cutoff_hz=30, leak_rate_hz=0.1, shot_noise_rate_hz=0.1,
                 seed=42):
        self.L_old = None  # memorized lin_log pixel values
        self.t_old = None  # timestamp of the memorized frame

        # "Finite intensity-dependent photoreceptor bandwidth", in Section 3
        self.L_1 = None  # L_1: an internal value representing the first stage of the filter
        self.L_lp = None  # L_lp: the lowpass-filtered brightness output

        # nominal threshold
        self.p_th = p_th
        self.n_th = n_th

        # actual threshold
        self.positive_threshold = None
        self.negative_threshold = None

        # std deviation of the threshold in log intensity
        self.sigma_th = sigma_th

        # noise settings
        self.cutoff_hz = cutoff_hz
        self.leak_rate_hz = leak_rate_hz
        self.shot_noise_rate_hz = shot_noise_rate_hz

        # random seed
        if seed > 0:
            np.random.seed(seed)

        # number of events
        self.on_events_num = 0
        self.off_events_num = 0
        self.total_events_num = 0

    def initialize(self, I_old, t_old):
        # I_old should be in [0, 255+]
        I_old = np.array(I_old, np.float32)  # convert to float

        # initialize the memorized lin_log pixel values
        # "Linear to logarithmic mapping", in Section 3
        # I_old: memorized linear pixel values
        # L_old: memorized lin_log pixel values
        self.L_old = lin_log(I_old)

        if self.leak_rate_hz > 0:
            # "Leak noise events", in Section 3
            self.L_old -= np.random.uniform(0, self.p_th, I_old.shape)  # initialize the leak noise

        # initialize both L_1 and L_lp to the same as the input for the first frame
        self.L_1 = np.copy(self.L_old)
        self.L_lp = np.copy(self.L_old)

        # initialize the actual threshold
        self.positive_threshold = np.random.normal(self.p_th, self.sigma_th, I_old.shape).astype(np.float32)
        self.negative_threshold = np.random.normal(self.n_th, self.sigma_th, I_old.shape).astype(np.float32)
        # "Hot pixels", in Section 3
        # limit the minimum threshold to 0.01 to prevent too many hot pixel events
        self.positive_threshold = np.clip(self.positive_threshold, a_min=0.01, a_max=None)
        self.negative_threshold = np.clip(self.negative_threshold, a_min=0.01, a_max=None)

        # initialize the timestamp of the memorized frame
        self.t_old = t_old

    def simulate(self, I_new, t_new):
        # I_new should be in [0, 255+]
        # check whether L_old and t_old are initialized
        if self.L_old is None or self.t_old is None:
            raise AttributeError('run initialize first!')

        # check the timestamp
        if t_new <= self.t_old:
            raise ValueError("this frame time={} must be later than previous frame time={}".format(t_new, self.t_old))

        I_new = np.array(I_new, np.float32)  # convert to float

        L_new = lin_log(I_new)
        Delta_t = t_new - self.t_old
        luma_factor = (I_new + 20) / 275

        # update L_1 and L_lp as Equation 8 and 9
        if self.cutoff_hz > 0:
            f_3dB = luma_factor * self.cutoff_hz
            tau = 1 / (2 * np.pi * f_3dB)
            eps = np.clip(Delta_t / tau, a_min=None, a_max=1)
        else:
            eps = 1
        self.L_1 = (1 - eps) * self.L_1 + eps * L_new
        self.L_lp = (1 - eps) * self.L_lp + eps * self.L_1

        # update L_old as Equation 13 and 14
        # 注意，code和原paper的公式不一致
        if self.leak_rate_hz > 0:
            delta_leak = Delta_t * self.leak_rate_hz * self.p_th
            self.L_old -= delta_leak

        Delta_L = self.L_lp - self.L_old  # log intensity change
        # for ON events
        positive_frame = Delta_L * (Delta_L > 0)  # only store the absolute value of positive log intensity change
        on_events_frame = positive_frame // self.positive_threshold  # calculate the number of ON events
        on_iters = int(np.max(on_events_frame))
        # for OFF events
        negative_frame = -Delta_L * (Delta_L < 0)  # only store the absolute value of negative log intensity change
        off_events_frame = negative_frame // self.negative_threshold  # calculate the number of OFF events
        off_iters = int(np.max(off_events_frame))
        # the number of iterations
        num_iters = max(on_iters, off_iters)

        # update L_old as Equation 11
        if on_iters > 0:
            on_events_bool = on_events_frame > 0
            self.L_old[on_events_bool] += on_events_frame[on_events_bool] * self.positive_threshold[on_events_bool]
        if off_iters > 0:
            off_events_bool = off_events_frame > 0
            self.L_old[off_events_bool] -= off_events_frame[off_events_bool] * self.negative_threshold[off_events_bool]

        events = []  # store the events

        # DVS event generation
        for i in range(num_iters):
            t = self.t_old + Delta_t * (i + 1) / num_iters  # timestamp of current iteration

            # calculate the coordinate of events
            # np.nonzero(bool_array) returns a tuple which store the coordinate per axis
            # eg:
            # k = np.array([[False, False,  True],
            #               [ True,  True,  True]])
            # np.nonzero(k)
            # Out: (array([0, 1, 1, 1], dtype=int64), array([2, 0, 1, 2], dtype=int64))
            on_events_coordinate = np.nonzero(on_events_frame >= (i + 1))
            on_events_num = on_events_coordinate[0].size
            off_events_coordinate = np.nonzero(off_events_frame >= (i + 1))
            off_events_num = off_events_coordinate[0].size
            events_num = on_events_num + off_events_num
            self.on_events_num += on_events_num
            self.off_events_num += off_events_num
            self.total_events_num += events_num

            # organize the events
            # (N, 4) array, each row contains [timestamp, column coordinate, row coordinate, sign of event]
            if on_events_num > 0:
                on_events = np.hstack(
                    (
                        np.ones((on_events_num, 1), dtype=np.float32) * t,  # timestamp
                        on_events_coordinate[1][..., None].astype(np.float32),  # column
                        on_events_coordinate[0][..., None].astype(np.float32),  # row
                        np.ones((on_events_num, 1), dtype=np.float32)  # sign
                    )
                )
            else:
                on_events = np.zeros((0, 4), dtype=np.float32)

            if off_events_num > 0:
                off_events = np.hstack(
                    (
                        np.ones((off_events_num, 1), dtype=np.float32) * t,  # timestamp
                        off_events_coordinate[1][..., None].astype(np.float32),  # column
                        off_events_coordinate[0][..., None].astype(np.float32),  # row
                        -np.ones((off_events_num, 1), dtype=np.float32)  # sign
                    )
                )
            else:
                off_events = np.zeros((0, 4), dtype=np.float32)

            # store the events of current iteration
            # initialize it with the events generated by log intensity change
            events_temp = np.vstack((on_events, off_events))

            # "Temporal noise", in Section 3
            if events_num > 0 and self.shot_noise_rate_hz > 0:
                # add shot noise events as Equation 15~19
                # 注意，code和原paper的公式不一致
                F = 0.25  # F: shot noise intensity factor
                delta_t = Delta_t / num_iters  # timestep
                R_n = self.shot_noise_rate_hz / 2  # observed noise rate
                r = R_n * ((F - 1) * luma_factor + 1)  # modified rate
                p = r * delta_t * self.p_th / self.positive_threshold  # probability for each pixel
                u = np.random.uniform(size=self.L_old.shape).astype(
                    np.float32)  # uniformly distributed sample in [0, 1)

                shot_noise_on_events_bool = u > (1 - p)
                shot_noise_on_events_coordinate = np.nonzero(shot_noise_on_events_bool)
                shot_noise_on_events_num = shot_noise_on_events_coordinate[0].size
                shot_noise_off_events_bool = u < p
                shot_noise_off_events_coordinate = np.nonzero(shot_noise_off_events_bool)
                shot_noise_off_events_num = shot_noise_off_events_coordinate[0].size
                shot_noise_events_num = shot_noise_on_events_num + shot_noise_off_events_num
                self.on_events_num += shot_noise_on_events_num
                self.off_events_num += shot_noise_off_events_num
                self.total_events_num += shot_noise_events_num

                # organize the shot noise events
                # (N, 4) array, each row contains [timestamp, column coordinate, row coordinate, sign of event]
                if shot_noise_on_events_num > 0:
                    shot_noise_on_events = np.hstack(
                        (
                            np.ones((shot_noise_on_events_num, 1), dtype=np.float32) * t,  # timestamp
                            shot_noise_on_events_coordinate[1][..., None].astype(np.float32),  # column
                            shot_noise_on_events_coordinate[0][..., None].astype(np.float32),  # row
                            np.ones((shot_noise_on_events_num, 1), dtype=np.float32)  # sign
                        )
                    )
                    # append the events generated by temporal noise
                    events_temp = np.append(events_temp, shot_noise_on_events, axis=0)
                    # update L_old as Equation 11
                    self.L_old[shot_noise_on_events_bool] += self.positive_threshold[shot_noise_on_events_bool]

                if shot_noise_off_events_num > 0:
                    shot_noise_off_events = np.hstack(
                        (
                            np.ones((shot_noise_off_events_num, 1), dtype=np.float32) * t,  # timestamp
                            shot_noise_off_events_coordinate[1][..., None].astype(np.float32),  # column
                            shot_noise_off_events_coordinate[0][..., None].astype(np.float32),  # row
                            np.ones((shot_noise_off_events_num, 1), dtype=np.float32)  # sign
                        )
                    )
                    # append the events generated by temporal noise
                    events_temp = np.append(events_temp, shot_noise_off_events, axis=0)
                    # update L_old as Equation 11
                    self.L_old[shot_noise_off_events_bool] -= self.negative_threshold[shot_noise_off_events_bool]

            # shuffle and append to events
            np.random.shuffle(events_temp)
            events.append(events_temp)

        self.t_old = t_new  # update the timestamp of the memorized frame to the timestamp of current frame

        if len(events) > 0:
            events = np.vstack(events)
            return events
        else:
            return


def generate_events_by_trajectory(image, trajectory, total_time=0.1, **dvs_model_simulator_args):
    # image: [0, 1+] grayscale
    H, W = image.shape
    samples = len(trajectory)
    dt = total_time / (samples - 1)
    dvs_model_simulator = DVSModelSimulator(**dvs_model_simulator_args)
    total_events = []
    blur_image = np.copy(image)
    dvs_model_simulator.initialize(image * 255, 0)
    for i in range(1, samples):
        d = trajectory[i] - trajectory[0]
        # convert complex plane into image coordinates
        dx = d.real  # axis-x keeps unchanged
        dy = -d.imag  # axis-y is flipped
        M = np.array([[1, 0, dx], [0, 1, dy]])
        current_frame = cv2.warpAffine(image, M, dsize=(W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        current_timestamp = i * dt
        events = dvs_model_simulator.simulate(current_frame * 255, current_timestamp)
        if events is not None:
            total_events.append(events)
        blur_image += current_frame
    total_events = np.vstack(total_events)
    blur_image /= samples

    return total_events, blur_image


def generate_events_by_patch_wise_trajectory(image, patch_wise_trajectory, patch_size, total_time=0.1,
                                             **dvs_model_simulator_args):
    # image: [0, 1+] grayscale
    H, W = image.shape
    Patch_number_H, Patch_number_W, samples = patch_wise_trajectory.shape
    dt = total_time / (samples - 1)
    dvs_model_simulator = DVSModelSimulator(**dvs_model_simulator_args)
    total_events = []
    blur_image = np.copy(image)
    dvs_model_simulator.initialize(image * 255, 0)
    for i in range(1, samples):
        current_frame = np.zeros_like(image)
        current_timestamp = i * dt
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
        events = dvs_model_simulator.simulate(current_frame * 255, current_timestamp)
        if events is not None:
            total_events.append(events)
        blur_image += current_frame
    total_events = np.vstack(total_events)
    blur_image /= samples

    return total_events, blur_image


def stack_events_to_voxel_grid(events, H, W, temporal_bins=13, total_time=0.1):
    voxel_grid = np.zeros((H, W, temporal_bins), dtype=np.float32)
    dt = total_time / (temporal_bins - 1)
    channel_max_index = temporal_bins - 1
    for event in events:
        timestamp, column, row, sign = event
        row = int(row)
        column = int(column)
        prev_channel = int(timestamp // dt)
        next_channel = min(prev_channel + 1, channel_max_index)
        next_weight = (timestamp % dt) / dt
        prev_weight = 1 - next_weight
        voxel_grid[row, column, prev_channel] += sign * prev_weight
        voxel_grid[row, column, next_channel] += sign * next_weight
    return voxel_grid


def visualize_voxel_grid(voxel_grid):
    single_frame = np.sum(voxel_grid, axis=2)
    H, W = single_frame.shape
    visual_image = np.zeros((H, W, 3), dtype=np.uint8)

    positive = np.expand_dims(np.uint8((single_frame > 0)), axis=2)
    no_polarity = np.expand_dims(np.uint8((single_frame == 0)), axis=2)
    negative = np.expand_dims(np.uint8((single_frame < 0)), axis=2)

    visual_image += positive * np.array([255, 0, 0], dtype=np.uint8) + no_polarity * np.array(
        [255, 255, 255], dtype=np.uint8) + negative * np.array([0, 0, 255], dtype=np.uint8)

    return visual_image


if __name__ == '__main__':
    image = cv2.imread('img.png', -1)  # grayscale img (H, W)
    image = cv2.resize(image, (320, 256), interpolation=cv2.INTER_CUBIC)
    image = np.float32(image) / 255.

    trajectory = np.load('trajectory.npy')
    ev_uniform, blur_image_uniform_by_avg = generate_events_by_trajectory(image, trajectory)
    np.save('ev_uniform.npy', ev_uniform)
    cv2.imwrite('blur_image_uniform_by_avg.png', blur_image_uniform_by_avg * 255)

    patch_wise_trajectory = np.load('patch_wise_trajectory.npy')
    ev_spatially_variant, blur_image_spatially_variant_by_avg = generate_events_by_patch_wise_trajectory(image,
                                                                                                         patch_wise_trajectory,
                                                                                                         32)
    np.save('ev_spatially_variant.npy', ev_spatially_variant)
    cv2.imwrite('blur_image_spatially_variant_by_avg.png', blur_image_spatially_variant_by_avg * 255)
