from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.utils.data as data
import numpy as np
import os
import h5py
import subprocess
import shlex

BASE_DIR = os.path.dirname(os.path.abspath(__file__))



def load_patch_data(h5_filename, skip_rate=1, use_randominput=True, norm=False):
    if use_randominput:
        print("use randominput, input h5 file is:", h5_filename)
        f = h5py.File(h5_filename)
        input = f['poisson_4096'][:]
        gt = f['poisson_4096'][:]
    else:
        print("Do not use randominput, input h5 file is:", h5_filename)
        f = h5py.File(h5_filename)
        gt = f['poisson_4096'][:]
        input = f['montecarlo_1024'][:]

    name = [str(item) for item in f['name'][:]]
    assert len(input) == len(gt)

    if norm:
        print("Normalize the data")
        data_radius = np.ones(shape=(len(input)))
        centroid = np.mean(gt[:, :, 0:3], axis=1, keepdims=True)
        gt[:, :, 0:3] = gt[:, :, 0:3] - centroid
        furthest_distance = np.amax(np.sqrt(np.sum(gt[:, :, 0:3] ** 2, axis=-1)), axis=1, keepdims=True)
        gt[:, :, 0:3] = gt[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
        input[:, :, 0:3] = input[:, :, 0:3] - centroid
        input[:, :, 0:3] = input[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
    else:
        print("Do not normalize the data")
        centroid = np.mean(gt[:, :, 0:3], axis=1, keepdims=True)
        furthest_distance = np.amax(np.sqrt(np.sum((gt[:, :, 0:3] - centroid) ** 2, axis=-1)), axis=1, keepdims=True)
        data_radius = furthest_distance[0, :]

    input = input[::skip_rate]
    gt = gt[::skip_rate]
    data_radius = data_radius[::skip_rate]
    name = name[::skip_rate]

    object_name = list(set([item.split('/')[-1].split('_')[0] for item in name]))
    object_name.sort()
    print("load object names {}".format(object_name))
    print("total %d samples" % (len(input)))

    return input, gt, data_radius, name


def load_refine_data(h5_filename, skip_rate=1, use_randominput=True, norm=False):
    if use_randominput:
        print("use randominput, input h5 file is:", h5_filename)
        f = h5py.File(h5_filename)
        input = f['input'][:]
        gt = f['gt'][:]
    else:
        print("Do not use randominput, input h5 file is:", h5_filename)
        f = h5py.File(h5_filename)
        gt = f['poisson_4096'][:]
        input = f['montecarlo_1024'][:]
    assert len(input) == len(gt)

    if norm:
        print("Normalize the data")
        data_radius = np.ones(shape=(len(input)))
        centroid = np.mean(gt[:, :, 0:3], axis=1, keepdims=True)
        gt[:, :, 0:3] = gt[:, :, 0:3] - centroid
        furthest_distance = np.amax(np.sqrt(np.sum(gt[:, :, 0:3] ** 2, axis=-1)), axis=1, keepdims=True)
        gt[:, :, 0:3] = gt[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
        input[:, :, 0:3] = input[:, :, 0:3] - centroid
        input[:, :, 0:3] = input[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
    else:
        print("Do not normalize the data")
        centroid = np.mean(gt[:, :, 0:3], axis=1, keepdims=True)
        furthest_distance = np.amax(np.sqrt(np.sum((gt[:, :, 0:3] - centroid) ** 2, axis=-1)), axis=1, keepdims=True)
        data_radius = furthest_distance[0, :]

    input = input[::skip_rate]
    gt = gt[::skip_rate]
    data_radius = data_radius[::skip_rate]
    print("total %d samples" % (len(input)))

    return input, gt, data_radius


def rotate_point_cloud_and_gt(batch_data, batch_gt=None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    for k in range(batch_data.shape[0]):
        angles = np.random.uniform(size=3) * 2 * np.pi
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))

        batch_data[k, ..., 0:3] = np.dot(batch_data[k, ..., 0:3].reshape((-1, 3)), rotation_matrix)
        if batch_data.shape[-1] > 3:
            batch_data[k, ..., 3:] = np.dot(batch_data[k, ..., 3:].reshape((-1, 3)), rotation_matrix)

        if batch_gt is not None:
            batch_gt[k, ..., 0:3] = np.dot(batch_gt[k, ..., 0:3].reshape((-1, 3)), rotation_matrix)
            if batch_gt.shape[-1] > 3:
                batch_gt[k, ..., 3:] = np.dot(batch_gt[k, ..., 3:].reshape((-1, 3)), rotation_matrix)

    return batch_data, batch_gt


def shift_point_cloud_and_gt(batch_data, batch_gt=None, shift_range=0.3):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, 0:3] += shifts[batch_index, 0:3]

    if batch_gt is not None:
        for batch_index in range(B):
            batch_gt[batch_index, :, 0:3] += shifts[batch_index, 0:3]

    return batch_data, batch_gt


def random_scale_point_cloud_and_gt(batch_data, batch_gt=None, scale_low=0.5, scale_high=2.0):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, 0:3] *= scales[batch_index]

    if batch_gt is not None:
        for batch_index in range(B):
            batch_gt[batch_index, :, 0:3] *= scales[batch_index]

    return batch_data, batch_gt, scales


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.03, angle_clip=0.09):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        batch_data[k, ..., 0:3] = np.dot(batch_data[k, ..., 0:3].reshape((-1, 3)), R)
        if batch_data.shape[-1] > 3:
            batch_data[k, ..., 3:] = np.dot(batch_data[k, ..., 3:].reshape((-1, 3)), R)

    return batch_data


def jitter_perturbation_point_cloud(batch_data, sigma=0.005, clip=0.02):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data[:, :, 3:] = 0
    jittered_data += batch_data
    return jittered_data

def nonuniform_sampling_idx(num=4096, sample_num=1024):
    sample = set()
    mask = np.zeros(num)
    loc = np.random.rand() * 0.8 + 0.1
    while len(sample) < sample_num:
        a = int(np.random.normal(loc=loc, scale=0.3) * num)
        if a < 0 or a >= num:
            continue
        sample.add(a)
        mask[a]=1
    return mask!=0


def replace_point_cloud(batch_data, sigma=0.005, clip=0.02):
    """ Randomly replace points. is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data[:, :, 3:] = 0
    points_to_be_replaced = nonuniform_sampling_idx(N,int(N/10))
    points_to_replace_with = nonuniform_sampling_idx(N, int(N / 10))
    for i in range(B):
        batch_data[i, points_to_be_replaced] = batch_data[i][points_to_replace_with]
    return batch_data+jittered_data


def save_pl(path, pc):
    if not os.path.exists(os.path.split(path)[0]):
        os.makedirs(os.path.split(path)[0])
    np.savetxt(path, pc)


def nonuniform_sampling(num=4096, sample_num=1024):
    sample = set()
    loc = np.random.rand() * 0.8 + 0.1
    while len(sample) < sample_num:
        a = int(np.random.normal(loc=loc, scale=0.3) * num)
        if a < 0 or a >= num:
            continue
        sample.add(a)
    return list(sample)


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


class LoaderDataset(data.Dataset):
    def __init__(self, dataset, batch_size, use_norm, gpu_ids, transforms=None, train=True, refinenet=False):
        super().__init__()
        self.dataset = dataset
        self.refinenet = refinenet

        if not refinenet:
            self.input_data, self.gt_data, self.radius_data, _ = load_patch_data(self.dataset, skip_rate=1, norm=use_norm)
        else:
            self.input_data, self.gt_data, self.radius_data = load_refine_data(self.dataset, skip_rate=1, norm=use_norm)

        self.batch_size = batch_size
        self.use_norm = use_norm
        self.sample_cnt = self.input_data.shape[0]
        self.num_batches = self.sample_cnt // self.batch_size
        self.gpu_ids = gpu_ids
        self.multi_gpus = len(gpu_ids) > 1
        self.scale = np.arange(1.1-(1e-7), 16.1-(1e-7), 0.1)
        self.len_scale = len(self.scale)
        self.cnt = 0
        self.cnt_total = 0
        self.num_repeat = 1
        self.repeat_total = (self.len_scale**2)*self.num_repeat-1
        self.min_input_npoint = int(4096/self.scale[-1])


        print("NUM_BATCH is %s" % self.num_batches)

    def get_scale_idx(self, num_repeat=20):
        self.cnt_total += 1
        if self.cnt_total == self.repeat_total:
            self.cnt_total = 0
        return int(self.cnt_total/(self.len_scale*num_repeat))


    def __getitem__(self, idx):
        if not self.refinenet:
            scale_idx = np.random.randint(self.len_scale)
            this_scale = self.scale[scale_idx]
            num_to_exclude = 0
            gt_npoint_to_sample = self.gt_data.shape[1]-num_to_exclude
            npoint_to_sample = int(gt_npoint_to_sample / this_scale)
            gt_npoint_to_sample = int(npoint_to_sample * this_scale)

            if self.cnt % self.num_batches == 0:
                idx = np.arange(self.sample_cnt)
                np.random.shuffle(idx)
                self.input_data = self.input_data[idx, ...]
                self.gt_data = self.gt_data[idx, ...]
                try:
                    self.radius_data = self.radius_data[idx, ...]
                except:
                    self.radius_data = self.radius_data
                self.cnt = 0


            start_idx = self.cnt * self.batch_size
            end_idx = (self.cnt + 1) * self.batch_size
            batch_input_data = self.input_data[start_idx:end_idx, :, :].copy()  # b,n,6
            batch_data_gt = self.gt_data[start_idx:end_idx, :, :].copy()  # b,n,6
            radius = self.radius_data[start_idx:end_idx].copy()  # b

            self.cnt += 1
            batch_data_gt = batch_data_gt[:,0:gt_npoint_to_sample,:]

            assert batch_data_gt.shape[1] == gt_npoint_to_sample
            new_batch_input = np.zeros((self.batch_size, npoint_to_sample, batch_input_data.shape[2]))
            if npoint_to_sample <= int(batch_input_data.shape[1]/2):
                for i in range(self.batch_size):
                    idx = nonuniform_sampling(batch_input_data.shape[1], sample_num=npoint_to_sample)
                    new_batch_input[i, ...] = batch_input_data[i][idx]
                batch_input_data = new_batch_input
            else:
                idx_all = np.arange(batch_input_data.shape[1])
                n_sample_exclude = batch_input_data.shape[1] - npoint_to_sample
                for i in range(self.batch_size):
                    idx = nonuniform_sampling(batch_input_data.shape[1], sample_num=n_sample_exclude)
                    idx = np.setdiff1d(idx_all, idx)
                    new_batch_input[i, ...] = batch_input_data[i][idx]
                batch_input_data = new_batch_input



            if self.use_norm:
                batch_input_data, batch_data_gt = rotate_point_cloud_and_gt(batch_input_data, batch_data_gt)
                batch_input_data, batch_data_gt, scales = random_scale_point_cloud_and_gt(batch_input_data,
                                                                                          batch_data_gt,
                                                                                          scale_low=0.8,
                                                                                          scale_high=1.2)
                radius = radius * scales
                batch_input_data, batch_data_gt = shift_point_cloud_and_gt(batch_input_data, batch_data_gt,
                                                                           shift_range=0.1)
                if np.random.rand() > 0.5:
                    batch_input_data = jitter_perturbation_point_cloud(batch_input_data, sigma=0.025, clip=0.05)
                if np.random.rand() > 0.5:
                    batch_input_data = rotate_perturbation_point_cloud(batch_input_data, angle_sigma=0.03,
                                                                       angle_clip=0.09)

            batch_input_data = torch.from_numpy(batch_input_data).type(torch.FloatTensor)
            batch_data_gt = torch.from_numpy(batch_data_gt).type(torch.FloatTensor)
            radius = torch.from_numpy(radius).type(torch.FloatTensor)
            this_scale = torch.from_numpy(np.array(this_scale)).type(torch.FloatTensor)

            return batch_input_data, batch_data_gt, radius, this_scale
        else: 
            if self.cnt % self.num_batches == 0:
                idx = np.arange(self.sample_cnt)
                np.random.shuffle(idx)
                self.input_data = self.input_data[idx, ...]
                self.gt_data = self.gt_data[idx, ...]
                try:
                    self.radius_data = self.radius_data[idx, ...]
                except:
                    self.radius_data = self.radius_data
                self.cnt = 0

            start_idx = self.cnt * self.batch_size
            end_idx = (self.cnt + 1) * self.batch_size
            batch_input_data = self.input_data[start_idx:end_idx, :, :].copy()  # b,n,6
            batch_data_gt = self.gt_data[start_idx:end_idx, :, :].copy()  # b,n,6

            self.cnt += 1

            batch_input_data, batch_data_gt = rotate_point_cloud_and_gt(batch_input_data, batch_data_gt)
            batch_input_data, batch_data_gt, _ = random_scale_point_cloud_and_gt(batch_input_data,
                                                                                      batch_data_gt,
                                                                                      scale_low=0.8,
                                                                                      scale_high=1.2)
            batch_input_data, batch_data_gt = shift_point_cloud_and_gt(batch_input_data, batch_data_gt,
                                                                       shift_range=0.1)
            if np.random.rand() > 0.75:
                batch_input_data = jitter_perturbation_point_cloud(batch_input_data, sigma=0.025, clip=0.05)
            elif np.random.rand() < 0.25:
                batch_input_data = jitter_perturbation_point_cloud(batch_input_data, sigma=0.005, clip=0.05)
            elif np.random.rand() < 0.5:
                batch_input_data = replace_point_cloud(batch_input_data)
            if np.random.rand() > 0.5:
                batch_input_data = rotate_perturbation_point_cloud(batch_input_data, angle_sigma=0.03,
                                                                   angle_clip=0.09)
            batch_input_data = torch.from_numpy(batch_input_data).type(torch.FloatTensor)
            batch_data_gt = torch.from_numpy(batch_data_gt).type(torch.FloatTensor)

            return batch_input_data, batch_data_gt


    def __len__(self):
        return self.num_batches

    def set_num_points(self, pts):
        self.num_points = min(self.input_data.shape[1], pts)

    def randomize(self):
        pass

