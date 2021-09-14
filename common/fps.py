"""
Farthest point sampling
CUDA: 10.1
torch: 1.7.1+cu101
"""
import torch
import numpy as np


def farthest_point_sample(xyz, npoint, RAN=True):
    """
    Input:
        xyz: pointcloud data, [B, N, C], type:Tensor[]
        npoint: number of samples, type:int
    Return:
        centroids: sampled pointcloud index, [B, npoint], type:Tensor[]
    """
    device = xyz.device
    # print(device)  # cpu
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  # 用来储存采样index
    distance = torch.ones(B, N).to(device) * 1e10  # 用来储存距离
    distance = distance.float()  # type(distance)==type(dist)

    if RAN:
        farthest = torch.randint(0, 1, (B,), dtype=torch.long).to(device)  # 表示上一次抽样到的点
    else:
        farthest = torch.randint(1, 2, (B,), dtype=torch.long).to(device)

    batch_indices = torch.arange(B, dtype=torch.long).to(device)  # 一个1-B的整数
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  # 找出上一次采样的点
        dist = torch.sum((xyz - centroid) ** 2, -1)
        dist = dist.float()  # type(distance)==type(dist)
        mask = dist < distance  # 更新每次最小距离
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]  # 求取最大距离
        # print(farthest)
    return centroids


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C],type:Tensor[]
        idx: sample index data, [B, S],type:Tensor[]
    Return:
        new_points:, indexed points data, [B, S, C],type:Tensor[]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


if __name__ == "__main__":
    pc = np.loadtxt("../data/test/103c9e43cdf6501c62b600da24e0965.txt")[:, :3]
    pc = pc[np.newaxis, :, :]
    pc = torch.from_numpy(pc)
    print(pc.shape)  # input: [1, 2632, 3]
    centroids = farthest_point_sample(pc, 1024)
    print(centroids.shape)  # index: [1, 1024]
    new_points = index_points(pc, centroids)
    print(new_points.shape)  # output: [1, 1024, 3]
    print(new_points.numpy()[0])
