from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import time
import math
import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

try:
    import utils
except:
    from . import utils




# from PU-gan get_repulsion_loss but find some error in it
# from https://github.com/yulequan/PU-Net/blob/dd768ea2eb349dc1a35d62f8c1fb019efc526fe7/code/model_utils.py
def get_repulsion_loss(pred, nsample=20, radius=0.07, h=0.03, min_num=(torch.zeros(1).cuda()+1e-9), device=None):
    min_num = min_num.to(device)
    # if knn:
    #     _, idx = knn_point_2(nsample, pred, pred)
    #     pts_cnt = tf.constant(nsample, shape=(30, 1024))
    # else:
    #     idx, pts_cnt = query_ball_point(radius, nsample, pred, pred)
    # tf.summary.histogram('smooth/unque_index', pts_cnt)
    # print(nsample)
    idx,pts_cnt = utils.query_ball_point(radius, nsample, pred, pred)  # (B, npoint, nsample)

    # grouped_pred = group_point(pred, idx)  # (batch_size, npoint, nsample, 3)
    grouped_pred = utils.grouping_operation(pred.transpose(1, 2).contiguous(), idx.detach()).permute(0, 2, 3,
                                                                                                   1).contiguous()  # return b,c,npoint,nsample, to b,npoint,nsample,c
    grouped_pred -= torch.unsqueeze(pred, 2)

    # get the uniform loss
    # if use_l1:
    #     dists = tf.reduce_sum(tf.abs(grouped_pred), axis=-1)
    # else:
    #     dists = tf.reduce_sum(grouped_pred ** 2, axis=-1)
    dists = torch.sum(grouped_pred ** 2, -1, keepdim=False, out=None)
    del grouped_pred, idx, pts_cnt

    # val, idx = tf.nn.top_k(-dists, 5)
    dists = torch.topk(-dists, k=5)[0]
    dists = -dists[:, :, 1:]  # remove the first one
    # val = val[:, :, 1:]  # remove the first one

    # if use_l1:
    #     h = np.sqrt(h)*2
    # print(("h is ", h))

    # val = tf.maximum(0.0, h + val)  # dd/np.sqrt(n)
    # val = torch.max(torch.zeros_like(val), h + val)  # dd/np.sqrt(n)
    # dists = torch.nn.ReLU()(h + dists)
    # dists = torch.max(min_num, dists)
    dists = torch.nn.ReLU()(dists)
    dists_sqrt = torch.sqrt(dists)
    weight = torch.exp(-dists/h**2)
    del dists
    # repulsion_loss = torch.mean(val.view(-1), 0, keepdim=False)
    # return repulsion_loss
    return torch.mean((radius-dists_sqrt*weight).view(-1), 0, keepdim=False)



def get_pairwise_distance(batch_features):
    """Compute pairwise distance of a point cloud.
    Args:
      batch_features: numpy (batch_size, num_points, num_dims)
    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """

    og_batch_size = len(batch_features.shape)

    if og_batch_size == 2: #just two dimension
        batch_features = torch.expand_dims(batch_features, axis=0)


    batch_features_transpose = torch.transpose(batch_features, (0, 2, 1))

    #batch_features_inner = batch_features@batch_features_transpose
    batch_features_inner = torch.matmul(batch_features,batch_features_transpose)

    #print(torch.max(batch_features_inner), torch.min(batch_features_inner))


    batch_features_inner = -2 * batch_features_inner
    batch_features_square = torch.sum(torch.square(batch_features), axis=-1, keepdims=True)


    batch_features_square_tranpose = torch.transpose(batch_features_square, (0, 2, 1))

    return batch_features_square + batch_features_inner + batch_features_square_tranpose



# from https://github.com/liruihui/PU-GAN/blob/2672e2ac37b64421a8745aac3688ecfa7d404576/Common/loss_utils.py
def py_uniform_loss(points,idx,pts_cn,radius):
    #print(type(idx))
    idx = idx.detach()
    B,N,C = points.shape
    _,npoint,nsample = idx.shape
    pts_cn = pts_cn.cpu().numpy()
    uniform_vals = torch.zeros([1], requires_grad=True).cuda()
    for i in range(B):
        point = points[i]
        for j in range(npoint):
            number = pts_cn[i,j]
            coverage = (number - nsample)**2 / nsample
            if number<5:
                uniform_vals+=coverage
                continue
            _idx = idx[i, j, :number]
            # print(_idx)
            # disk_point = point[_idx]
            try:
                disk_point = point[_idx]
            except:
                # print(_idx)
                disk_point = torch.take(point, _idx.type(torch.LongTensor).cuda().detach())
            if disk_point.shape[0]<0:
                pair_dis = get_pairwise_distance(disk_point)#(batch_size, num_points, num_points)
                nan_valid = torch.where(pair_dis<1e-7)
                pair_dis[nan_valid]=0
                pair_dis = torch.squeeze(pair_dis, axis=0)
                pair_dis = torch.sort(pair_dis, axis=1)
                shortest_dis = torch.sqrt(pair_dis[:, 1])
            else:
                shortest_dis = get_knn_dis(disk_point,disk_point,2)
                shortest_dis = shortest_dis[:,1]
            disk_area = math.pi * (radius ** 2) / disk_point.shape[0]
            #expect_d = math.sqrt(disk_area)
            expect_d = math.sqrt(2 * disk_area / 1.732)  # using hexagon
            dis = (shortest_dis - expect_d)**2 / expect_d
            uniform_val = coverage * torch.mean(dis)

            uniform_vals+=uniform_val

    # uniform_dis = torch.array(uniform_vals).astype(torch.float32)

    # uniform_dis = torch.mean(uniform_dis)
    # uniform_dis = torch.mean(uniform_vals)
    # return uniform_dis
    return uniform_vals/npoint


from sklearn.neighbors import NearestNeighbors
def get_knn_dis(queries, pc, k):
    """
    queries [M, C]
    pc [P, C]
    """
    knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
    try:
        # print(pc)
        knn_search.fit(pc.cpu().detach().numpy())
    except:
        knn_search.fit(pc.cpu().detach().numpy().reshape(-1, 1))
    dis,knn_idx = knn_search.kneighbors(queries.cpu().detach().numpy().reshape(-1, 1), return_distance=True)
    #k_patches = np.take(pc, knn_idx, axis=0)  # M, K, C
    return torch.tensor(dis).cuda()



#whole version, slower
# from https://github.com/liruihui/PU-GAN/blob/2672e2ac37b64421a8745aac3688ecfa7d404576/Common/loss_utils.py
def get_uniform_loss2(pcd, percentages=[0.002,0.004,0.006,0.008,0.010,0.012,0.015], radius=1.0):
    B,N,C = pcd.shape
    npoint = int(N * 0.05)
    loss=torch.zeros([1], requires_grad=True).cuda()
    for p in percentages:
        nsample = int(N*p)
        r = math.sqrt(p*radius)
        new_xyz = pcd.transpose(1,2).contiguous() # b,3,n
        new_xyz = utils.gather_operation(new_xyz, utils.furthest_point_sample(pcd, npoint))
        new_xyz = new_xyz.transpose(1, 2).contiguous()  # b,npoint,3
        # idx, pts_cnt = query_ball_point(r, nsample, pcd, new_xyz)#(batch_size, npoint, nsample)
        idx, pts_cnt = utils.query_ball_point(r, nsample, pcd, new_xyz)  # (B, npoint, nsample)

        # uniform_val = tf.py_func(py_uniform_loss, [pcd, idx, pts_cnt, r], tf.float32)
        uniform_val = py_uniform_loss(pcd, idx.detach(), pts_cnt, r)

        mean = torch.mean(uniform_val, 0)*math.sqrt(p*100)
        loss += mean
    return loss/len(percentages)



def knn_point(k, xyz1, xyz2):
    """
    Input:
        k: int32, number of k in k-nn search
        xyz1: (batch_size, ndataset, c) float32 array, input points
        xyz2: (batch_size, npoint, c) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    """
    xyz1 = torch.unsqueeze(xyz1, 1)
    xyz2 = torch.unsqueeze(xyz2, 2)
    # print('!!!!!!!!!!!xyz1 xyz2!!!!!!!!!!')
    # print(xyz1.shape)
    # print(xyz2.shape)
    # dist = torch.squeeze((xyz1 - xyz2) ** 2, -1)
    # print(xyz1.shape)
    # print(xyz2.shape)
    dist = torch.sum((xyz1 - xyz2) ** 2, -1)
    # print(dist.shape)

    val, idx = torch.topk(-dist, k=k)
    del dist

    return val, idx.int()

