#!/usr/bin/env bash

# 1. 点云格式转换
# pcd -> xyz
cd lib/
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/pcd_2_xyz/ --input_format pcd --output_format xyz
# pcd -> pts
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/pcd_2_pts/ --input_format pcd --output_format pts
# pcd -> txt
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/pcd_2_txt/ --input_format pcd --output_format txt
# pcd -> csv
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/pcd_2_csv/ --input_format pcd --output_format csv
# pcd -> ply
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/pcd_2_ply/ --input_format pcd --output_format ply

# las -> pcd
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/las_2_pcd/ --input_format las --output_format pcd
# las -> xyz
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/las_2_xyz/ --input_format las --output_format xyz
# las -> pts
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/las_2_pts/ --input_format las --output_format pts
# las -> ply
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/las_2_ply/ --input_format las --output_format ply
# las -> txt
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/las_2_txt/ --input_format las --output_format txt
# las -> csv
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/las_2_csv/ --input_format las --output_format csv

# ply -> pcd
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/ply_2_pcd/ --input_format ply --output_format pcd
# ply -> xyz
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/ply_2_xyz/ --input_format ply --output_format xyz
# ply -> pts
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/ply_2_pts/ --input_format ply --output_format pts
# ply -> txt
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/ply_2_txt/ --input_format ply --output_format txt
# ply -> csv
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/ply_2_csv/ --input_format ply --output_format csv

# xyz -> pcd
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/xyz_2_pcd/ --input_format xyz --output_format pcd
# xyz -> pts
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/xyz_2_pts/ --input_format xyz --output_format pts
# xyz -> ply
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/xyz_2_ply/ --input_format xyz --output_format ply
# xyz -> txt
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/xyz_2_txt/ --input_format xyz --output_format txt
# xyz -> csv
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/xyz_2_csv/ --input_format xyz --output_format csv

# pts -> pcd
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/pts_2_pcd/ --input_format pts --output_format pcd
# pts -> xyz
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/pts_2_xyz/ --input_format pts --output_format xyz
# pts -> ply
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/pts_2_ply/ --input_format pts --output_format ply
# pts -> txt
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/pts_2_txt/ --input_format pts --output_format txt
# pts -> csv
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/pts_2_csv/ --input_format pts --output_format csv

# txt -> pcd
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/txt_2_pcd/ --input_format txt --output_format pcd
# txt -> ply
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/txt_2_ply/ --input_format txt --output_format ply
# txt -> xyz
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/txt_2_xyz/ --input_format txt --output_format xyz
# txt -> pts
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/txt_2_pts/ --input_format txt --output_format pts

# 2. 计算3D Mesh的表面积和体积
python SurfaceAreaVolume.py --input_dir ../data/test --input_format ply
python SurfaceAreaVolume.py --input_dir ../data/test --input_format obj
python SurfaceAreaVolume.py --input_dir ../data/test --input_format off
python SurfaceAreaVolume.py --input_dir ../data/test --input_format stl

# 3. Mesh格式转换
# ply -> obj
python mesh_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/ply_2_obj/ --input_format ply --output_format obj
# ply -> stl
python mesh_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/ply_2_stl/ --input_format ply --output_format stl
# ply -> off
python mesh_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/ply_2_off/ --input_format ply --output_format off

# obj -> ply
python mesh_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/obj_2_ply/ --input_format obj --output_format ply
# obj -> off
python mesh_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/obj_2_off/ --input_format obj --output_format off
# obj -> stl
python mesh_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/obj_2_stl/ --input_format obj --output_format stl

# off -> ply
python mesh_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/off_2_ply/ --input_format off --output_format ply
# off -> obj
python mesh_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/off_2_obj/ --input_format off --output_format obj
# off -> stl
python mesh_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/off_2_stl/ --input_format off --output_format stl

# stl -> ply
python mesh_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/stl_2_ply/ --input_format stl --output_format ply
# stl -> obj
python mesh_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/stl_2_obj/ --input_format stl --output_format obj
# stl -> off
python mesh_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/stl_2_off/ --input_format stl --output_format off

# 4. Mesh采样成点云
# Supported format: ["off", "ply", "obj", "stl"]->["ply", "xyz", "pcd"]
# 泊松磁盘采样
# off->xyz(1024p)
python mesh_factory.py --mode 2 --input_dir ../data/test --output_dir ../result/possion/  --input_format off --output_format xyz --sampler poisson_disk_sampling --point_num 1024 --factor 5
# 均匀网格采样
# off->xyz(1024p)
python mesh_factory.py --mode 2 --input_dir ../data/test --output_dir ../result/uniform/  --input_format off --output_format xyz --sampler uniform_sampling --point_num 1024

# 5. 点云/网格滤波
# 点云滤波
# Supported format: [ply, xyz, pts, pcd, txt]
# PassThroughFilter
python pc_factory.py --mode 3 --filter PassThroughFilter --upper_limit 5 --input_dir ../data/test --output_dir ../result/filter/PassThroughFilter/ --input_format pcd
# VoxelGrid
python pc_factory.py --mode 3 --filter VoxelGridFilter --voxel_size 0.1 --input_dir ../data/test --output_dir ../result/filter/VoxelGridFilter/ --input_format pcd
# project_inliers
python pc_factory.py --mode 3 --filter project_inliers --input_dir ../data/test --output_dir ../result/filter/project_inliers/ --input_format pcd
# remove_outlier
python pc_factory.py --mode 3 --filter remove_outliers --removal radius --radius 5.0 --min_neighbor 3 --input_dir ../data/test --output_dir ../result/filter/remove_outlier/ --input_format pcd
python pc_factory.py --mode 3 --filter remove_outliers --removal condition --radius 5.0 --min_neighbor 3 --input_dir ../data/test --output_dir ../result/filter/remove_outlier/ --input_format pcd
# statistical_removal
python pc_factory.py --mode 3 --filter statistical_removal --std_dev 1.0 --input_dir ../data/test --output_dir ../result/filter/statistical_removal/ --input_format pcd

# 网格滤波
# Supported format: [ply, obj, stl]
# taubin filter
python mesh_factory.py --mode 4 --mesh_filter taubin --input_dir ../data/test --output_dir ../result/filter/taubin/ --input_format ply
# Laplacian smooth filter
python mesh_factory.py --mode 4 --mesh_filter laplacian --input_dir ../data/test --output_dir ../result/filter/laplacian/ --input_format ply
# simple neighbour average
python mesh_factory.py --mode 4 --mesh_filter neighbour --input_dir ../data/test --output_dir ../result/filter/simple/ --input_format ply

# 6. 点云下采样
# Supported format: [ply, xyz, pts, pcd, txt]
# FPS(recommended)
python pc_factory.py --mode 5 --down_sampler fps --point_num 2048 --input_dir ../data/test --output_dir ../result/downsample/fps/ --input_format ply
# random down sampling
python pc_factory.py --mode 5 --down_sampler random --point_num 2048 --input_dir ../data/test --output_dir ../result/downsample/random/ --input_format ply
# uniform down sampling
python pc_factory.py --mode 5 --down_sampler uniform --k 4 --input_dir ../data/test --output_dir ../result/downsample/uniform/ --input_format ply
# voxel down sampling
python pc_factory.py --mode 5 --down_sampler voxel --voxel_size 0.5 --input_dir ../data/test --output_dir ../result/downsample/voxel/ --input_format ply

# 7. mesh subdivision
# suitable iteration may be in [1,2,3,4]
python mesh_factory.py --mode 6 --subdivision_type loop --iteration 1 --input_dir ../data/test --output_dir ../result/subdivision/loop/ --input_format ply
python mesh_factory.py --mode 6 --subdivision_type midpoint --iteration 1 --input_dir ../data/test --output_dir ../result/subdivision/midpoint/ --input_format ply

# 8. Convert 3d mesh into voxel grid
# supported input format: ["obj", "off", "dxf", "ply", "stl"]
# supported output format: ["binvox", "hips", "mira", "vtk", "msh"]
# binvox
python voxel_factory.py --mode 7 --input_dir ../data/test --output_dir ../result/mesh_2_voxel/ply2binvox/ --input_format ply --output_format binvox --d 256
# mira
python voxel_factory.py --mode 7 --input_dir ../data/test --output_dir ../result/mesh_2_voxel/ply2mira/ --input_format ply --output_format mira --d 256
# vtk
python voxel_factory.py --mode 7 --input_dir ../data/test --output_dir ../result/mesh_2_voxel/ply2vtk/ --input_format ply --output_format vtk --d 256
# msh
python voxel_factory.py --mode 7 --input_dir ../data/test --output_dir ../result/mesh_2_voxel/ply2msh/ --input_format ply --output_format msh --d 256
# hips
python voxel_factory.py --mode 7 --input_dir ../data/test --output_dir ../result/mesh_2_voxel/ply2hips/ --input_format ply --output_format hips --d 256

# 9. voxel grid visualization
# supported input format: ["binvox", "mira"]
python voxel_factory.py --mode 8 --input_dir ../result/mesh_2_voxel/ply2binvox --input_format binvox --input_file ../result/mesh_2_voxel/ply2binvox/14.binvox
python voxel_factory.py --mode 8 --input_dir ../result/mesh_2_voxel/ply2mira --input_format mira --input_file ../result/mesh_2_voxel/ply2mira/14.mira

# 10. point cloud registration
# supported input format: [pcd,ply,xyz]
cd ../common
# ICP
python iterative_closest_point.py --s_file ../data/registration/bun000.ply --t_file ../data/registration/bun045.ply
# RANSAC
python RANSAC.py --s_file ../data/registration/bun000.ply --t_file ../data/registration/bun045.ply
