#!/usr/bin/env bash

# 1. 点云格式转换
# pcd -> *
# pcd -> [xyz, pts, txt, csv, ply]
cd lib/
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/pcd_2_xyz/ --input_format pcd --output_format xyz

# las -> *
# las -> [pcd, xyz, pts, ply, txt, csv]
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/las_2_pcd/ --input_format las --output_format pcd

# ply -> *
# ply -> [pcd, xyz, pts, txt, csv]
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/ply_2_pcd/ --input_format ply --output_format pcd

# xyz -> *
# xyz -> [pcd, pts, ply, txt, csv]
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/xyz_2_pcd/ --input_format xyz --output_format pcd

# pts -> *
# pts -> [pcd, xyz, ply, txt, csv]
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/pts_2_pcd/ --input_format pts --output_format pcd

# txt -> *
# txt -> [pcd, ply, xyz, pts]
python pc_factory.py --mode 0 --input_dir ../data/test --output_dir ../result/txt_2_pcd/ --input_format txt --output_format pcd

# 2. 计算3D Mesh的表面积和体积
# supported format: [ply, obj, off, stl]
python SurfaceAreaVolume.py --input_dir ../data/test --input_format ply

# 3. Mesh格式转换
# ply -> *
# ply -> [obj, stl, off]
python mesh_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/ply_2_obj/ --input_format ply --output_format obj

# obj -> *
# obj -> [ply, off, stl]
python mesh_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/obj_2_ply/ --input_format obj --output_format ply

# off -> *
# off -> [ply, obj, stl]
python mesh_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/off_2_ply/ --input_format off --output_format ply

# stl -> *
# stl -> [ply, obj, off]
python mesh_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/stl_2_ply/ --input_format stl --output_format ply

# 4. Mesh采样成点云
# Supported format: ["off", "ply", "obj", "stl"]->["ply", "xyz", "pcd"]
# 泊松磁盘采样(poisson disk sampling)
# off->xyz(1024p)
python mesh_factory.py --mode 2 --input_dir ../data/test --output_dir ../result/possion/  --input_format off --output_format xyz --sampler poisson_disk_sampling --point_num 1024 --factor 5
# 均匀网格采样(uniform sampling)
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
# loop
python mesh_factory.py --mode 6 --subdivision_type loop --iteration 1 --input_dir ../data/test --output_dir ../result/subdivision/loop/ --input_format ply
# midpoint
python mesh_factory.py --mode 6 --subdivision_type midpoint --iteration 1 --input_dir ../data/test --output_dir ../result/subdivision/midpoint/ --input_format ply

# 8. Convert 3d mesh into voxel grid
# supported input format: ["obj", "off", "dxf", "ply", "stl"]
# supported output format: ["binvox", "hips", "mira", "vtk", "msh"]
# binvox
python voxel_factory.py --mode 7 --input_dir ../data/test --output_dir ../result/mesh_2_voxel/ply2binvox/ --input_format ply --output_format binvox --d 256

# 9. voxel grid visualization
# supported input format: ["binvox", "mira"]
python voxel_vis.py --mode 8 --input_file ../result/mesh_2_voxel/ply2binvox/14.binvox
python voxel_vis.py --mode 8 --input_file ../result/mesh_2_voxel/ply2mira/14.mira

# 10. point cloud registration
# supported input format: [pcd,ply,xyz]
cd ../common
# ICP
python iterative_closest_point.py --s_file ../data/registration/bun000.ply --t_file ../data/registration/bun045.ply
# RANSAC
python RANSAC.py --s_file ../data/registration/bun000.ply --t_file ../data/registration/bun045.ply

# 11. point cloud upsampling
# Meta-PU
# supported system: Linux
# supported pc format: .xyz
python pc_factory.py --mode 9 --input_dir ../PU/Meta-PU/model/data/all_testset/4/input --input_format xyz --pu_model Meta-PU --scale 5.5

# 12. Convert 3d point cloud into voxel grid
# supported pc format: [pcd, ply, xyz, pts, txt]
# supported voxel format: [binvox]
# batch processing
python pc_factory.py --mode 10 --input_dir ../data/test --output_dir ../result/pc_voxel_grid/ --input_format pcd --output_format binvox --voxel 64
# single processing
python PointCloud2Voxel.py --mode 10 --input_file ../data/test/plant_0312.xyz --output_dir ../result/pc_voxel_grid/ --output_format binvox --voxel 64

# 13. convert 3d point cloud into mesh (3d construction)
# supported pc format: [pcd, xyz, pts, txt]
# supported mesh format: [off, ply, obj, stl]
# poisson surface reconstruction
python pc_factory.py --mode 11 --input_dir ../data/test --output_dir ../result/3d_poisson/ --input_format pcd --output_format off --constructor poisson --depth 9
# ball pivoting
python pc_factory.py --mode 11 --input_dir ../data/test --output_dir ../result/ball_pivoting/ --input_format pcd --output_format off --constructor ball_pivoting

# 14. point cloud visualization
# supported pc format: [pcd, xyz, pts, ply, txt, las]
cd ../common
python pointcloud_vis.py --mode 12 --input_file ../data/test/000001.pcd --scale_factor 0.1
python pointcloud_vis.py --mode 12 --input_file ../data/test/14.ply --scale_factor 0.8
python pointcloud_vis.py --mode 12 --input_file ../data/test/ --scale_factor 0.1
python pointcloud_vis.py --mode 12 --input_file ../data/test/simple1_3.las --scale_factor 0.1
python pointcloud_vis.py --mode 12 --input_file ../data/test/103c9e43cdf6501c62b600da24e0965.txt --scale_factor 0.008
python pointcloud_vis.py --mode 12 --input_file ../data/test/airplane.pts --scale_factor 0.008

# 15. mesh visualization
# supported mesh format: [ply, vtk, stl, obj, off, msh]
python mesh_vis.py --mode 13 --input_file ../data/test/b1.ply --screenshot ../result/snapshot/
python mesh_vis.py --mode 13 --input_file ../data/test/A380.obj --screenshot ../result/snapshot/
python mesh_vis.py --mode 13 --input_file ../data/test/02691156.130934b3dd2fddfaaf4f36f817f09501.stl --screenshot ../result/snapshot/
python mesh_vis.py --mode 13 --input_file ../result/mesh_2_voxel/ply2vtk/14.vtk --screenshot ../result/snapshot/
python mesh_vis.py --mode 13 --input_file ../data/test/02691156.3fb7ceab42d7b17219ba010ddb4974fe.off --screenshot ../result/snapshot/
python mesh_vis.py --mode 13 --input_file ../data/test/dumpbell.msh --screenshot ../result/snapshot/

