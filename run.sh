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
python pc_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/ply_2_obj/ --input_format ply --output_format obj
# ply -> stl
python pc_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/ply_2_stl/ --input_format ply --output_format stl
# ply -> off
python pc_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/ply_2_off/ --input_format ply --output_format off

# obj -> ply
python pc_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/obj_2_ply/ --input_format obj --output_format ply
# obj -> off
python pc_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/obj_2_off/ --input_format obj --output_format off
# obj -> stl
python pc_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/obj_2_stl/ --input_format obj --output_format stl

# off -> ply
python pc_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/off_2_ply/ --input_format off --output_format ply
# off -> obj
python pc_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/off_2_obj/ --input_format off --output_format obj
# off -> stl
python pc_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/off_2_stl/ --input_format off --output_format stl

# stl -> ply
python pc_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/stl_2_ply/ --input_format stl --output_format ply
# stl -> obj
python pc_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/stl_2_obj/ --input_format stl --output_format obj
# stl -> off
python pc_factory.py --mode 1 --input_dir ../data/test --output_dir ../result/stl_2_off/ --input_format stl --output_format off

# 4. Mesh采样成点云
# Supported format: ["off", "ply", "obj", "stl"]->["ply", "xyz", "pcd"]
# 泊松磁盘采样
# off->xyz(1024p)
python pc_factory.py --mode 2 --input_dir ../data/test --output_dir ../result/possion/  --input_format off --output_format xyz --sampler poisson_disk_sampling --point_num 1024 --factor 5
# 均匀网格采样
# off->xyz(1024p)
python pc_factory.py --mode 2 --input_dir ../data/test --output_dir ../result/possion/  --input_format off --output_format xyz --sampler uniform_sampling --point_num 1024

# 5. 点云采样


