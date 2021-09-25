import numpy as np
from pyntcloud import PyntCloud
import os, sys
import shutil
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../lib'))
from configs import FLAGS
import binvox_rw
from pathlib import Path
import pathlib
import open3d as o3d


def pc2xyz(input_file, input_pc_format):
    """
    将其它点云格式转成xyz格式
    :param input_file: 输入其它的点云文件
    :param input_pc_format: 输入的点云格式
    :return: 同名的xyz格式的点云
    """
    # 获取文件名
    stem = Path(input_file).stem
    # 创建临时文件夹
    tmp = "../data/tmp/"
    pathlib.Path(tmp).mkdir(parents=True, exist_ok=True)
    output_file = tmp + stem + ".xyz"
    if input_pc_format == 'pcd':
        # pcd->xyz
        pc = o3d.io.read_point_cloud(input_file)
        o3d.io.write_point_cloud(output_file, pc)
        print("Done! tmp xyz point cloud is saved in: ", output_file)
    elif input_pc_format == 'ply':
        # ply->xyz
        pc = o3d.io.read_point_cloud(input_file)
        o3d.io.write_point_cloud(output_file, pc)
        print("Done! tmp xyz point cloud is saved in: ", output_file)
    elif input_pc_format == 'pts':
        # pts->xyz
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            first_line = lines[0].split(' ')
            try:
                assert len(first_line) == 1
                # header
                pc = o3d.io.read_point_cloud(input_file)
                o3d.io.write_point_cloud(output_file, pc)
                print("Done! tmp xyz point cloud is saved in: ", output_file)
            except:
                # no header
                pc = np.loadtxt(input_file, delimiter=' ')
                # 创建open3d对象
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc)
                o3d.io.write_point_cloud(output_file, pcd)
                print("Done! tmp xyz point cloud is saved in: ", output_file)
    elif input_pc_format == "txt":
        pc = np.loadtxt(input_file, delimiter=' ')[:, :3]
        # 创建open3d对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        o3d.io.write_point_cloud(output_file, pcd)
        print("Done! tmp xyz point cloud is saved in: ", output_file)


def pointcloud2voxel(input_file, input_pc_format, voxel_size, output_file):
    """
    将xyz格式的点云转成体素
    :param input_file: 输入的任意格式的点云文件
    :param input_pc_format: 输入的点云格式
    :param voxel_size: 体素的大小：voxel_size×voxel_size×voxel_size
    :param output_file: 体素文件的保存路径
    :return:
    """
    # 获取文件名
    stem = Path(input_file).stem
    # 首先将点云转成xyz
    if input_pc_format in ['pcd', 'ply', 'pts', 'txt']:
        print(":: convert {} to xyz.".format(input_pc_format))
        pc2xyz(input_file, input_pc_format)
        input_file = "../data/tmp/" + stem + ".xyz"
    else:
        input_file = input_file

    # xyz->binvox
    print(":: convert xyz to binvox.")
    cloud = PyntCloud.from_file(input_file,
                                sep=" ",
                                header=0,
                                names=["x", "y", "z"])

    voxelgrid_id = cloud.add_structure("voxelgrid", n_x=voxel_size, n_y=voxel_size, n_z=voxel_size)
    # print(voxelgrid_id)
    voxelgrid = cloud.structures[voxelgrid_id]
    # cloud.plot(mesh=True, backend="threejs")
    # voxelgrid.plot(d=3, mode="density", cmap="hsv")

    x_cords = voxelgrid.voxel_x
    y_cords = voxelgrid.voxel_y
    z_cords = voxelgrid.voxel_z

    voxel = np.zeros((voxel_size, voxel_size, voxel_size)).astype(np.bool8)

    for x, y, z in zip(x_cords, y_cords, z_cords):
        voxel[x][y][z] = True

    with open(output_file, 'wb') as f:
        v = binvox_rw.Voxels(voxel, (voxel_size, voxel_size, voxel_size), (0, 0, 0), 1, 'xyz')
        v.write(f)
    print("Done! result is saved in {}".format(output_file))
    print(":::::::::::::::::::::::::::::::::::::::::::::::")
    # 删除临时文件夹及所有文件
    tmp_dir = "../data/tmp/"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    if FLAGS.mode == 10:
        file_path = FLAGS.input_file
        output_path = FLAGS.output_dir
        voxel_size = FLAGS.voxel
        ext = os.path.splitext(file_path)[-1][1:]
        stem = pathlib.Path(file_path).stem
        output_format = FLAGS.output_format
        output_file = output_path + stem + "." + output_format
        pointcloud2voxel(file_path, ext, voxel_size, output_file)
