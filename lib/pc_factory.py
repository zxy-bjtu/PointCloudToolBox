"""
This script is used for point cloud and 3D mesh format conversion.
The common pc format: .pts, .las, .pcd, .xyz, .ply, .txt
The common mesh format: .obj, .off, .ply, .stl
The common voxel format: .vxl, .vox, .kvx, .kv6, .v3a, .v3b
"""
import os
import sys
import platform
from time import time

import open3d as o3d
from pathlib import Path
import pathlib
import numpy as np
import pcl
import vtk
from plyfile import PlyData
from openmesh import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../common'))
from configs import FLAGS
from pc_io import get_all_files
from read_las import read_las
from filter import passThroughFilter, voxelGrid, project_inliers, remove_outlier, StatisticalOutlierRemovalFilter


class PointCloud_FormatFactory(object):
    def __init__(self, opts, filelist):
        """
        :param opts: 超参
        :param filelist: 要处理的文件列表
        """
        self.opts = opts
        self.filelist = filelist
        self.filenum = len(self.filelist)  # 要处理的文件个数

    def pc_pc(self):
        """
        点云格式转换函数
        :return:
        """
        for i in range(self.filenum):
            print("Processing: ", self.filelist[i])
            file_path = self.filelist[i]  # 要处理的文件
            input_pc_format = self.opts.input_format
            output_pc_format = self.opts.output_format

            # pcd->*
            if input_pc_format == 'pcd':
                # pcd->xyz
                if output_pc_format == 'xyz':
                    pc = o3d.io.read_point_cloud(file_path)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.xyz'
                    o3d.io.write_point_cloud(output_file, pc)
                    print("Done! result is saved in: ", output_file)
                # pcd->pts
                elif output_pc_format == 'pts':
                    pc = o3d.io.read_point_cloud(file_path)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.pts'
                    o3d.io.write_point_cloud(output_file, pc)
                    print("Done! result is saved in: ", output_file)
                # pcd->txt
                elif output_pc_format == 'txt':
                    pc = o3d.io.read_point_cloud(file_path)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.txt'
                    # txt is not supported!
                    # o3d.io.write_point_cloud(output_file, pc)
                    xyz = np.asarray(pc.points)
                    # print(xyz)
                    np.savetxt(output_file, xyz)
                    print("Done! result is saved in: ", output_file)
                # pcd->csv
                elif output_pc_format == 'csv':
                    pc = o3d.io.read_point_cloud(file_path)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.csv'
                    # csv is not supported!
                    # o3d.io.write_point_cloud(output_file, pc)
                    xyz = np.asarray(pc.points)
                    # print(xyz)
                    np.savetxt(output_file, xyz, delimiter=',')
                    print("Done! result is saved in: ", output_file)
                # pcd->ply
                elif output_pc_format == 'ply':
                    pc = o3d.io.read_point_cloud(file_path)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.ply'
                    o3d.io.write_point_cloud(output_file, pc)
                    print("Done! result is saved in: ", output_file)
                else:
                    raise Exception('Unsupported Output file format! Only [xyz, pts, txt, csv, ply] is supported!')

            # las->*
            elif input_pc_format == 'las':
                # las->pcd
                if output_pc_format == 'pcd':
                    pc_list = read_las(file_path)
                    pc = []
                    for i in range(len(pc_list)):
                        pc.append(list(pc_list[i]))
                    point_xyz = np.array(pc)
                    # print(point_xyz)
                    # 创建open3d对象
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(point_xyz)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.pcd'
                    o3d.io.write_point_cloud(output_file, pcd)
                    print("Done! result is saved in: ", output_file)
                # las->xyz
                elif output_pc_format == 'xyz':
                    pc_list = read_las(file_path)
                    pc = []
                    for i in range(len(pc_list)):
                        pc.append(list(pc_list[i]))
                    point_xyz = np.array(pc)
                    # print(point_xyz)
                    # 创建open3d对象
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(point_xyz)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.xyz'
                    o3d.io.write_point_cloud(output_file, pcd)
                    print("Done! result is saved in: ", output_file)
                # las->pts
                elif output_pc_format == 'pts':
                    pc_list = read_las(file_path)
                    pc = []
                    for i in range(len(pc_list)):
                        pc.append(list(pc_list[i]))
                    point_xyz = np.array(pc)
                    # print(point_xyz)
                    # 创建open3d对象
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(point_xyz)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.pts'
                    o3d.io.write_point_cloud(output_file, pcd)
                    print("Done! result is saved in: ", output_file)
                # las->ply
                elif output_pc_format == 'ply':
                    pc_list = read_las(file_path)
                    pc = []
                    for i in range(len(pc_list)):
                        pc.append(list(pc_list[i]))
                    point_xyz = np.array(pc)
                    # print(point_xyz)
                    # 创建open3d对象
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(point_xyz)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.ply'
                    o3d.io.write_point_cloud(output_file, pcd)
                    print("Done! result is saved in: ", output_file)
                # las->txt
                elif output_pc_format == 'txt':
                    pc_list = read_las(file_path)
                    pc = []
                    for i in range(len(pc_list)):
                        pc.append(list(pc_list[i]))
                    point_xyz = np.array(pc)
                    # print(point_xyz)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.txt'
                    np.savetxt(output_file, point_xyz)
                    print("Done! result is saved in: ", output_file)
                # las->csv
                elif output_pc_format == 'csv':
                    pc_list = read_las(file_path)
                    pc = []
                    for i in range(len(pc_list)):
                        pc.append(list(pc_list[i]))
                    point_xyz = np.array(pc)
                    # print(point_xyz)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.csv'
                    np.savetxt(output_file, point_xyz, delimiter=',')
                    print("Done! result is saved in: ", output_file)
                else:
                    raise Exception('Unsupported Output file format! Only [pcd, xyz, pts, txt, csv, ply] is supported!')

            # ply->*
            elif input_pc_format == 'ply':
                # ply->pcd
                if output_pc_format == 'pcd':
                    pc = o3d.io.read_point_cloud(file_path)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.pcd'
                    o3d.io.write_point_cloud(output_file, pc)
                    print("Done! result is saved in: ", output_file)
                # ply->xyz
                elif output_pc_format == 'xyz':
                    pc = o3d.io.read_point_cloud(file_path)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.xyz'
                    o3d.io.write_point_cloud(output_file, pc)
                    print("Done! result is saved in: ", output_file)
                # ply->pts
                elif output_pc_format == 'pts':
                    pc = o3d.io.read_point_cloud(file_path)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.pts'
                    o3d.io.write_point_cloud(output_file, pc)
                    print("Done! result is saved in: ", output_file)
                # ply->txt
                elif output_pc_format == 'txt':
                    pc = o3d.io.read_point_cloud(file_path)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.txt'
                    xyz = np.asarray(pc.points)
                    np.savetxt(output_file, xyz)
                    # o3d.io.write_point_cloud(output_file, pc)
                    print("Done! result is saved in: ", output_file)
                # ply->csv
                elif output_pc_format == 'csv':
                    pc = o3d.io.read_point_cloud(file_path)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.csv'
                    xyz = np.asarray(pc.points)
                    np.savetxt(output_file, xyz, delimiter=',')
                    # o3d.io.write_point_cloud(output_file, pc)
                    print("Done! result is saved in: ", output_file)
                else:
                    raise Exception('Unsupported Output file format! Only [pcd, xyz, pts, txt, csv] is supported!')

            # xyz->*
            elif input_pc_format == 'xyz':
                # xyz->pcd
                if output_pc_format == 'pcd':
                    pc = o3d.io.read_point_cloud(file_path)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.pcd'
                    o3d.io.write_point_cloud(output_file, pc)
                    print("Done! result is saved in: ", output_file)
                # xyz->pts
                elif output_pc_format == 'pts':
                    pc = o3d.io.read_point_cloud(file_path)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.pts'
                    o3d.io.write_point_cloud(output_file, pc)
                    print("Done! result is saved in: ", output_file)
                # xyz->ply
                elif output_pc_format == 'ply':
                    pc = o3d.io.read_point_cloud(file_path)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.ply'
                    o3d.io.write_point_cloud(output_file, pc)
                    print("Done! result is saved in: ", output_file)
                # xyz->txt
                elif output_pc_format == 'txt':
                    pc = o3d.io.read_point_cloud(file_path)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.txt'
                    xyz = np.asarray(pc.points)
                    np.savetxt(output_file, xyz)
                    print("Done! result is saved in: ", output_file)
                # xyz->csv
                elif output_pc_format == 'csv':
                    pc = o3d.io.read_point_cloud(file_path)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.csv'
                    xyz = np.asarray(pc.points)
                    np.savetxt(output_file, xyz, delimiter=',')
                    print("Done! result is saved in: ", output_file)
                else:
                    raise Exception('Unsupported Output file format! Only [pcd, pts, ply, txt, csv] is supported!')

            # pts->*
            elif input_pc_format == "pts":
                # pts->pcd
                if output_pc_format == 'pcd':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        first_line = lines[0].split(' ')
                        # print(first_line)
                        try:
                            assert len(first_line) == 1
                            # header
                            pc = o3d.io.read_point_cloud(file_path)
                            # 获取文件名
                            stem = Path(file_path).stem
                            # 输出文件夹不存在则创建
                            pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                            output_file = self.opts.output_dir + stem + '.pcd'
                            o3d.io.write_point_cloud(output_file, pc)
                            print("Done! result is saved in: ", output_file)
                        except:
                            # no header
                            pc = np.loadtxt(file_path, delimiter=' ')
                            # print(pc)
                            # 创建open3d对象
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(pc)
                            # 获取文件名
                            stem = Path(file_path).stem
                            # 输出文件夹不存在则创建
                            pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                            output_file = self.opts.output_dir + stem + '.pcd'
                            o3d.io.write_point_cloud(output_file, pcd)
                            print("Done! result is saved in: ", output_file)
                # pts->xyz
                elif output_pc_format == 'xyz':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        first_line = lines[0].split(' ')
                        try:
                            assert len(first_line) == 1
                            # header
                            pc = o3d.io.read_point_cloud(file_path)
                            # 获取文件名
                            stem = Path(file_path).stem
                            # 输出文件夹不存在则创建
                            pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                            output_file = self.opts.output_dir + stem + '.xyz'
                            o3d.io.write_point_cloud(output_file, pc)
                            print("Done! result is saved in: ", output_file)
                        except:
                            # no header
                            pc = np.loadtxt(file_path, delimiter=' ')
                            # 创建open3d对象
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(pc)
                            # 获取文件名
                            stem = Path(file_path).stem
                            # 输出文件夹不存在则创建
                            pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                            output_file = self.opts.output_dir + stem + '.xyz'
                            o3d.io.write_point_cloud(output_file, pcd)
                            print("Done! result is saved in: ", output_file)
                # pts->ply
                elif output_pc_format == 'ply':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        first_line = lines[0].split(' ')
                        try:
                            assert len(first_line) == 1
                            # header
                            pc = o3d.io.read_point_cloud(file_path)
                            # 获取文件名
                            stem = Path(file_path).stem
                            # 输出文件夹不存在则创建
                            pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                            output_file = self.opts.output_dir + stem + '.ply'
                            o3d.io.write_point_cloud(output_file, pc)
                            print("Done! result is saved in: ", output_file)
                        except:
                            # no header
                            pc = np.loadtxt(file_path, delimiter=' ')
                            # 创建open3d对象
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(pc)
                            # 获取文件名
                            stem = Path(file_path).stem
                            # 输出文件夹不存在则创建
                            pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                            output_file = self.opts.output_dir + stem + '.ply'
                            o3d.io.write_point_cloud(output_file, pcd)
                            print("Done! result is saved in: ", output_file)
                # pts->txt
                elif output_pc_format == 'txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        first_line = lines[0].split(' ')
                        try:
                            assert len(first_line) == 1
                            # header
                            pc = o3d.io.read_point_cloud(file_path)
                            # 获取文件名
                            stem = Path(file_path).stem
                            # 输出文件夹不存在则创建
                            pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                            output_file = self.opts.output_dir + stem + '.txt'
                            xyz = np.asarray(pc.points)
                            np.savetxt(output_file, xyz)
                            print("Done! result is saved in: ", output_file)
                        except:
                            # no header
                            pc = np.loadtxt(file_path, delimiter=' ')
                            # 获取文件名
                            stem = Path(file_path).stem
                            # 输出文件夹不存在则创建
                            pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                            output_file = self.opts.output_dir + stem + '.txt'
                            np.savetxt(output_file, pc)
                            print("Done! result is saved in: ", output_file)
                # pts->csv
                elif output_pc_format == 'csv':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        first_line = lines[0].split(' ')
                        try:
                            assert len(first_line) == 1
                            # header
                            pc = o3d.io.read_point_cloud(file_path)
                            # 获取文件名
                            stem = Path(file_path).stem
                            # 输出文件夹不存在则创建
                            pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                            output_file = self.opts.output_dir + stem + '.csv'
                            xyz = np.asarray(pc.points)
                            np.savetxt(output_file, xyz, delimiter=',')
                            print("Done! result is saved in: ", output_file)
                        except:
                            # no header
                            pc = np.loadtxt(file_path, delimiter=' ')
                            # 获取文件名
                            stem = Path(file_path).stem
                            # 输出文件夹不存在则创建
                            pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                            output_file = self.opts.output_dir + stem + '.csv'
                            np.savetxt(output_file, pc, delimiter=',')
                            print("Done! result is saved in: ", output_file)
                else:
                    raise Exception('Unsupported Output file format! Only [pcd, xyz, ply, txt, csv] is supported!')

            # txt->*
            elif input_pc_format == 'txt':
                # txt->pcd
                if output_pc_format == 'pcd':
                    pc = np.loadtxt(file_path, delimiter=' ')[:, :3]
                    # 创建open3d对象
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pc)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.pcd'
                    o3d.io.write_point_cloud(output_file, pcd)
                    print("Done! result is saved in: ", output_file)
                # txt->ply
                elif output_pc_format == 'ply':
                    pc = np.loadtxt(file_path, delimiter=' ')[:, :3]
                    # 创建open3d对象
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pc)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.ply'
                    o3d.io.write_point_cloud(output_file, pcd)
                    print("Done! result is saved in: ", output_file)
                # txt->xyz
                elif output_pc_format == 'xyz':
                    pc = np.loadtxt(file_path, delimiter=' ')[:, :3]
                    # 创建open3d对象
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pc)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.xyz'
                    o3d.io.write_point_cloud(output_file, pcd)
                    print("Done! result is saved in: ", output_file)
                # txt->pts
                elif output_pc_format == 'pts':
                    pc = np.loadtxt(file_path, delimiter=' ')[:, :3]
                    # 创建open3d对象
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pc)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.pts'
                    o3d.io.write_point_cloud(output_file, pcd)
                    print("Done! result is saved in: ", output_file)
                else:
                    raise Exception('Unsupported Output file format! Only [pcd, ply, xyz, pts] is supported!')
            else:
                raise Exception('Unsupported Input file format! Only [pcd, las, ply, xyz, pts, txt] is supported!')

    def mesh_mesh(self):
        """
        3d mesh格式转换
        :return:
        """
        for i in range(self.filenum):
            print("Processing: ", self.filelist[i])
            file_path = self.filelist[i]  # 要处理的文件
            input_pc_format = self.opts.input_format
            output_pc_format = self.opts.output_format

            # ply->*
            if input_pc_format == 'ply':
                # ply->obj
                if output_pc_format == 'obj':
                    ply = PlyData.read(file_path)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.obj'
                    with open(output_file, 'w') as f:
                        f.write("# OBJ file\n")
                        verteces = ply['vertex']
                        for v in verteces:
                            p = [v['x'], v['y'], v['z']]
                            if 'red' in v and 'green' in v and 'blue' in v:
                                c = [v['red'] / 256, v['green'] / 256, v['blue'] / 256]
                            else:
                                c = [0, 0, 0]
                            a = p + c
                            f.write("v %.6f %.6f %.6f %.6f %.6f %.6f \n" % tuple(a))
                        for v in verteces:
                            if 'nx' in v and 'ny' in v and 'nz' in v:
                                n = (v['nx'], v['ny'], v['nz'])
                                f.write("vn %.6f %.6f %.6f\n" % n)
                        for v in verteces:
                            if 's' in v and 't' in v:
                                t = (v['s'], v['t'])
                                f.write("vt %.6f %.6f\n" % t)
                        if 'face' in ply:
                            for i in ply['face']['vertex_indices']:
                                f.write("f")
                                for j in range(i.size):
                                    ii = [i[j] + 1, i[j] + 1, i[j] + 1]
                                    f.write(" %d/%d/%d" % tuple(ii))
                                f.write("\n")
                    print("Done! result is saved in: ", output_file)
                # ply->stl
                elif output_pc_format == 'stl':
                    mesh = o3d.io.read_triangle_mesh(file_path)
                    # print("计算法线: ")
                    mesh.compute_vertex_normals()
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.stl'
                    o3d.io.write_triangle_mesh(output_file, mesh)
                    print("Done! result is saved in: ", output_file)
                # ply->off
                elif output_pc_format == 'off':
                    mesh = o3d.io.read_triangle_mesh(file_path)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.off'
                    o3d.io.write_triangle_mesh(output_file, mesh)
                    print("Done! result is saved in: ", output_file)
                else:
                    raise Exception('Unsupported Output file format! Only [obj, stl, off] is supported!')

            # obj->*
            elif input_pc_format == 'obj':
                # obj->ply
                if output_pc_format == 'ply':
                    mesh = o3d.io.read_triangle_mesh(file_path)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.ply'
                    o3d.io.write_triangle_mesh(output_file, mesh)
                    print("Done! result is saved in: ", output_file)
                # obj->off
                elif output_pc_format == 'off':
                    mesh = read_polymesh(file_path)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.off'
                    write_mesh(output_file, mesh)
                    print("Done! result is saved in: ", output_file)
                # obj->stl
                elif output_pc_format == 'stl':
                    mesh = o3d.io.read_triangle_mesh(file_path)
                    # print("计算法线: ")
                    mesh.compute_vertex_normals()
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.stl'
                    o3d.io.write_triangle_mesh(output_file, mesh)
                    print("Done! result is saved in: ", output_file)
                else:
                    raise Exception('Unsupported Output file format! Only [ply, off, stl] is supported!')

            # off->*
            elif input_pc_format == 'off':
                # off->ply
                if output_pc_format == 'ply':
                    mesh = o3d.io.read_triangle_mesh(file_path)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.ply'
                    o3d.io.write_triangle_mesh(output_file, mesh)
                    print("Done! result is saved in: ", output_file)
                # off->obj
                elif output_pc_format == 'obj':
                    mesh = o3d.io.read_triangle_mesh(file_path)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.obj'
                    o3d.io.write_triangle_mesh(output_file, mesh)
                    print("Done! result is saved in: ", output_file)
                # off->stl
                elif output_pc_format == 'stl':
                    mesh = o3d.io.read_triangle_mesh(file_path)
                    # print("计算法线: ")
                    mesh.compute_vertex_normals()
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.stl'
                    o3d.io.write_triangle_mesh(output_file, mesh)
                    print("Done! result is saved in: ", output_file)
                else:
                    raise Exception('Unsupported Output file format! Only [ply, obj, stl] is supported!')

            # stl->*
            elif input_pc_format == 'stl':
                # stl->ply
                if output_pc_format == 'ply':
                    mesh = o3d.io.read_triangle_mesh(file_path)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.ply'
                    o3d.io.write_triangle_mesh(output_file, mesh)
                    print("Done! result is saved in: ", output_file)
                # stl->obj
                elif output_pc_format == 'obj':
                    mesh = o3d.io.read_triangle_mesh(file_path)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.obj'
                    o3d.io.write_triangle_mesh(output_file, mesh)
                    print("Done! result is saved in: ", output_file)
                # stl->off
                elif output_pc_format == 'off':
                    mesh = read_polymesh(file_path)
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    output_file = self.opts.output_dir + stem + '.off'
                    write_mesh(output_file, mesh)
                    print("Done! result is saved in: ", output_file)
                else:
                    raise Exception('Unsupported Output file format! Only [ply, obj, off] is supported!')
            else:
                raise Exception('Unsupported Input file format! Only [ply, obj, off, stl] is supported!')

    def mesh_to_pc(self):
        """
        3d mesh转点云
        :return:
        """
        t1 = time()
        for i in range(self.filenum):
            print("Processing: ", self.filelist[i])
            file_path = self.filelist[i]  # 要处理的文件
            input_pc_format = self.opts.input_format
            output_pc_format = self.opts.output_format
            sampler = self.opts.sampler
            num_of_points = self.opts.point_num
            factors = self.opts.factor
            # 检查输入输出文件格式是否合法
            assert input_pc_format in ["off", "ply", "obj", "stl"]
            assert output_pc_format in ["ply", "xyz", "pcd"]
            # mesh采样
            if input_pc_format in ["off", "ply", "obj", "stl"]:
                # 获取文件名
                stem = Path(file_path).stem
                # 输出文件夹不存在则创建
                pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                output_file = self.opts.output_dir + stem + "." + output_pc_format
                mesh = o3d.io.read_triangle_mesh(file_path)
                if sampler == "poisson_disk_sampling":
                    sampling_pc = o3d.geometry.TriangleMesh.sample_points_poisson_disk(mesh, num_of_points, factors)
                    o3d.io.write_point_cloud(output_file, sampling_pc)
                elif sampler == "uniform_sampling":
                    sampling_pc = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, num_of_points)
                    o3d.io.write_point_cloud(output_file, sampling_pc)
                else:
                    raise Exception("Unsupported down sampler! Only [poisson_disk_sampling, uniform_sampling] are "
                                    "supported!")
            else:
                raise Exception("Unsupported input format!")

        print(f"Finished! Results are saved in {self.opts.output_dir}, Time Cost: {time()-t1} s")

    def filter(self):
        """
        点云/mesh 滤波
        :return: 滤波之后的点云/mesh
        """
        for i in range(self.filenum):
            print("Processing: ", self.filelist[i])
            file_path = self.filelist[i]  # 要处理的文件
            input_pc_format = self.opts.input_format
            filters = self.opts.filter
            # 检查输入输出文件格式是否合法
            assert input_pc_format in ["xyz", "pcd", "pts", "ply", "txt", "off", "obj", "stl"]
            # ["xyz", "pcd", "ply"]
            if self.opts.type == "pc":
                assert input_pc_format in ["xyz", "pcd", "pts", "ply", "txt"]
                if input_pc_format in ["xyz", "pcd", "ply"]:
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    # 以原格式输出
                    output_file = self.opts.output_dir + stem + '.' + input_pc_format
                    pc = o3d.io.read_point_cloud(file_path)
                    xyz = np.asarray(pc.points, dtype=np.float32)
                    # print(xyz)
                    if filters == "PassThroughFilter":
                        # You need set suitable upper limit value of passThroughFilter, or you will get 0 point.
                        upper_limit = self.opts.upper_limit
                        filter_pc = passThroughFilter(xyz, upper_limit)
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(filter_pc)
                        o3d.io.write_point_cloud(output_file, pcd)
                    elif filters == "VoxelGridFilter":
                        # You need set suitable voxel size of passThroughFilter
                        voxel_size = self.opts.voxel_size
                        filter_pc = voxelGrid(xyz, voxel_size)
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(filter_pc)
                        o3d.io.write_point_cloud(output_file, pcd)
                    elif filters == "project_inliers":
                        filter_pc = project_inliers(xyz)
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(filter_pc)
                        o3d.io.write_point_cloud(output_file, pcd)
                    elif filters == "remove_outliers":
                        choices = self.opts.removal
                        radius = self.opts.radius
                        min_neighbor = self.opts.min_neighbor
                        # Maybe you should set suitable radius and min_neighbor
                        filter_pc = remove_outlier(xyz, choices, radius, min_neighbor)
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(filter_pc)
                        o3d.io.write_point_cloud(output_file, pcd)
                    elif filters == "statistical_removal":
                        std_dev = self.opts.std_dev
                        # Maybe you should set suitable std_dev
                        filter_pc = StatisticalOutlierRemovalFilter(xyz, std_dev)
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(filter_pc)
                        o3d.io.write_point_cloud(output_file, pcd)
                    else:
                        raise Exception("Unsupported filter! Choices = [PassThroughFilter, VoxelGridFilter, "
                                        "project_inliers, remove_outliers, statistical_removal]")
                    print("Done! Filtered point cloud is saved in: ", output_file)

                # pts
                elif input_pc_format == "pts":
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        first_line = lines[0].split(' ')
                        try:
                            assert len(first_line) == 1
                            # header
                            # 获取文件名
                            stem = Path(file_path).stem
                            # 输出文件夹不存在则创建
                            pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                            # 以原格式输出
                            output_file = self.opts.output_dir + stem + '.' + input_pc_format
                            pc = o3d.io.read_point_cloud(file_path)
                            xyz = np.asarray(pc.points, dtype=np.float32)
                            if filters == "PassThroughFilter":
                                # You need set suitable upper limit value of passThroughFilter, or you will get 0 point.
                                upper_limit = self.opts.upper_limit
                                filter_pc = passThroughFilter(xyz, upper_limit)
                                pcd = o3d.geometry.PointCloud()
                                pcd.points = o3d.utility.Vector3dVector(filter_pc)
                                o3d.io.write_point_cloud(output_file, pcd)
                            elif filters == "VoxelGridFilter":
                                # You need set suitable voxel size of passThroughFilter
                                voxel_size = self.opts.voxel_size
                                filter_pc = voxelGrid(xyz, voxel_size)
                                pcd = o3d.geometry.PointCloud()
                                pcd.points = o3d.utility.Vector3dVector(filter_pc)
                                o3d.io.write_point_cloud(output_file, pcd)
                            elif filters == "project_inliers":
                                filter_pc = project_inliers(xyz)
                                pcd = o3d.geometry.PointCloud()
                                pcd.points = o3d.utility.Vector3dVector(filter_pc)
                                o3d.io.write_point_cloud(output_file, pcd)
                            elif filters == "remove_outliers":
                                choices = self.opts.removal
                                radius = self.opts.radius
                                min_neighbor = self.opts.min_neighbor
                                # Maybe you should set suitable radius and min_neighbor
                                filter_pc = remove_outlier(xyz, choices, radius, min_neighbor)
                                pcd = o3d.geometry.PointCloud()
                                pcd.points = o3d.utility.Vector3dVector(filter_pc)
                                o3d.io.write_point_cloud(output_file, pcd)
                            elif filters == "statistical_removal":
                                std_dev = self.opts.std_dev
                                # Maybe you should set suitable std_dev
                                filter_pc = StatisticalOutlierRemovalFilter(xyz, std_dev)
                                pcd = o3d.geometry.PointCloud()
                                pcd.points = o3d.utility.Vector3dVector(filter_pc)
                                o3d.io.write_point_cloud(output_file, pcd)
                            else:
                                raise Exception("Unsupported filter! Choices = [PassThroughFilter, VoxelGridFilter, "
                                                "project_inliers, remove_outliers, statistical_removal]")
                            print("Done! Filtered point cloud is saved in: ", output_file)
                        except:
                            # no header
                            pc = np.loadtxt(file_path, delimiter=' ', dtype=np.float32)
                            # 获取文件名
                            stem = Path(file_path).stem
                            # 输出文件夹不存在则创建
                            pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                            # 以原格式输出
                            output_file = self.opts.output_dir + stem + '.' + input_pc_format
                            if filters == "PassThroughFilter":
                                # You need set suitable upper limit value of passThroughFilter, or you will get 0 point.
                                upper_limit = self.opts.upper_limit
                                filter_pc = passThroughFilter(pc, upper_limit)
                                pcd = o3d.geometry.PointCloud()
                                pcd.points = o3d.utility.Vector3dVector(filter_pc)
                                o3d.io.write_point_cloud(output_file, pcd)
                            elif filters == "VoxelGridFilter":
                                # You need set suitable voxel size of passThroughFilter
                                voxel_size = self.opts.voxel_size
                                filter_pc = voxelGrid(pc, voxel_size)
                                pcd = o3d.geometry.PointCloud()
                                pcd.points = o3d.utility.Vector3dVector(filter_pc)
                                o3d.io.write_point_cloud(output_file, pcd)
                            elif filters == "project_inliers":
                                filter_pc = project_inliers(pc)
                                pcd = o3d.geometry.PointCloud()
                                pcd.points = o3d.utility.Vector3dVector(filter_pc)
                                o3d.io.write_point_cloud(output_file, pcd)
                            elif filters == "remove_outliers":
                                choices = self.opts.removal
                                radius = self.opts.radius
                                min_neighbor = self.opts.min_neighbor
                                # Maybe you should set suitable radius and min_neighbor
                                filter_pc = remove_outlier(pc, choices, radius, min_neighbor)
                                pcd = o3d.geometry.PointCloud()
                                pcd.points = o3d.utility.Vector3dVector(filter_pc)
                                o3d.io.write_point_cloud(output_file, pcd)
                            elif filters == "statistical_removal":
                                std_dev = self.opts.std_dev
                                # Maybe you should set suitable std_dev
                                filter_pc = StatisticalOutlierRemovalFilter(pc, std_dev)
                                pcd = o3d.geometry.PointCloud()
                                pcd.points = o3d.utility.Vector3dVector(filter_pc)
                                o3d.io.write_point_cloud(output_file, pcd)
                            else:
                                raise Exception("Unsupported filter! Choices = [PassThroughFilter, VoxelGridFilter, "
                                                "project_inliers, remove_outliers, statistical_removal]")
                            print("Done! Filtered point cloud is saved in: ", output_file)
                # txt
                elif input_pc_format == "txt":
                    pc = np.loadtxt(file_path, delimiter=' ', dtype=np.float32)[:, :3]
                    # 获取文件名
                    stem = Path(file_path).stem
                    # 输出文件夹不存在则创建
                    pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                    # 以原格式输出
                    output_file = self.opts.output_dir + stem + '.' + input_pc_format
                    if filters == "PassThroughFilter":
                        # You need set suitable upper limit value of passThroughFilter, or you will get 0 point.
                        upper_limit = self.opts.upper_limit
                        filter_pc = passThroughFilter(pc, upper_limit)
                        np.savetxt(output_file, filter_pc)
                    elif filters == "VoxelGridFilter":
                        # You need set suitable voxel size of passThroughFilter
                        voxel_size = self.opts.voxel_size
                        filter_pc = voxelGrid(pc, voxel_size)
                        np.savetxt(output_file, filter_pc)
                    elif filters == "project_inliers":
                        filter_pc = project_inliers(pc)
                        np.savetxt(output_file, filter_pc)
                    elif filters == "remove_outliers":
                        choices = self.opts.removal
                        radius = self.opts.radius
                        min_neighbor = self.opts.min_neighbor
                        # Maybe you should set suitable radius and min_neighbor
                        filter_pc = remove_outlier(pc, choices, radius, min_neighbor)
                        np.savetxt(output_file, filter_pc)
                    elif filters == "statistical_removal":
                        std_dev = self.opts.std_dev
                        # Maybe you should set suitable std_dev
                        filter_pc = StatisticalOutlierRemovalFilter(pc, std_dev)
                        np.savetxt(output_file, filter_pc)
                    else:
                        raise Exception("Unsupported filter! Choices = [PassThroughFilter, VoxelGridFilter, "
                                        "project_inliers, remove_outliers, statistical_removal]")
                    print("Done! Filtered point cloud is saved in: ", output_file)
                else:
                    raise Exception("Unsupport input file format. Choices=[xyz, pcd, pts, ply, txt]")

            elif self.opts.type == "mesh":
                pass

    def pc_sample(self):
        """
        点云采样（下采样+上采样）
        :return:
        """

    def pc_voxel(self):
        """
        点云体素化
        :return:
        """


if __name__ == "__main__":
    # 获取文件列表
    if platform.system() == "Windows":
        fl = get_all_files(FLAGS.input_dir, FLAGS.input_format)
        # print(fl)
        formatFactory = PointCloud_FormatFactory(FLAGS, fl)
        if FLAGS.mode == 0:
            # 点云格式转换
            formatFactory.pc_pc()
        elif FLAGS.mode == 1:
            # mesh格式转换
            formatFactory.mesh_mesh()
        elif FLAGS.mode == 2:
            # mesh转点云
            formatFactory.mesh_to_pc()
        elif FLAGS.mode == 3:
            # 滤波
            formatFactory.filter()

