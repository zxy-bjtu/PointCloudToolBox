"""
This script is used for point cloud processing.
The common pc format: .pts, .las, .pcd, .xyz, .ply, .txt
"""
import os
import sys
import platform

import open3d as o3d
from pathlib import Path
import pathlib
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../common'))
from configs import FLAGS
from pc_io import get_all_files
from read_las import read_las
from filter import passThroughFilter, voxelGrid, project_inliers, remove_outlier, StatisticalOutlierRemovalFilter
from fps import farthest_point_sample, index_points
from mls import MovingLeastSquares


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

    def filter(self):
        """
        点云滤波
        :return: 滤波之后的点云
        """
        for i in range(self.filenum):
            print("Processing: ", self.filelist[i])
            file_path = self.filelist[i]  # 要处理的文件
            input_pc_format = self.opts.input_format
            filters = self.opts.filter
            # 检查输入输出文件格式是否合法
            # ["xyz", "pcd", "ply"]
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
                            # You need set suitable voxel size of VoxelGridFilter
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
                raise Exception("Unsupported input file format. Choices=[xyz, pcd, pts, ply, txt]")

    def pc_downsample(self):
        """
        点云下采样
        :return:
        """
        for i in range(self.filenum):
            print("Processing: ", self.filelist[i])
            file_path = self.filelist[i]  # 要处理的文件
            input_pc_format = self.opts.input_format
            down_sampler = self.opts.down_sampler
            point_num = self.opts.point_num
            # 获取文件名
            stem = Path(file_path).stem
            # 输出文件夹不存在则创建
            pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
            # 下采样后以原格式输出
            output_file = self.opts.output_dir + stem + '.' + input_pc_format
            # xyz, pcd, ply
            if input_pc_format in ["xyz", "pcd", "ply"]:
                pc = o3d.io.read_point_cloud(file_path)
                xyz = np.asarray(pc.points, dtype=np.float32)
                # farthest point sampling used in PointNet++
                if down_sampler == "fps":
                    if point_num > xyz.shape[0]:
                        raise Exception("The point num cannot be greater than the actual number of points.")
                    else:
                        xyz = xyz[np.newaxis, :, :]
                        xyz = torch.from_numpy(xyz)
                        centroids = farthest_point_sample(xyz, point_num)
                        new_points = index_points(xyz, centroids)
                        ds_points = new_points.numpy()[0]
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(ds_points)
                        o3d.io.write_point_cloud(output_file, pcd)
                        print("Done! result is saved in: ", output_file)
                elif down_sampler == "random":
                    if point_num > xyz.shape[0]:
                        raise Exception("The point num cannot be greater than the actual number of points.")
                    else:
                        sampling_ratio = point_num * 1.0 / xyz.shape[0]
                        pc = o3d.geometry.PointCloud.random_down_sample(pc, sampling_ratio)
                        o3d.io.write_point_cloud(output_file, pc)
                        print("Done! result is saved in: ", output_file)
                elif down_sampler == "uniform":
                    k = self.opts.k
                    pc = o3d.geometry.PointCloud.uniform_down_sample(pc, k)
                    o3d.io.write_point_cloud(output_file, pc)
                    print("Done! result is saved in: ", output_file)
                elif down_sampler == "voxel":
                    voxel_size = self.opts.voxel_size
                    pc = o3d.geometry.PointCloud.voxel_down_sample(pc, voxel_size)
                    o3d.io.write_point_cloud(output_file, pc)
                    print("Done! result is saved in: ", output_file)
                else:
                    raise Exception("Unsupported down sampler. Choices=[fps, random, uniform, voxel]")

            # txt
            elif input_pc_format == "txt":
                pc = np.loadtxt(file_path, delimiter=' ')[:, :3]
                xyz = pc[np.newaxis, :, :]
                # farthest point sampling used in PointNet++
                if down_sampler == "fps":
                    if point_num > pc.shape[0]:
                        raise Exception("The point num cannot be greater than the actual number of points.")
                    else:
                        xyz = torch.from_numpy(xyz)
                        centroids = farthest_point_sample(xyz, point_num)
                        new_points = index_points(xyz, centroids)
                        ds_points = new_points.numpy()[0]
                        np.savetxt(output_file, ds_points)
                        print("Done! result is saved in: ", output_file)
                elif down_sampler == "random":
                    if point_num > pc.shape[0]:
                        raise Exception("The point num cannot be greater than the actual number of points.")
                    else:
                        sampling_ratio = point_num * 1.0 / pc.shape[0]
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(pc)
                        pc = o3d.geometry.PointCloud.random_down_sample(pcd, sampling_ratio)
                        ds_points = np.asarray(pc.points, dtype=np.float32)
                        np.savetxt(output_file, ds_points)
                        print("Done! result is saved in: ", output_file)
                elif down_sampler == "uniform":
                    k = self.opts.k
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pc)
                    pc = o3d.geometry.PointCloud.uniform_down_sample(pcd, k)
                    ds_points = np.asarray(pc.points, dtype=np.float32)
                    np.savetxt(output_file, ds_points)
                    print("Done! result is saved in: ", output_file)
                elif down_sampler == "voxel":
                    voxel_size = self.opts.voxel_size
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pc)
                    pc = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size)
                    ds_points = np.asarray(pc.points, dtype=np.float32)
                    np.savetxt(output_file, ds_points)
                    print("Done! result is saved in: ", output_file)
                else:
                    raise Exception("Unsupported down sampler. Choices=[fps, random, uniform, voxel]")

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
                        # farthest point sampling used in PointNet++
                        if down_sampler == "fps":
                            if point_num > xyz.shape[0]:
                                raise Exception("The point num cannot be greater than the actual number of points.")
                            else:
                                xyz = xyz[np.newaxis, :, :]
                                xyz = torch.from_numpy(xyz)
                                centroids = farthest_point_sample(xyz, point_num)
                                new_points = index_points(xyz, centroids)
                                ds_points = new_points.numpy()[0]
                                pcd = o3d.geometry.PointCloud()
                                pcd.points = o3d.utility.Vector3dVector(ds_points)
                                o3d.io.write_point_cloud(output_file, pcd)
                                print("Done! result is saved in: ", output_file)
                        elif down_sampler == "random":
                            if point_num > xyz.shape[0]:
                                raise Exception("The point num cannot be greater than the actual number of points.")
                            else:
                                sampling_ratio = point_num * 1.0 / xyz.shape[0]
                                pc = o3d.geometry.PointCloud.random_down_sample(pc, sampling_ratio)
                                o3d.io.write_point_cloud(output_file, pc)
                                print("Done! result is saved in: ", output_file)
                        elif down_sampler == "uniform":
                            k = self.opts.k
                            pc = o3d.geometry.PointCloud.uniform_down_sample(pc, k)
                            o3d.io.write_point_cloud(output_file, pc)
                            print("Done! result is saved in: ", output_file)
                        elif down_sampler == "voxel":
                            voxel_size = self.opts.voxel_size
                            pc = o3d.geometry.PointCloud.voxel_down_sample(pc, voxel_size)
                            o3d.io.write_point_cloud(output_file, pc)
                            print("Done! result is saved in: ", output_file)
                        else:
                            raise Exception("Unsupported down sampler. Choices=[fps, random, uniform, voxel]")

                    except:
                        # no header
                        pc = np.loadtxt(file_path, delimiter=' ', dtype=np.float32)
                        # 获取文件名
                        stem = Path(file_path).stem
                        # 输出文件夹不存在则创建
                        pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                        # 以原格式输出
                        output_file = self.opts.output_dir + stem + '.' + input_pc_format
                        xyz = pc[np.newaxis, :, :]
                        # farthest point sampling used in PointNet++
                        if down_sampler == "fps":
                            if point_num > pc.shape[0]:
                                raise Exception("The point num cannot be greater than the actual number of points.")
                            else:
                                xyz = torch.from_numpy(xyz)
                                centroids = farthest_point_sample(xyz, point_num)
                                new_points = index_points(xyz, centroids)
                                ds_points = new_points.numpy()[0]
                                pcd = o3d.geometry.PointCloud()
                                pcd.points = o3d.utility.Vector3dVector(ds_points)
                                o3d.io.write_point_cloud(output_file, pcd)
                                print("Done! result is saved in: ", output_file)
                        elif down_sampler == "random":
                            if point_num > pc.shape[0]:
                                raise Exception("The point num cannot be greater than the actual number of points.")
                            else:
                                sampling_ratio = point_num * 1.0 / pc.shape[0]
                                pcd = o3d.geometry.PointCloud()
                                pcd.points = o3d.utility.Vector3dVector(pc)
                                pc = o3d.geometry.PointCloud.random_down_sample(pcd, sampling_ratio)
                                o3d.io.write_point_cloud(output_file, pc)
                                print("Done! result is saved in: ", output_file)
                        elif down_sampler == "uniform":
                            k = self.opts.k
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(pc)
                            pc = o3d.geometry.PointCloud.uniform_down_sample(pcd, k)
                            o3d.io.write_point_cloud(output_file, pc)
                            print("Done! result is saved in: ", output_file)
                        elif down_sampler == "voxel":
                            voxel_size = self.opts.voxel_size
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(pc)
                            pc = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size)
                            o3d.io.write_point_cloud(output_file, pc)
                            print("Done! result is saved in: ", output_file)
                        else:
                            raise Exception("Unsupported down sampler. Choices=[fps, random, uniform, voxel]")

            else:
                raise Exception("Unsupported input format! Choices=[ply, xyz, pts, pcd, txt]")

    def pc_upsampling(self):
        """
        点云上采样
        支持的上采样模型：
        1. Meta-PU
            支持的采样率：任意的浮点数，如：5.5
            支持的运行系统：Linux
            支持的点云格式：仅xyz
        :return:
        """
        for i in range(self.filenum):
            print("Processing: ", self.filelist[i])
            file_path = self.filelist[i]  # 要处理的文件
            input_pc_format = self.opts.input_format
            pu_model = self.opts.pu_model
            scale_R = self.opts.scale
            if pu_model == "Meta-PU":
                # 检查系统
                assert platform.system() == "Linux"
                # 检查输入输出文件格式是否合法
                assert input_pc_format in ["xyz"]
                os.system('cd ../PU/Meta-PU')
                # model/data/all_testset/4/input : the path of the point cloud to be sampled
                # result is saved in /model/new/result/${R}input/
                os.system('python main_gan.py --phase test --dataset model/data/all_testset/4/input --log_dir '
                          'model/new --batch_size 4 --model model_res_mesh_pool --model_path 60 --gpu 0 '
                          '--test_scale ' + str(scale_R))
            else:
                raise Exception("unsupported PU model.")

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
        elif FLAGS.mode == 3:
            # 点云滤波
            formatFactory.filter()
        elif FLAGS.mode == 5:
            # 点云下采样
            formatFactory.pc_downsample()

    elif platform.system() == "Linux":
        fl = get_all_files(FLAGS.input_dir, FLAGS.input_format)
        # print(fl)
        formatFactory = PointCloud_FormatFactory(FLAGS, fl)
        if FLAGS.mode == 0:
            # 点云格式转换
            formatFactory.pc_pc()
        elif FLAGS.mode == 3:
            # 点云滤波
            formatFactory.filter()
        elif FLAGS.mode == 5:
            # 点云下采样
            formatFactory.pc_downsample()
        elif FLAGS.mode == 9:
            # 点云上采样
            formatFactory.pc_upsampling()

    else:
        raise Exception("Unsupported Operating System!")
