"""
This script is used for 3D mesh processing.
The common mesh format: .obj, .off, .ply, .stl
"""
import os
import sys
import platform
from time import time

import open3d as o3d
from pathlib import Path
import pathlib
from plyfile import PlyData
from openmesh import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../common'))
from configs import FLAGS
from pc_io import get_all_files


class Mesh_FormatFactory(object):
    def __init__(self, opts, filelist):
        """
        :param opts: 超参
        :param filelist: 要处理的文件列表
        """
        self.opts = opts
        self.filelist = filelist
        self.filenum = len(self.filelist)  # 要处理的文件个数

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
        mesh 滤波
        :return: 滤波之后的mesh
        """
        for i in range(self.filenum):
            print("Processing: ", self.filelist[i])
            file_path = self.filelist[i]  # 要处理的文件
            input_pc_format = self.opts.input_format
            mesh_filters = self.opts.mesh_filter
            # 检查输入输出文件格式是否合法
            assert input_pc_format in ["ply", "obj", "stl"]
            # ply, obj
            if input_pc_format in ["ply", "obj"]:
                # 获取文件名
                stem = Path(file_path).stem
                # 输出文件夹不存在则创建
                pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                # 以原格式输出
                output_file = self.opts.output_dir + stem + '.' + input_pc_format
                mesh = o3d.io.read_triangle_mesh(file_path)
                if mesh_filters == "taubin":
                    taubin_mesh = o3d.geometry.TriangleMesh.filter_smooth_taubin(mesh, number_of_iterations=1, mu=-0.53)
                    o3d.io.write_triangle_mesh(output_file, taubin_mesh)
                    print("Done! result is saved in: ", output_file)
                elif mesh_filters == "laplacian":
                    lap_mesh = o3d.geometry.TriangleMesh.filter_smooth_laplacian(mesh, number_of_iterations=1)
                    o3d.io.write_triangle_mesh(output_file, lap_mesh)
                    print("Done! result is saved in: ", output_file)
                elif mesh_filters == "neighbour":
                    neighbor_mesh = o3d.geometry.TriangleMesh.filter_smooth_simple(mesh, number_of_iterations=1)
                    o3d.io.write_triangle_mesh(output_file, neighbor_mesh)
                    print("Done! result is saved in: ", output_file)
                else:
                    raise Exception("Unsupported mesh filter!")
            # stl
            elif input_pc_format == "stl":
                # 获取文件名
                stem = Path(file_path).stem
                # 输出文件夹不存在则创建
                pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                # 以原格式输出
                output_file = self.opts.output_dir + stem + '.' + input_pc_format
                mesh = o3d.io.read_triangle_mesh(file_path)
                if mesh_filters == "taubin":
                    taubin_mesh = o3d.geometry.TriangleMesh.filter_smooth_taubin(mesh, number_of_iterations=1, mu=-0.53)
                    taubin_mesh.compute_vertex_normals()  # compute normals
                    o3d.io.write_triangle_mesh(output_file, taubin_mesh)
                    print("Done! result is saved in: ", output_file)
                elif mesh_filters == "laplacian":
                    lap_mesh = o3d.geometry.TriangleMesh.filter_smooth_laplacian(mesh, number_of_iterations=1)
                    lap_mesh.compute_vertex_normals()  # compute normals
                    o3d.io.write_triangle_mesh(output_file, lap_mesh)
                    print("Done! result is saved in: ", output_file)
                elif mesh_filters == "neighbour":
                    neighbor_mesh = o3d.geometry.TriangleMesh.filter_smooth_simple(mesh, number_of_iterations=1)
                    neighbor_mesh.compute_vertex_normals()  # compute normals
                    o3d.io.write_triangle_mesh(output_file, neighbor_mesh)
                    print("Done! result is saved in: ", output_file)
                else:
                    raise Exception("Unsupported mesh filter!")
            else:
                raise Exception("Unsupported input file format. Choices=[ply, obj, stl]")

    def subdivision(self):
        """
        3d mesh subdivision: divide each triangle into a number of smaller triangles
        :return:
        """
        for i in range(self.filenum):
            print("Processing: ", self.filelist[i])
            file_path = self.filelist[i]  # 要处理的文件
            input_pc_format = self.opts.input_format
            subdivision_type = self.opts.subdivision_type
            num_of_iteration = self.opts.iteration
            # 检查输入输出文件格式是否合法
            assert input_pc_format in ["ply", "obj", "stl"]
            # ply, obj
            if input_pc_format in ["ply", "obj"]:
                # 获取文件名
                stem = Path(file_path).stem
                # 输出文件夹不存在则创建
                pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                # 以原格式输出
                output_file = self.opts.output_dir + stem + '.' + input_pc_format
                mesh = o3d.io.read_triangle_mesh(file_path)
                if subdivision_type == "loop":
                    loop_mesh = mesh.subdivide_loop(number_of_iterations=num_of_iteration)
                    o3d.io.write_triangle_mesh(output_file, loop_mesh)
                    print("Done! result is saved in: ", output_file)
                elif subdivision_type == "midpoint":
                    midpoint_mesh = mesh.subdivide_midpoint(number_of_iterations=num_of_iteration)
                    o3d.io.write_triangle_mesh(output_file, midpoint_mesh)
                    print("Done! result is saved in: ", output_file)
                else:
                    raise Exception("Unsupported mesh subdivision type!")
            # stl
            elif input_pc_format == "stl":
                # 获取文件名
                stem = Path(file_path).stem
                # 输出文件夹不存在则创建
                pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
                # 以原格式输出
                output_file = self.opts.output_dir + stem + '.' + input_pc_format
                mesh = o3d.io.read_triangle_mesh(file_path)
                if subdivision_type == "loop":
                    loop_mesh = mesh.subdivide_loop(number_of_iterations=num_of_iteration)
                    loop_mesh.compute_vertex_normals()  # compute normals
                    o3d.io.write_triangle_mesh(output_file, loop_mesh)
                    print("Done! result is saved in: ", output_file)
                elif subdivision_type == "midpoint":
                    midpoint_mesh = mesh.subdivide_midpoint(number_of_iterations=num_of_iteration)
                    midpoint_mesh.compute_vertex_normals()  # compute normals
                    o3d.io.write_triangle_mesh(output_file, midpoint_mesh)
                    print("Done! result is saved in: ", output_file)
                else:
                    raise Exception("Unsupported mesh subdivision type!")
            else:
                raise Exception("Unsupported input file format. Choices=[ply, obj, stl]")


if __name__ == "__main__":
    # 获取文件列表
    if platform.system() == "Windows":
        fl = get_all_files(FLAGS.input_dir, FLAGS.input_format)
        # print(fl)
        formatFactory = Mesh_FormatFactory(FLAGS, fl)
        if FLAGS.mode == 1:
            # mesh格式转换
            formatFactory.mesh_mesh()
        elif FLAGS.mode == 2:
            # mesh转点云
            formatFactory.mesh_to_pc()
        elif FLAGS.mode == 4:
            # mesh滤波
            formatFactory.filter()
        elif FLAGS.mode == 6:
            # mesh精细化
            formatFactory.subdivision()

    elif platform.system() == "Linux":
        fl = get_all_files(FLAGS.input_dir, FLAGS.input_format)
        # print(fl)
        formatFactory = Mesh_FormatFactory(FLAGS, fl)
        if FLAGS.mode == 1:
            # mesh格式转换
            formatFactory.mesh_mesh()
        elif FLAGS.mode == 2:
            # mesh转点云
            formatFactory.mesh_to_pc()
        elif FLAGS.mode == 4:
            # mesh滤波
            formatFactory.filter()
        elif FLAGS.mode == 6:
            # mesh精细化
            formatFactory.subdivision()
    else:
        raise Exception("Unsupported Operating System!")


