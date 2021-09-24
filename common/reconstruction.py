import numpy as np
import os, sys
import shutil
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from pathlib import Path
import pathlib
import open3d as o3d
from openmesh import read_polymesh, write_mesh


def poisson_surface_reconstruction(file_path, input_format, output_format, output_file, depth):
    """
    点云泊松表面重建
    reference: M.Kazhdan and M. Bolitho and H. Hoppe: Poisson surface reconstruction, Eurographics, 2006.
    :param file_path: 输入点云文件
    :param input_format: 输入的点云格式 ["pcd", "xyz", "pts", "txt"]
    :param output_format: 输出的mesh格式 ["off", "ply", "obj", "stl"]
    :param output_file: 输出的mesh文件
    :param depth: poisson重建的深度
    :return:
    """
    if input_format in ["pcd", "xyz"]:
        pc = o3d.io.read_point_cloud(file_path)
        # normal estimation
        pc.estimate_normals()
        print(":: Run Poisson surface reconstruction")
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=depth)
            if output_format == "off":
                # 获取文件名
                stem = Path(file_path).stem
                tmp_dir = "../result/tmp/"
                pathlib.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
                tmp_output_file = tmp_dir + stem + '.ply'
                o3d.io.write_triangle_mesh(tmp_output_file, mesh)
                # ply->off
                mesh = read_polymesh(tmp_output_file)
                write_mesh(output_file, mesh)
                print("Done! result is saved in: ", output_file)
                if os.path.exists(tmp_dir):
                    shutil.rmtree(tmp_dir)
            elif output_format in ["ply", "obj"]:
                o3d.io.write_triangle_mesh(output_file, mesh)
                print("Done! result is saved in: ", output_file)
            elif output_format == "stl":
                mesh.compute_vertex_normals()
                o3d.io.write_triangle_mesh(output_file, mesh)
                print("Done! result is saved in: ", output_file)
        print("::::::::::::::::::::::::::::::::::::::::::::::::")
    elif input_format == "txt":
        pc = np.loadtxt(file_path, delimiter=' ', dtype=np.float32)[:, :3]
        # normal estimation
        # 创建open3d对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.estimate_normals()
        print(":: Run Poisson surface reconstruction")
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
            if output_format == "off":
                # 获取文件名
                stem = Path(file_path).stem
                tmp_dir = "../result/tmp/"
                pathlib.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
                tmp_output_file = tmp_dir + stem + '.ply'
                o3d.io.write_triangle_mesh(tmp_output_file, mesh)
                # ply->off
                mesh = read_polymesh(tmp_output_file)
                write_mesh(output_file, mesh)
                print("Done! result is saved in: ", output_file)
                if os.path.exists(tmp_dir):
                    shutil.rmtree(tmp_dir)
            elif output_format in ["ply", "obj"]:
                o3d.io.write_triangle_mesh(output_file, mesh)
                print("Done! result is saved in: ", output_file)
            elif output_format == "stl":
                mesh.compute_vertex_normals()
                o3d.io.write_triangle_mesh(output_file, mesh)
                print("Done! result is saved in: ", output_file)
        print("::::::::::::::::::::::::::::::::::::::::::::::::")
    elif input_format == "pts":
        # pts->xyz
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            first_line = lines[0].split(' ')
            try:
                assert len(first_line) == 1
                # header
                pc = o3d.io.read_point_cloud(file_path)
                # normal estimation
                pc.estimate_normals()
                print(":: Run Poisson surface reconstruction")
                with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=depth)
                    if output_format == "off":
                        # 获取文件名
                        stem = Path(file_path).stem
                        tmp_dir = "../result/tmp/"
                        pathlib.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
                        tmp_output_file = tmp_dir + stem + '.ply'
                        o3d.io.write_triangle_mesh(tmp_output_file, mesh)
                        # ply->off
                        mesh = read_polymesh(tmp_output_file)
                        write_mesh(output_file, mesh)
                        print("Done! result is saved in: ", output_file)
                        if os.path.exists(tmp_dir):
                            shutil.rmtree(tmp_dir)
                    elif output_format in ["ply", "obj"]:
                        o3d.io.write_triangle_mesh(output_file, mesh)
                        print("Done! result is saved in: ", output_file)
                    elif output_format == "stl":
                        mesh.compute_vertex_normals()
                        o3d.io.write_triangle_mesh(output_file, mesh)
                        print("Done! result is saved in: ", output_file)
                print("::::::::::::::::::::::::::::::::::::::::::::::::")
            except:
                # no header
                pc = np.loadtxt(file_path, delimiter=' ')
                # normal estimation
                # 创建open3d对象
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc)
                pcd.estimate_normals()
                print(":: Run Poisson surface reconstruction")
                with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
                    if output_format == "off":
                        # 获取文件名
                        stem = Path(file_path).stem
                        tmp_dir = "../result/tmp/"
                        pathlib.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
                        tmp_output_file = tmp_dir + stem + '.ply'
                        o3d.io.write_triangle_mesh(tmp_output_file, mesh)
                        # ply->off
                        mesh = read_polymesh(tmp_output_file)
                        write_mesh(output_file, mesh)
                        print("Done! result is saved in: ", output_file)
                        if os.path.exists(tmp_dir):
                            shutil.rmtree(tmp_dir)
                    elif output_format in ["ply", "obj"]:
                        o3d.io.write_triangle_mesh(output_file, mesh)
                        print("Done! result is saved in: ", output_file)
                    elif output_format == "stl":
                        mesh.compute_vertex_normals()
                        o3d.io.write_triangle_mesh(output_file, mesh)
                        print("Done! result is saved in: ", output_file)
                print("::::::::::::::::::::::::::::::::::::::::::::::::")


def ball_pivot_surface_reconstruction(file_path, input_format, output_format, output_file):
    """
    ball pivoting algorithm (BPA) [Bernardini1999]
    reference: The ball-pivoting algorithm for surface reconstruction
    :param filepath: 输入点云文件
    :param input_format: 输入的点云格式
    :param output_format: 输出的mesh格式
    :param output_file: 输出的mesh文件
    :return:
    """
    radii = [0.005, 0.01, 0.02, 0.04]
    if input_format in ["pcd", "xyz"]:
        pc = o3d.io.read_point_cloud(file_path)
        # normal estimation
        pc.estimate_normals()
        print(":: Run ball pivoting reconstruction")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pc, o3d.utility.DoubleVector(radii))
        if output_format == "off":
            # 获取文件名
            stem = Path(file_path).stem
            tmp_dir = "../result/tmp/"
            pathlib.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
            tmp_output_file = tmp_dir + stem + '.ply'
            o3d.io.write_triangle_mesh(tmp_output_file, mesh)
            # ply->off
            mesh = read_polymesh(tmp_output_file)
            write_mesh(output_file, mesh)
            print("Done! result is saved in: ", output_file)
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
        elif output_format in ["ply", "obj"]:
            o3d.io.write_triangle_mesh(output_file, mesh)
            print("Done! result is saved in: ", output_file)
        elif output_format == "stl":
            mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(output_file, mesh)
            print("Done! result is saved in: ", output_file)
        print("::::::::::::::::::::::::::::::::::::::::::::::::")

    elif input_format == "txt":
        pc = np.loadtxt(file_path, delimiter=' ', dtype=np.float32)[:, :3]
        # normal estimation
        # 创建open3d对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.estimate_normals()
        print(":: Run ball pivoting reconstruction")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
        if output_format == "off":
            # 获取文件名
            stem = Path(file_path).stem
            tmp_dir = "../result/tmp/"
            pathlib.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
            tmp_output_file = tmp_dir + stem + '.ply'
            o3d.io.write_triangle_mesh(tmp_output_file, mesh)
            # ply->off
            mesh = read_polymesh(tmp_output_file)
            write_mesh(output_file, mesh)
            print("Done! result is saved in: ", output_file)
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
        elif output_format in ["ply", "obj"]:
            o3d.io.write_triangle_mesh(output_file, mesh)
            print("Done! result is saved in: ", output_file)
        elif output_format == "stl":
            mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(output_file, mesh)
            print("Done! result is saved in: ", output_file)
        print("::::::::::::::::::::::::::::::::::::::::::::::::")
    elif input_format == "pts":
        # pts->xyz
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            first_line = lines[0].split(' ')
            try:
                assert len(first_line) == 1
                # header
                pc = o3d.io.read_point_cloud(file_path)
                # normal estimation
                pc.estimate_normals()
                print(":: Run ball pivoting reconstruction")
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pc,o3d.utility.DoubleVector(radii))
                if output_format == "off":
                    # 获取文件名
                    stem = Path(file_path).stem
                    tmp_dir = "../result/tmp/"
                    pathlib.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
                    tmp_output_file = tmp_dir + stem + '.ply'
                    o3d.io.write_triangle_mesh(tmp_output_file, mesh)
                    # ply->off
                    mesh = read_polymesh(tmp_output_file)
                    write_mesh(output_file, mesh)
                    print("Done! result is saved in: ", output_file)
                    if os.path.exists(tmp_dir):
                        shutil.rmtree(tmp_dir)
                elif output_format in ["ply", "obj"]:
                    o3d.io.write_triangle_mesh(output_file, mesh)
                    print("Done! result is saved in: ", output_file)
                elif output_format == "stl":
                    mesh.compute_vertex_normals()
                    o3d.io.write_triangle_mesh(output_file, mesh)
                    print("Done! result is saved in: ", output_file)
                print("::::::::::::::::::::::::::::::::::::::::::::::::")
            except:
                # no header
                pc = np.loadtxt(file_path, delimiter=' ')
                # normal estimation
                # 创建open3d对象
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc)
                pcd.estimate_normals()
                print(":: Run ball pivoting reconstruction")
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,
                                                                                       o3d.utility.DoubleVector(radii))
                if output_format == "off":
                    # 获取文件名
                    stem = Path(file_path).stem
                    tmp_dir = "../result/tmp/"
                    pathlib.Path(tmp_dir).mkdir(parents=True, exist_ok=True)
                    tmp_output_file = tmp_dir + stem + '.ply'
                    o3d.io.write_triangle_mesh(tmp_output_file, mesh)
                    # ply->off
                    mesh = read_polymesh(tmp_output_file)
                    write_mesh(output_file, mesh)
                    print("Done! result is saved in: ", output_file)
                    if os.path.exists(tmp_dir):
                        shutil.rmtree(tmp_dir)
                elif output_format in ["ply", "obj"]:
                    o3d.io.write_triangle_mesh(output_file, mesh)
                    print("Done! result is saved in: ", output_file)
                elif output_format == "stl":
                    mesh.compute_vertex_normals()
                    o3d.io.write_triangle_mesh(output_file, mesh)
                    print("Done! result is saved in: ", output_file)
                print("::::::::::::::::::::::::::::::::::::::::::::::::")
