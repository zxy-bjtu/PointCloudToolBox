"""
This script is used for calculating surfaceArea and Volume
vtk: 8.2.0
"""
import vtk
import os
import sys
import open3d as o3d
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../common'))
from configs import FLAGS
from pc_io import get_all_files


def calculate_surfaceArea_and_Volume(fl):
    """
    计算点云的表面积和体积
    :param input_pointcloud:
    :return:
    """
    for f in fl:
        # print(f)
        if FLAGS.input_format == "ply":
            vtkReader = vtk.vtkPLYReader()
            # print(vtkReader)
            vtkReader.SetFileName(f)
            vtkReader.Update()
            polydata = vtkReader.GetOutput()
            mass = vtk.vtkMassProperties()
            mass.SetInputData(polydata)
            print("Processing: ", f)
            print("[Mesh Surface Area] S=", mass.GetSurfaceArea(), "(square units)")
            print("[Mesh Volume] V=", mass.GetVolume(), "(cube units)")
        elif FLAGS.input_format == "obj":
            vtkReader = vtk.vtkOBJReader()
            vtkReader.SetFileName(f)
            vtkReader.Update()
            triangleFilter = vtk.vtkTriangleFilter()
            triangleFilter.SetInputData(vtkReader.GetOutput())
            triangleFilter.Update()
            polygonProperties = vtk.vtkMassProperties()
            polygonProperties.SetInputData(triangleFilter.GetOutput())
            polygonProperties.Update()
            print("Processing: ", f)
            print("[Mesh Surface Area] S=", polygonProperties.GetSurfaceArea(), "(square units)")
            print("[Mesh Volume] V=", polygonProperties.GetVolume(), "(cube units)")
        elif FLAGS.input_format == "off":
            mesh = o3d.io.read_triangle_mesh(f)
            mesh.compute_vertex_normals()
            print("Processing: ", f)
            surface_area = o3d.geometry.TriangleMesh.get_surface_area(mesh)
            print("[Mesh Surface Area] S=", surface_area, "(square units)")
            try:
                volume = o3d.geometry.TriangleMesh.get_volume(mesh)
                print("[Mesh Volume] V=", volume, "(cube units)")
            except:
                continue
        elif FLAGS.input_format == "stl":
            mesh = o3d.io.read_triangle_mesh(f)
            mesh.compute_vertex_normals()
            print("Processing: ", f)
            surface_area = o3d.geometry.TriangleMesh.get_surface_area(mesh)
            print("[Mesh Surface Area] S=", surface_area, "(square units)")
            try:
                volume = o3d.geometry.TriangleMesh.get_volume(mesh)
                print("[Mesh Volume] V=", volume, "(cube units)")
            except:
                continue
        else:
            raise Exception("Unsupported point cloud format! Only [ply, obj, off, stl] is supported!")


if __name__ == "__main__":
    # 获取文件列表
    fl = get_all_files(FLAGS.input_dir, FLAGS.input_format)
    calculate_surfaceArea_and_Volume(fl)

