"""
使用pyvista可视化网格
"""

import os
import pathlib
import sys
import pyvista as pv
from pyvista import themes

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../lib'))
from configs import FLAGS


def mesh_vis(file_path, input_format, screenshot):
    """
    mesh visualization
    :param file_path: 输入的mesh文件
    :param input_format: 输入的点云格式
    :param screenshot: 截图路径
    :return:
    """
    if input_format in ["ply", "vtk", "stl", "obj", "off", "msh"]:
        mesh = pv.read(file_path)
        my_theme = themes.DefaultTheme()
        my_theme.color = 'black'
        my_theme.lighting = True
        my_theme.show_edges = True
        my_theme.edge_color = 'red'
        my_theme.background = 'white'
        mesh.plot(theme=my_theme, interactive=True, return_viewer=True, screenshot=screenshot)
    else:
        raise Exception("Unsupported mesh format!")


if __name__ == "__main__":
    if FLAGS.mode == 13:
        file_path = FLAGS.input_file
        stem = pathlib.Path(file_path).stem
        ext = os.path.splitext(file_path)[-1][1:]
        screen_shot = FLAGS.screenshot
        pathlib.Path(screen_shot).mkdir(parents=True, exist_ok=True)
        output_file = screen_shot + stem + ".png"
        print(":: Visual 3d mesh {}".format(file_path))
        mesh_vis(file_path, ext, output_file)
        print(":: finished!")
