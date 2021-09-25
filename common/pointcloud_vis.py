"""
使用mayavi可视化点云
"""

import os
import sys
import open3d as o3d
import numpy as np
import mayavi.mlab as mlab
from read_las import read_las

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../lib'))
from configs import FLAGS


def pc_vis(file_path, input_format, scale_factor):
    """
    point cloud visualization
    :param file_path: 输入的点云文件
    :param input_format: 输入的点云格式
    :param scale_factor: 每个点的球径，值越大画出来的点越大，值在0.1附近较为合适
    :return:
    """
    # fgcolor: (0.25, 0.88, 0.81) 青色
    # fgcolor: (0.01, 0.01, 0.01) 黑色
    # fgcolor: (0.8549, 0.64706, 0.12549) 土黄色
    # fgcolor: (0.51373, 0.43529, 1) 浅紫色
    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0.25, 0.88, 0.81))
    if input_format in ["ply", "pcd", "xyz"]:
        pc = o3d.io.read_point_cloud(file_path)
        xyz = np.asarray(pc.points)
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        # colormap:
        # accent, autumn, black-white, blue-red, blues, bone, brbg, bugn, bupu, cool, copper, dark2, flag, gist_earth
        # gist_array, gist_heat, gist_ncar, gist_rainbow, gist_stern, gist_yarg, gnbu, gray, greens, greys, hot, hsv,
        # jet, oranges, orrd, paired, pastel1, pastel2, pink, piyg, prgn, prism, pubu, pubugn, puor, purd, purples,
        # rdbu, rdgy, rdpu, rdylbu, rdylgn, reds, set1, set2, set3, spectral, spring, summer, winter, ylgnbu, ylgn,
        # ylorbr, ylorrd
        # mode:
        # sphere(default)、cube
        mlab.points3d(x, y, z, colormap='spectral', scale_factor=scale_factor, mode='sphere')
        mlab.show()
    elif input_format == "las":
        pc_list = read_las(file_path)
        pc = []
        for i in range(len(pc_list)):
            pc.append(list(pc_list[i]))
        point_xyz = np.array(pc)
        x = point_xyz[:, 0]
        y = point_xyz[:, 1]
        z = point_xyz[:, 2]
        mlab.points3d(x, y, z, colormap='spectral', scale_factor=scale_factor, mode='sphere')
        mlab.show()
    elif input_format == "txt":
        pc = np.loadtxt(file_path, delimiter=' ')[:, :3]
        x = pc[:, 0]
        y = pc[:, 1]
        z = pc[:, 2]
        mlab.points3d(x, y, z, colormap='spectral', scale_factor=scale_factor, mode='sphere')
        mlab.show()
    elif input_format == "pts":
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            first_line = lines[0].split(' ')
            try:
                assert len(first_line) == 1
                # header
                pc = o3d.io.read_point_cloud(file_path)
                xyz = np.asarray(pc.points)
                x = xyz[:, 0]
                y = xyz[:, 1]
                z = xyz[:, 2]
                mlab.points3d(x, y, z, colormap='spectral', scale_factor=scale_factor, mode='sphere')
                mlab.show()
            except:
                pc = np.loadtxt(file_path, delimiter=' ')
                x = pc[:, 0]
                y = pc[:, 1]
                z = pc[:, 2]
                mlab.points3d(x, y, z, colormap='spectral', scale_factor=scale_factor, mode='sphere')
                mlab.show()
    else:
        raise Exception("Unsupported input point cloud format. ")


if __name__ == "__main__":
    if FLAGS.mode == 12:
        file_path = FLAGS.input_file
        ext = os.path.splitext(file_path)[-1][1:]
        scale_factor = FLAGS.scale_factor
        print(":: Visual point cloud {}".format(file_path))
        pc_vis(file_path, ext, scale_factor)
        print(":: finished!")
