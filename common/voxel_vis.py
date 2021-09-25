"""
使用viewvox进行体素文件的可视化
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../lib'))
sys.path.append(os.path.join(BASE_DIR, '../vox/linux'))
sys.path.append(os.path.join(BASE_DIR, '../vox/windows'))
from configs import FLAGS


def voxel_visualzation(file_path, input_format):
    """
    voxel grid visualization, supported format:["binvox", "mira"]
    :return:
    """
    assert input_format in ["binvox", "mira"]
    os.system('viewvox -ki ' + file_path)


if __name__ == "__main__":
    if FLAGS.mode == 8:
        file_path = FLAGS.input_file
        ext = os.path.splitext(file_path)[-1][1:]
        print(":: Visual voxel {}".format(file_path))
        voxel_visualzation(file_path, ext)
        print(":: finished!")
