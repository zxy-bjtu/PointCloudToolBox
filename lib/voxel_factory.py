"""
This script is used for voxel processing.
The common voxel format: .binvox, .vxl, .vox, .kvx, .kv6, .v3a, .v3b
"""
import os
import sys
import platform
from pathlib import Path
import pathlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../common'))
sys.path.append(os.path.join(BASE_DIR, '../vox/linux'))
sys.path.append(os.path.join(BASE_DIR, '../vox/windows'))
from configs import FLAGS
from pc_io import get_all_files


class Voxel_FormatFactory(object):
    def __init__(self, opts, filelist):
        """
        :param opts: 超参
        :param filelist: 要处理的文件列表
        """
        self.opts = opts
        self.filelist = filelist
        self.filenum = len(self.filelist)  # 要处理的文件个数

    def mesh_voxel(self):
        """
        转换mesh成3d体素网格
        :return:
        """
        for i in range(self.filenum):
            print("Processing: ", self.filelist[i])
            file_path = self.filelist[i]  # 要处理的文件
            input_pc_format = self.opts.input_format
            output_pc_format = self.opts.output_format
            d = self.opts.d  # voxel size
            assert input_pc_format in ["obj", "off", "dxf", "ply", "stl"]
            assert output_pc_format in ["binvox", "hips", "mira", "vtk", "msh", "nrrd"]

            # 获取文件名
            stem = Path(file_path).stem
            # 输出文件夹不存在则创建
            pathlib.Path(self.opts.output_dir).mkdir(parents=True, exist_ok=True)
            output_file_path = self.opts.output_dir + stem + '.' + output_pc_format

            if platform.system() == "Windows":
                os.system('binvox -c -d ' + str(d) + ' -t ' + output_pc_format+' '+file_path)
                os.system('mv ' + self.opts.input_dir + '/'+stem+'.'+output_pc_format + ' '+output_file_path)
            elif platform.system() == "Linux":
                os.system('binvox -c -d ' + str(d) + ' -t ' + output_pc_format+' '+file_path)
                os.system('mv ' + self.opts.input_dir + '/'+stem+'.'+output_pc_format + ' '+output_file_path)
            else:
                raise Exception("Unsupported operating system!")

    def voxel_visualzation(self):
        """
        voxel grid visualization, supported format:["binvox", "mira"]
        :return:
        """
        for i in range(self.filenum):
            print("Processing: ", self.filelist[i])
            file_path = self.filelist[i]  # 要处理的文件
            input_pc_format = self.opts.input_format
            assert input_pc_format in ["binvox", "mira"]
            file = self.opts.input_file
            if file_path == file:
                if platform.system() == "Windows":
                    os.system('viewvox -ki ' + file)
                elif platform.system() == "Linux":
                    os.system('viewvox -ki ' + file)
                else:
                    raise Exception("Unsupported operating system!")


if __name__ == "__main__":
    # 获取文件列表
    fl = get_all_files(FLAGS.input_dir, FLAGS.input_format)
    # print(fl)
    formatFactory = Voxel_FormatFactory(FLAGS, fl)
    if FLAGS.mode == 7:
        # mesh转换成体素网格
        formatFactory.mesh_voxel()
    elif FLAGS.mode == 8:
        # 体素网格可视化
        formatFactory.voxel_visualzation()




