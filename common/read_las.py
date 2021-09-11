import struct
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from header import get_header
from header import Point


def get_version(input_las):
    f = open(input_las, 'rb')
    f.read(24)
    version_major, version_minor = struct.unpack('2B', f.read(2))
    print(f"las版本:{version_major}.{version_minor}")
    return version_major, version_minor


def read_las(input_las):
    """
    读取点云文件.las的数据
    :param input_las: 输入的las文件的路径
    :return:
    """
    f = open(input_las, 'rb')
    version = get_version(input_las)
    header = get_header(f, version)
    points = Point(header.x_scale_factor,
                   header.y_scale_factor,
                   header.z_scale_factor,
                   header.x_offset,
                   header.y_offset,
                   header.z_offset,
                   )
    data = points.read_point(f, header.offset_to_point_data,
                             header.point_data_record_format,
                             header.number_of_point_records)
    # print(data)
    return data

