"""
定义las1.0~1.4的公共头数据类。
根据不同的版本调用不同类来解析文件
"""
import struct


class OneZeroHeader(object):
    """
    1.0版本
    """

    def __init__(self):
        self.file_signature = None  # 文件签名
        self.reserved = None  # 预留字段
        self.guid_1 = None  # 项目ID1
        self.guid_2 = None  # 项目ID2
        self.guid_3 = None  # 项目ID3
        self.guid_4 = None  # 项目ID4
        self.version_major = None  # 大版本
        self.version_minor = None  # 小版本
        self.system_identifier = None  # 硬件信息
        self.generating_software = None  # 软件信息
        self.flight_date_julian = None  # 文件创建天数
        self.year = None  # 文件创建年份
        self.header_size = None  # 头文件大小
        self.offset_to_data = None  # 点数据记录起始位置
        self.number_of_variable_length_records = None  # 可变数据记录长度
        self.point_data_record_format = None  # 点数据记录格式
        self.point_data_record_length = None  # 点数据记录长度
        self.number_of_point_records = None  # 总共的点数量
        self.number_of_points_by_return = None  # 回波点数量
        self.x_scale_factor = None  # x比例因子
        self.y_scale_factor = None  # y比例因子
        self.z_scale_factor = None  # z比例因子
        self.x_offset = None  # x偏移量
        self.y_offset = None  # y偏移量
        self.z_offset = None  # z偏移量
        self.max_x = None  # x最大值
        self.min_x = None  # x最小值
        self.max_y = None  # y最大值
        self.min_y = None  # y最小值
        self.max_z = None  # z最大值
        self.min_z = None  # z最小值

    def read_header(self, f):
        """
        读取头
        :param f:
        :return:
        """
        f.seek(0)
        self.file_signature = f.read(4).decode("utf-8")
        self.reserved = struct.unpack('L', f.read(4))[0]
        self.guid_1 = struct.unpack('L', f.read(4))[0]
        self.guid_2 = struct.unpack('H', f.read(2))[0]
        self.guid_3 = struct.unpack('H', f.read(2))[0]
        self.guid_4 = struct.unpack('8s', f.read(8))[0].decode("utf-8")
        self.version_major = struct.unpack('B', f.read(1))[0]
        self.version_minor = struct.unpack('B', f.read(1))[0]
        self.system_identifier = struct.unpack('32s', f.read(32))[0].decode("utf-8")
        self.generating_software = struct.unpack('32s', f.read(32))[0].decode("utf-8")
        self.flight_date_julian = struct.unpack('H', f.read(2))[0]
        self.year = struct.unpack('H', f.read(2))[0]
        self.header_size = struct.unpack('H', f.read(2))[0]
        self.offset_to_data = struct.unpack('L', f.read(4))[0]
        self.number_of_variable_length_records = struct.unpack('L', f.read(4))[0]
        self.point_data_record_format = struct.unpack('B', f.read(1))[0]
        self.point_data_record_length = struct.unpack('H', f.read(2))[0]
        self.number_of_point_records = struct.unpack('L', f.read(4))[0]
        self.number_of_points_by_return = struct.unpack('5L', f.read(20))[0]
        self.x_scale_factor = struct.unpack('d', f.read(8))[0]
        self.y_scale_factor = struct.unpack('d', f.read(8))[0]
        self.z_scale_factor = struct.unpack('d', f.read(8))[0]
        self.x_offset = struct.unpack('d', f.read(8))[0]
        self.y_offset = struct.unpack('d', f.read(8))[0]
        self.z_offset = struct.unpack('d', f.read(8))[0]
        self.max_x = struct.unpack('d', f.read(8))[0]
        self.min_x = struct.unpack('d', f.read(8))[0]
        self.max_y = struct.unpack('d', f.read(8))[0]
        self.min_y = struct.unpack('d', f.read(8))[0]
        self.max_z = struct.unpack('d', f.read(8))[0]
        self.min_z = struct.unpack('d', f.read(8))[0]


class OneOneHeader(object):
    """
    1.1版本
    """

    def __init__(self):
        self.file_signature = None  # 文件签名
        self.file_source_id = None  # 文件ID
        self.reserved = None  # 预留字段
        self.project_id_1 = None  # 项目ID1
        self.project_id_2 = None  # 项目ID2
        self.project_id_3 = None  # 项目ID3
        self.project_id_4 = None  # 项目ID4
        self.version_major = None  # 大版本
        self.version_minor = None  # 小版本
        self.system_identifier = None  # 硬件信息
        self.generating_software = None  # 软件信息
        self.file_creation_day_of_year = None  # 文件创建天数
        self.file_creation_year = None  # 文件创建年份
        self.header_size = None  # 头文件大小
        self.offset_to_point_data = None  # 点数据记录起始位置
        self.number_of_variable_length_records = None  # 可变数据记录长度
        self.point_data_record_format = None  # 点数据记录格式
        self.point_data_record_length = None  # 点数据记录长度
        self.number_of_point_records = None  # 总共的点数量
        self.number_of_points_by_return = None  # 回波点数量
        self.x_scale_factor = None  # x比例因子
        self.y_scale_factor = None  # y比例因子
        self.z_scale_factor = None  # z比例因子
        self.x_offset = None  # x偏移量
        self.y_offset = None  # y偏移量
        self.z_offset = None  # z偏移量
        self.max_x = None  # x最大值
        self.min_x = None  # x最小值
        self.max_y = None  # y最大值
        self.min_y = None  # y最小值
        self.max_z = None  # z最大值
        self.min_z = None  # z最小值

    def read_header(self, f):
        f.seek(0)
        self.file_signature = f.read(4).decode("utf-8")
        self.file_source_id = struct.unpack('H', f.read(2))[0]
        self.reserved = struct.unpack('H', f.read(2))[0]
        self.project_id_1 = struct.unpack('L', f.read(4))[0]
        self.project_id_2 = struct.unpack('H', f.read(2))[0]
        self.project_id_3 = struct.unpack('H', f.read(2))[0]
        self.project_id_4 = struct.unpack('8s', f.read(8))[0].decode("utf-8")
        self.version_major = struct.unpack('B', f.read(1))[0]
        self.version_minor = struct.unpack('B', f.read(1))[0]
        self.system_identifier = struct.unpack('32s', f.read(32))[0].decode("utf-8")
        self.generating_software = struct.unpack('32s', f.read(32))[0].decode("utf-8")
        self.file_creation_day_of_year = struct.unpack('H', f.read(2))[0]
        self.file_creation_year = struct.unpack('H', f.read(2))[0]
        self.header_size = struct.unpack('H', f.read(2))[0]
        self.offset_to_point_data = struct.unpack('L', f.read(4))[0]
        self.number_of_variable_length_records = struct.unpack('L', f.read(4))[0]
        self.point_data_record_format = struct.unpack('B', f.read(1))[0]
        self.point_data_record_length = struct.unpack('H', f.read(2))[0]
        self.number_of_point_records = struct.unpack('L', f.read(4))[0]
        self.number_of_points_by_return = struct.unpack('5L', f.read(20))[0]
        self.x_scale_factor = struct.unpack('d', f.read(8))[0]
        self.y_scale_factor = struct.unpack('d', f.read(8))[0]
        self.z_scale_factor = struct.unpack('d', f.read(8))[0]
        self.x_offset = struct.unpack('d', f.read(8))[0]
        self.y_offset = struct.unpack('d', f.read(8))[0]
        self.z_offset = struct.unpack('d', f.read(8))[0]
        self.max_x = struct.unpack('d', f.read(8))[0]
        self.min_x = struct.unpack('d', f.read(8))[0]
        self.max_y = struct.unpack('d', f.read(8))[0]
        self.min_y = struct.unpack('d', f.read(8))[0]
        self.max_z = struct.unpack('d', f.read(8))[0]
        self.min_z = struct.unpack('d', f.read(8))[0]


class OneTwoHeader(object):
    """
    1.2版本
    """

    def __init__(self):
        self.file_signature = None  # 文件签名
        self.file_source_id = None  # 文件ID
        self.global_encoding = None  # 全球编码
        self.project_id_1 = None  # 项目ID1
        self.project_id_2 = None  # 项目ID2
        self.project_id_3 = None  # 项目ID3
        self.project_id_4 = None  # 项目ID4
        self.version_major = None  # 大版本
        self.version_minor = None  # 小版本
        self.system_identifier = None  # 硬件信息
        self.generating_software = None  # 软件信息
        self.file_creation_day_of_year = None  # 文件创建天数
        self.file_creation_year = None  # 文件创建年份
        self.header_size = None  # 头文件大小
        self.offset_to_point_data = None  # 点数据记录起始位置
        self.number_of_variable_length_records = None  # 可变数据记录长度
        self.point_data_record_format = None  # 点数据记录格式
        self.point_data_record_length = None  # 点数据记录长度
        self.number_of_point_records = None  # 总共的点数量
        self.number_of_points_by_return = None  # 回波点数量
        self.x_scale_factor = None  # x比例因子
        self.y_scale_factor = None  # y比例因子
        self.z_scale_factor = None  # z比例因子
        self.x_offset = None  # x偏移量
        self.y_offset = None  # y偏移量
        self.z_offset = None  # z偏移量
        self.max_x = None  # x最大值
        self.min_x = None  # x最小值
        self.max_y = None  # y最大值
        self.min_y = None  # y最小值
        self.max_z = None  # z最大值
        self.min_z = None  # z最小值

    def read_header(self, f):
        f.seek(0)
        self.file_signature = f.read(4).decode("utf-8")
        self.file_source_id = struct.unpack('H', f.read(2))[0]
        self.global_encoding = struct.unpack('H', f.read(2))[0]
        self.project_id_1 = struct.unpack('L', f.read(4))[0]
        self.project_id_2 = struct.unpack('H', f.read(2))[0]
        self.project_id_3 = struct.unpack('H', f.read(2))[0]
        self.project_id_4 = struct.unpack('8s', f.read(8))[0].decode("utf-8")
        self.version_major = struct.unpack('B', f.read(1))[0]
        self.version_minor = struct.unpack('B', f.read(1))[0]
        self.system_identifier = struct.unpack('32s', f.read(32))[0].decode("utf-8")
        self.generating_software = struct.unpack('32s', f.read(32))[0].decode("utf-8")
        self.file_creation_day_of_year = struct.unpack('H', f.read(2))[0]
        self.file_creation_year = struct.unpack('H', f.read(2))[0]
        self.header_size = struct.unpack('H', f.read(2))[0]
        self.offset_to_point_data = struct.unpack('L', f.read(4))[0]
        self.number_of_variable_length_records = struct.unpack('L', f.read(4))[0]
        self.point_data_record_format = struct.unpack('B', f.read(1))[0]
        self.point_data_record_length = struct.unpack('H', f.read(2))[0]
        self.number_of_point_records = struct.unpack('L', f.read(4))[0]
        self.number_of_points_by_return = struct.unpack('5L', f.read(20))[0]
        self.x_scale_factor = struct.unpack('d', f.read(8))[0]
        self.y_scale_factor = struct.unpack('d', f.read(8))[0]
        self.z_scale_factor = struct.unpack('d', f.read(8))[0]
        self.x_offset = struct.unpack('d', f.read(8))[0]
        self.y_offset = struct.unpack('d', f.read(8))[0]
        self.z_offset = struct.unpack('d', f.read(8))[0]
        self.max_x = struct.unpack('d', f.read(8))[0]
        self.min_x = struct.unpack('d', f.read(8))[0]
        self.max_y = struct.unpack('d', f.read(8))[0]
        self.min_y = struct.unpack('d', f.read(8))[0]
        self.max_z = struct.unpack('d', f.read(8))[0]
        self.min_z = struct.unpack('d', f.read(8))[0]


class OneThreeHeader(object):
    """
    1.3版本
    """

    def __init__(self):
        self.file_signature = None  # 文件签名
        self.file_source_id = None  # 文件ID
        self.global_encoding = None  # 全球编码
        self.project_id_1 = None  # 项目ID1
        self.project_id_2 = None  # 项目ID2
        self.project_id_3 = None  # 项目ID3
        self.project_id_4 = None  # 项目ID4
        self.version_major = None  # 大版本
        self.version_minor = None  # 小版本
        self.system_identifier = None  # 硬件信息
        self.generating_software = None  # 软件信息
        self.file_creation_day_of_year = None  # 文件创建天数
        self.file_creation_year = None  # 文件创建年份
        self.header_size = None  # 头文件大小
        self.offset_to_point_data = None  # 点数据记录起始位置
        self.number_of_variable_length_records = None  # 可变数据记录长度
        self.point_data_record_format = None  # 点数据记录格式
        self.point_data_record_length = None  # 点数据记录长度
        self.number_of_point_records = None  # 总共的点数量
        self.number_of_points_by_return = None  # 回波点数量
        self.x_scale_factor = None  # x比例因子
        self.y_scale_factor = None  # y比例因子
        self.z_scale_factor = None  # z比例因子
        self.x_offset = None  # x偏移量
        self.y_offset = None  # y偏移量
        self.z_offset = None  # z偏移量
        self.max_x = None  # x最大值
        self.min_x = None  # x最小值
        self.max_y = None  # y最大值
        self.min_y = None  # y最小值
        self.max_z = None  # z最大值
        self.min_z = None  # z最小值
        self.start_of_waveform_data_packet_record = None  # 波形数据包记录的开始位置

    def read_header(self, f):
        f.seek(0)
        self.file_signature = f.read(4).decode("utf-8")
        self.file_source_id = struct.unpack('H', f.read(2))[0]
        self.global_encoding = struct.unpack('H', f.read(2))[0]
        self.project_id_1 = struct.unpack('L', f.read(4))[0]
        self.project_id_2 = struct.unpack('H', f.read(2))[0]
        self.project_id_3 = struct.unpack('H', f.read(2))[0]
        self.project_id_4 = struct.unpack('8s', f.read(8))[0].decode("utf-8")
        self.version_major = struct.unpack('B', f.read(1))[0]
        self.version_minor = struct.unpack('B', f.read(1))[0]
        self.system_identifier = struct.unpack('32s', f.read(32))[0].decode("utf-8")
        self.generating_software = struct.unpack('32s', f.read(32))[0].decode("utf-8")
        self.file_creation_day_of_year = struct.unpack('H', f.read(2))[0]
        self.file_creation_year = struct.unpack('H', f.read(2))[0]
        self.header_size = struct.unpack('H', f.read(2))[0]
        self.offset_to_point_data = struct.unpack('L', f.read(4))[0]
        self.number_of_variable_length_records = struct.unpack('L', f.read(4))[0]
        self.point_data_record_format = struct.unpack('B', f.read(1))[0]
        self.point_data_record_length = struct.unpack('H', f.read(2))[0]
        self.number_of_point_records = struct.unpack('L', f.read(4))[0]
        self.number_of_points_by_return = struct.unpack('5L', f.read(20))[0]
        self.x_scale_factor = struct.unpack('d', f.read(8))[0]
        self.y_scale_factor = struct.unpack('d', f.read(8))[0]
        self.z_scale_factor = struct.unpack('d', f.read(8))[0]
        self.x_offset = struct.unpack('d', f.read(8))[0]
        self.y_offset = struct.unpack('d', f.read(8))[0]
        self.z_offset = struct.unpack('d', f.read(8))[0]
        self.max_x = struct.unpack('d', f.read(8))[0]
        self.min_x = struct.unpack('d', f.read(8))[0]
        self.max_y = struct.unpack('d', f.read(8))[0]
        self.min_y = struct.unpack('d', f.read(8))[0]
        self.max_z = struct.unpack('d', f.read(8))[0]
        self.min_z = struct.unpack('d', f.read(8))[0]
        self.start_of_waveform_data_packet_record = struct.unpack('Q', f.read(8))[0]


class OneFourHeader(object):
    """
    1.4版本
    """

    def __init__(self):
        self.file_signature = None  # 文件签名
        self.file_source_id = None  # 文件ID
        self.global_encoding = None  # 全球编码
        self.project_id_1 = None  # 项目ID1
        self.project_id_2 = None  # 项目ID2
        self.project_id_3 = None  # 项目ID3
        self.project_id_4 = None  # 项目ID4
        self.version_major = None  # 大版本
        self.version_minor = None  # 小版本
        self.system_identifier = None  # 硬件信息
        self.generating_software = None  # 软件信息
        self.file_creation_day_of_year = None  # 文件创建天数
        self.file_creation_year = None  # 文件创建年份
        self.header_size = None  # 头文件大小
        self.offset_to_point_data = None  # 点数据记录起始位置
        self.number_of_variable_length_records = None  # 可变数据记录长度
        self.point_data_record_format = None  # 点数据记录格式
        self.point_data_record_length = None  # 点数据记录长度
        self.legacy_number_of_point_records = None  # 老格式总共的点数量
        self.legacy_number_of_points_by_return = None  # 老格式回波点数量
        self.x_scale_factor = None  # x比例因子
        self.y_scale_factor = None  # y比例因子
        self.z_scale_factor = None  # z比例因子
        self.x_offset = None  # x偏移量
        self.y_offset = None  # y偏移量
        self.z_offset = None  # z偏移量
        self.max_x = None  # x最大值
        self.min_x = None  # x最小值
        self.max_y = None  # y最大值
        self.min_y = None  # y最小值
        self.max_z = None  # z最大值
        self.min_z = None  # z最小值
        self.start_of_waveform_data_packet_record = None  # 波形数据记录起始点
        self.start_of_first_extended_variable_length_record = None  # 扩展变长记录起始
        self.number_of_extended_variable_length_records = None  # 扩展变长记录数目
        self.number_of_point_records = None  # 点记录总数
        self.number_of_points_by_return = None  # 回波点数量

    def read_header(self, f):
        f.seek(0)
        self.file_signature = f.read(4).decode("utf-8")
        self.file_source_id = struct.unpack('H', f.read(2))[0]
        self.global_encoding = struct.unpack('H', f.read(2))[0]
        self.project_id_1 = struct.unpack('L', f.read(4))[0]
        self.project_id_2 = struct.unpack('H', f.read(2))[0]
        self.project_id_3 = struct.unpack('H', f.read(2))[0]
        self.project_id_4 = struct.unpack('8s', f.read(8))[0].decode("utf-8")
        self.version_major = struct.unpack('B', f.read(1))[0]
        self.version_minor = struct.unpack('B', f.read(1))[0]
        self.system_identifier = struct.unpack('32s', f.read(32))[0].decode("utf-8")
        self.generating_software = struct.unpack('32s', f.read(32))[0].decode("utf-8")
        self.file_creation_day_of_year = struct.unpack('H', f.read(2))[0]
        self.file_creation_year = struct.unpack('H', f.read(2))[0]
        self.header_size = struct.unpack('H', f.read(2))[0]
        self.offset_to_point_data = struct.unpack('L', f.read(4))[0]
        self.number_of_variable_length_records = struct.unpack('L', f.read(4))[0]
        self.point_data_record_format = struct.unpack('B', f.read(1))[0]
        self.point_data_record_length = struct.unpack('H', f.read(2))[0]
        self.legacy_number_of_point_records = struct.unpack('L', f.read(4))[0]
        self.legacy_number_of_points_by_return = struct.unpack('5L', f.read(20))[0]
        self.x_scale_factor = struct.unpack('d', f.read(8))[0]
        self.y_scale_factor = struct.unpack('d', f.read(8))[0]
        self.z_scale_factor = struct.unpack('d', f.read(8))[0]
        self.x_offset = struct.unpack('d', f.read(8))[0]
        self.y_offset = struct.unpack('d', f.read(8))[0]
        self.z_offset = struct.unpack('d', f.read(8))[0]
        self.max_x = struct.unpack('d', f.read(8))[0]
        self.min_x = struct.unpack('d', f.read(8))[0]
        self.max_y = struct.unpack('d', f.read(8))[0]
        self.min_y = struct.unpack('d', f.read(8))[0]
        self.max_z = struct.unpack('d', f.read(8))[0]
        self.min_z = struct.unpack('d', f.read(8))[0]
        self.start_of_waveform_data_packet_record = struct.unpack('Q', f.read(8))[0]
        self.start_of_first_extended_variable_length_record = struct.unpack('Q', f.read(8))[0]
        self.number_of_extended_variable_length_records = struct.unpack('L', f.read(4))[0]
        self.number_of_point_records = struct.unpack('Q', f.read(8))[0]
        self.number_of_points_by_return = struct.unpack('15Q', f.read(120))[0]


def get_header(f, version):
    if version == (1, 0):
        new_header = OneZeroHeader()
    elif version == (1, 1):
        new_header = OneOneHeader()
    elif version == (1, 2):
        new_header = OneTwoHeader()
    elif version == (1, 3):
        new_header = OneThreeHeader()
    elif version == (1, 4):
        new_header = OneFourHeader()
    else:
        raise Exception("未找到对应文件版本")

    new_header.read_header(f)
    return new_header


# """
# 定义las的Point类
# """
class Point(object):

    def __init__(self, x_s_f, y_s_f, z_s_f, x_o, y_o, z_o):
        self.x_scale_factor = x_s_f
        self.y_scale_factor = y_s_f
        self.z_scale_factor = z_s_f
        self.x_offset = x_o
        self.y_offset = y_o
        self.z_offset = z_o

    def get_offset_bytes(self, point_data_record_format):
        """
        根据不同的点格式跳过的字节数
        :param point_data_record_format:点格式0~10
        :return:
        """
        # x,y,z共占12字节
        data_format = {
            0: 8,  # 点格式0 共20字节
            1: 16,  # 点格式1 共28字节
            2: 14,  # 点格式2 共26字节
            3: 22,  # 点格式3 共34字节
            4: 45,  # 点格式4 共57字节
            5: 51,  # 点格式5 共63字节
            6: 18,  # 点格式6 共30字节
            7: 24,  # 点格式7 共36字节
            8: 26,  # 点格式8 共38字节
            9: 47,  # 点格式9 共59字节
            10: 55,  # 点格式10 共67字节
        }

        offset_bytes = data_format.get(point_data_record_format, None)
        if offset_bytes is None:
            raise Exception(f"不存在当前的点格式{point_data_record_format}")
        return offset_bytes

    def read_point(self, f, offset_to_point_data, point_data_record_format, num):
        """
        读取当前文件中的点数据
        :param f:  数据文件
        :param offset_to_point_data: 点数据开始读取的地方
        :param point_data_record_format:  点数据格式0~10
        :param num: 读取多少个点
        :return: 读取的数据点
        """
        offset_bytes = self.get_offset_bytes(point_data_record_format)
        f.seek(offset_to_point_data)
        points = list()

        i = 0
        while i < num:
            point_bytes = f.read(12)
            x_record, y_record, z_record = struct.unpack_from('3l', point_bytes)
            x = x_record * self.x_scale_factor + self.x_offset
            y = y_record * self.y_scale_factor + self.y_offset
            z = z_record * self.z_scale_factor + self.z_offset
            i += 1
            f.read(offset_bytes)
            points.append((x, y, z))

        return points
