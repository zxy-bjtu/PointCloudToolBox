import os
import re
import platform


def get_all_files(dirPath, fileType):
    """
    Windows/Linux系统下获取某种后缀的所有文件的路径列表
    :param dirPath:文件路径
    :param fileType:文件类型（eg: txt, pcd, off, xyz等）
    :return:
    """
    fileList = []
    files = os.listdir(dirPath)
    pattern = re.compile(".*\." + fileType)
    for f in files:
        # 如果是fold，递归调用get_all_win_files
        if os.path.isdir(dirPath + '/' + f):
            get_all_files(dirPath + '/' + f, fileType)
        #  如果是文件，看是否为所需类型
        elif os.path.isfile(dirPath + '/' + f):
            matches = pattern.match(f)  # 判断f的文件名是否符合正则表达式，即是否为off后缀
            if matches is not None:
                fileList.append(dirPath + '/' + matches.group())
        else:
            pass
    return fileList


if __name__ == "__main__":
    sysstr = platform.system()
    print(sysstr)
    if sysstr == "Windows":
        print(get_all_files("../data/test", "off"))
    elif sysstr == "Linux":
        print(get_all_files("../data/test", "off"))
    else:
        raise Exception("Unsupported Operation System!")


