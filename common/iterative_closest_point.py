"""
ICP algorithm for point cloud registration.
"""
import pcl
import os
import sys
import open3d as o3d
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../lib'))
from configs import FLAGS


def icp(np_xyz_in, np_xyz_out):
    """
    icp algorithm
    More point cloud registration algorithm:
    GICP: https://github.com/avsegal/gicp
    NICP: http://jacoposerafin.com/nicp/
    :param np_xyz_in: source point cloud (type:numpy)
    :param np_xyz_out: target point cloud (type:numpy)
    :return:
    """
    cloud_in = pcl.PointCloud()
    cloud_in.from_array(np_xyz_in)

    cloud_out = pcl.PointCloud()
    cloud_out.from_array(np_xyz_out)

    # If ValueError: order must be one of 'C', 'F', 'A', or 'K' (got 'fortran')
    # Please refer to https://github.com/strawlab/python-pcl/issues/358
    # after modify the pcl source code, try to compile and install it again.
    icp = cloud_in.make_IterativeClosestPoint()
    converged, transf, estimate, fitness = icp.icp(cloud_in, cloud_out)
    print('ICP algorithm has converged:' + str(converged) + ' \nscore: ' + str(fitness))
    print('T: ', str(transf))


if __name__ == "__main__":
    s_pc = o3d.io.read_point_cloud(FLAGS.s_file)
    t_pc = o3d.io.read_point_cloud(FLAGS.t_file)
    xyz_in = np.asarray(s_pc.points, dtype=np.float32)
    # print(xyz_in)
    xyz_out = np.asarray(t_pc.points, dtype=np.float32)
    # print(xyz_out)
    print("**********ICP**********")
    print("source file: ", FLAGS.s_file)
    print("target file: ", FLAGS.t_file)
    icp(xyz_in, xyz_out)
    print("***********************")


