import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=int, default=0,
                    help='Point cloud data processing mode, you can set: ' \
                         '0: Format conversion between point clouds and point cloud' \
                         '1: Format conversion between 3d mesh and 3d mesh' \
                         '2: Convert mesh to point cloud'
                    )

# point cloud IO
parser.add_argument('--input_dir', type=str, help='the path of point cloud which you want to deal with')
parser.add_argument('--output_dir', type=str, help='the output path of results')

# p->p/m->m format conversion
parser.add_argument('--input_format', type=str, default='pcd',
                    help='the input format of point cloud/mesh')
parser.add_argument('--output_format', type=str, default='xyz',
                    help='the output format of point cloud/mesh')

# mesh downsampling
parser.add_argument('--sampler', type=str, default='possion_disk_sampling',
                    help="[poisson_disk_sampling, uniform_sampling]")
parser.add_argument('--point_num', type=int, default=1024, help="number of points that should be sampled")
parser.add_argument('--factor', type=int, default=5, help="Factor for the initial uniformly sampled PointCloud. "
                                                          "This init PointCloud is used for sample elimination")

# pointcloud/mesh filter
# point cloud filter
parser.add_argument('--type', type=str, default='pc', help="the file type, [pc, mesh] ")
# 点云滤波器
parser.add_argument('--filter', type=str, default='PassThroughFilter',
                    help="[PassThroughFilter, VoxelGridFilter, project_inliers, remove_outliers, "
                         "statistical_removal]")
parser.add_argument('--upper_limit', type=float, default=0.5, help="the upper limit value of passThroughFilter")
parser.add_argument('--voxel_size', type=float, default=0.01, help="the voxel size of VoxelGridFilter")
parser.add_argument('--removal', '-r', choices=('radius', 'condition'), default='',
                    help='RadiusOutlier/Condition Removal')
parser.add_argument('--radius', type=float, default=1.0, help='search radius for RadiusOutlier')
parser.add_argument('--min_neighbor', type=int, default=2, help='min neighbors in radius for RadiusOutlier')
parser.add_argument('--std_dev', type=int, default=1.0, help='std dev used in Statistical Outlier Removal filter')

# 网格滤波器
parser.add_argument('--mesh_filter', type=str, default='taubin', help="[taubin, laplacian, neighbour]")

FLAGS = parser.parse_args()
