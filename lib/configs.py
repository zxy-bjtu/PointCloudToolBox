import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=int, default=0,
                    help='Point cloud data processing mode, you can set: ' \
                         '0: Format conversion between point clouds and point cloud' \
                         '1: Format conversion between 3d mesh and 3d mesh' \
                         '2: Convert mesh to point cloud' \
                         '3: point cloud filtering' \
                         '4: 3d mesh filtering' \
                         '5: point cloud down sampling' \
                         '6: 3d mesh subdivision' \
                         '7: Convert 3d mesh into a binary 3D voxel grid' \
                         '8: voxel grid visualization'
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
parser.add_argument('--std_dev', type=float, default=1.0, help='std dev used in Statistical Outlier Removal filter')

# 网格滤波器
parser.add_argument('--mesh_filter', type=str, default='taubin', help="[taubin, laplacian, neighbour]")

# 点云下采样
parser.add_argument('--down_sampler', type=str, default='fps', help="[fps, random, uniform, voxel]")
# uniform sampling
parser.add_argument('--k', type=int, default=4, help="every_k_points (int): Sample rate")

# mesh subdivision
parser.add_argument('--subdivision_type', type=str, default='loop', help="[midpoint, loop]")
parser.add_argument('--iteration', type=int, default=2, help="The parameter number_of_iterations defines how many "
                                                             "times 3d mesh subdivision should be repeated.")

# convert mesh into voxel grid
parser.add_argument('--d', type=int, default=256, help="specify voxel grid size, max:1024")

# voxel grid visualization
parser.add_argument('--input_file', type=str, help="the file you want to see.")

# point cloud registration
parser.add_argument('--s_file', type=str, help="source point cloud file")
parser.add_argument('--t_file', type=str, help="target point cloud file")

FLAGS = parser.parse_args()
