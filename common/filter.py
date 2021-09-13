import pcl


def passThroughFilter(np_xyz, up_limit):
    """
    点云直通滤波器
    :param np_xyz: 点云的XYZ(type:numpy)
    :param up_limit: 滤波器的上限值
    :return: 滤波后的XYZ(type:numpy)
    """
    cloud = pcl.PointCloud()
    cloud.from_array(np_xyz)
    passThrough = cloud.make_passthrough_filter()
    passThrough.set_filter_field_name("z")
    passThrough.set_filter_limits(0.0, up_limit)
    cloud_filtered = passThrough.filter()
    return cloud_filtered


def voxelGrid(np_xyz, voxel_size):
    """
    体素网格滤波器
    :param np_xyz: 点云的XYZ(type:numpy)
    :param voxel_size: 体素的大小
    :return: 滤波后的XYZ(type:numpy)
    """
    cloud = pcl.PointCloud()
    cloud.from_array(np_xyz)
    sor = cloud.make_voxel_grid_filter()
    sor.set_leaf_size(voxel_size, voxel_size, voxel_size)
    cloud_filtered = sor.filter()
    return cloud_filtered


def project_inliers(np_xyz):
    """
    :param np_xyz: 点云的XYZ(type:numpy)
    :return: 滤波后的XYZ(type:numpy)
    """
    cloud = pcl.PointCloud()
    cloud.from_array(np_xyz)
    proj = cloud.make_ProjectInliers()
    proj.set_model_type(pcl.SACMODEL_PLANE)
    cloud_projected = proj.filter()
    return cloud_projected


def remove_outlier(np_xyz, choices, radius, min_neighbor):
    """
    :param np_xyz: 点云的XYZ(type:numpy)
    :return: 滤波后的XYZ(type:numpy)
    """
    cloud = pcl.PointCloud()
    cloud.from_array(np_xyz)
    if choices == "radius":
        outrem = cloud.make_RadiusOutlierRemoval()
        outrem.set_radius_search(radius)
        outrem.set_MinNeighborsInRadius(min_neighbor)
        cloud_filtered = outrem.filter()
        return cloud_filtered
    elif choices == "condition":
        range_cond = cloud.make_ConditionAnd()
        range_cond.add_Comparison2('z', pcl.CythonCompareOp_Type.GT, 0.0)
        range_cond.add_Comparison2('z', pcl.CythonCompareOp_Type.LT, radius)
        # build the filter
        condrem = cloud.make_ConditionalRemoval(range_cond)
        condrem.set_KeepOrganized(True)
        # apply filter
        cloud_filtered = condrem.filter()
        return cloud_filtered
    else:
        print("please specify command line arg paramter 'radius' or 'condition'")


def StatisticalOutlierRemovalFilter(np_xyz, std_dev):
    """
    StatisticalOutlierRemovalFilter
    :param np_xyz: 点云的XYZ(type:numpy)
    :param std_dev: 标准差
    :return: 滤波后的XYZ(type:numpy)
    """
    cloud = pcl.PointCloud()
    cloud.from_array(np_xyz)
    filter = cloud.make_statistical_outlier_filter()
    filter.set_mean_k(50)
    filter.set_std_dev_mul_thresh(std_dev)
    cloud_filtered = filter.filter()
    return cloud_filtered

