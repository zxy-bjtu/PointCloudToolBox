# Point Cloud ToolBox


This point cloud processing tool library can be used to process point clouds, 3d meshes, and voxels.

## Environment
```markdown
python 3.7.5

Dependent packages:
- openmesh==1.1.6
- open3d==0.13.0
- plyfile==0.7.4
- numpy==1.21.0
- vtk==8.2.0
- python-pcl==0.3.0rc1

How to install the environment:
# pip install requirements.txt
```

## Todo
### Point Cloud
- [x] format conversion (pointcloud->pointcloud)  
     - [x] pcd -> xyz | pts | txt | csv | ply 
     - [x] las -> pcd | xyz | pts | ply | txt | csv
     - [x] ply -> pcd | xyz | pts | txt | csv   
     - [x] xyz -> pcd | pts | ply | txt | csv
     - [x] pts -> pcd | xyz | ply | txt | csv
     - [x] txt -> pcd | ply | xyz | pts
- [x] Calculate the surface area and volume of 3D Mesh
- [ ] Point cloud voxelization
- [x] downsampling
    - [x] farthest point sampling(FPS)
    - [x] random sampling
    - [x] uniform sampling
    - [x] voxel sampling
- [ ] upsampling
- [x] filtering
    - [x] PassThrough Filter
    - [x] VoxelGrid Filter
    - [x] project_inliers Filter
    - [x] remove_outliers Filter
    - [x] statistical_removal Filter
- [ ] registration
- [ ] 3D reconstruction
- [ ] visualization

### 3d Mesh
- [x] format conversion (mesh->mesh)
    - [x] ply -> obj | stl | off
    - [x] obj -> ply | off | stl
    - [x] off -> ply | obj | stl
- [x] down sampling into point cloud
    - [x] poisson disk sampling
    - [x] uniform sampling
- [x] mesh filtering
    - [x] Taubin filter
    - [x] Laplacian smooth
    - [x] simple neighbour average
- [ ] mesh voxelization
- [ ] mesh subdivision
- [ ] 3d mesh visualization

### voxel
- [ ] visualization

## Use
You can find command in `run.sh`.


**To be continued...**

    
   