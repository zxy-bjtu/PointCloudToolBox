# Point Cloud ToolBox


This point cloud processing tool library can be used to process point clouds, 3d meshes, and voxels.

## Environment
```markdown
python 3.7.5

Dependent packages:
  - scipy==1.6.2
  - torch==1.7.1+cu101
  - openmesh==1.1.6
  - open3d==0.13.0
  - plyfile==0.7.4
  - numpy==1.21.2
  - vtk==8.2.0
  - python-pcl==0.3.0rc1
  - pyntcloud==0.1.5
  - pythreejs==2.3.0 

How to install the environment:
$ pip install requirements.txt
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
- [x] point cloud voxelization (pointcloud->voxel grid)
    - [x] pcd | ply | pts | txt | xyz -> binvox
- [x] downsampling
    - [x] farthest point sampling(FPS)
    - [x] random sampling
    - [x] uniform sampling
    - [x] voxel sampling
- [x] upsampling
    - [x] Meta-PU [[published in TVCG2021](./PU/Meta-PU/README.md)]
- [x] filtering
    - [x] PassThrough Filter
    - [x] VoxelGrid Filter
    - [x] project_inliers Filter
    - [x] remove_outliers Filter
    - [x] statistical_removal Filter
- [x] registration
    - [x] ICP
    - [x] RANSAC
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
- [x] Calculate the surface area and volume of 3D Mesh
- [x] Convert 3d mesh to voxel grid
    - [x] obj -> binvox | hips | mira | vtk | msh
    - [x] off -> binvox | hips | mira | vtk | msh
    - [x] dxf -> binvox | hips | mira | vtk | msh
    - [x] ply -> binvox | hips | mira | vtk | msh
    - [x] stl -> binvox | hips | mira | vtk | msh
- [x] mesh subdivision
    - [x] loop
    - [x] midpoint
- [ ] 3d mesh visualization

### voxel
- [x] voxel visualization
    - [x] binvox
    - [x] mira

## Use
You can find command in `run.sh`.


**To be continued...**

    
   