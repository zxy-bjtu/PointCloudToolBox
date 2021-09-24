#pcl_binvox

Command-line tools to convert between .binvox and .pcd file formats.

binvox is a binary format for a 3D voxel grid, developed by [Patrick Min](http://www.patrickmin.com/binvox/).  
pcd is Point Cloud Data from PCL ([PointCloud Library](http://www.pointclouds.org)).

David Butterworth, 2016.

### Install ROS Melodic
```shell script
$ sudo sh -c '. /etc/lsb-release && echo "deb http://mirrors.tuna.tsinghua.edu.cn/ros/ubuntu/ `lsb_release -cs` main" > /etc/apt/sources.list.d/ros-latest.list'
$ sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
$ sudo apt update
$ sudo apt install ros-melodic-desktop-full
$ sudo apt install python-rosdep2
$ sudo rosdep init
$ rosdep update
```
If you meet the question `ERROR: error loading sources list:
    The read operation timed out` when execute `rosdep update`, you can find solution in: `http://obgeqwh.top/rosdep-update-read-operation-time-out-%E4%B8%8D%E7%94%A8%E7%BF%BB%E5%A2%99%E7%9A%84%E8%A7%A3%E5%86%B3%E6%96%B9%E6%B3%95/`

```shell script
$ echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
$ source ~/.bashrc
$ sudo apt install python-rosinstall python-rosinstall-generator python-wstool build-essential
$ rosversion -d
```

### Install PCL(C++ Library) in catkin environment
```shell script
$ mkdir ~/catkin_ws
$ cd ~/catkin_ws
$ mkdir src
$ catkin_make
$ cd src
$ git clone https://github.com/PointCloudLibrary/pcl.git
$ cd pcl-1.9.1/
$ mkdir build && cd build
$ cmake -DCMAKE_BUILD_TYPE=None -DBUILD_GPU=ON -DBUILD_apps=ON -DBUILD_examples=ON .. 
$ make -j2
$ make install
```

### compile pcl_binvox
```shell script
$ git clone https://github.com/dbworth/pcl_binvox.git
$ apt-get install ros-melodic-pcl-ros
$ catkin_make -DCATKIN_WHITELIST_PACKAGES="pcl_binvox"
$ export PATH=$PATH:~/catkin_ws/devel/lib/pcl_binvox/  
```
### Usage

This was tested on Ubuntu 18.04 with PCL 1.9  

**Convert from .binvox to .pcd**  
The output file name is specified first. Then you can list multiple binvox files as input.
```
$ binvox2pcd -o output.pcd data/chair.binvox
```

**Convert from .pcd to .binvox**  
Specify the voxel grid resolution, between 32 and 1024.
```
$ pcd2binvox -d 32 data/teapot.pcd output.binvox
```

### Reference

```markdown
https://github.com/dbworth/pcl_binvox
```
