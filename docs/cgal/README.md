# How to install CGAL in Ubuntu?

## 1. Install Dependencies

```shell script
sudo apt-get install libboost-all-dev

sudo apt-get install libgmp-dev

sudo apt-get install libgmp3-dev

sudo apt-get install libmpfr-dev

sudo apt-get install geomview

sudo apt install freeglut3 freeglut3-dev

sudo apt-get install binutils-gold

sudo apt-get install libglew-dev

sudo apt-get install g++

sudo apt-get install mesa-common-dev

sudo apt-get install build-essential

sudo apt-get install libeigen3-dev

sudo apt-get install libtbb-dev

sudo apt-get install zlib1g-dev

sudo apt-get install libqt5svg5-dev
```

## 2. Install QT

```shell script
sudo apt-get install qtcreator
sudo apt-get install qt5-default
```

You can run QT: 
```shell script
$ qtcreator
```

## 3. Install libQGLViewer
```shell script
sudo apt-get install libqglviewer-headers
```

## 4. Install CGAL
My version: [cgal-releases-CGAL-4.12.2](https://github.com/CGAL/cgal/releases/download/releases%2FCGAL-4.12.2/CGAL-4.12.2.zip)

other version: [https://github.com/CGAL/cgal](https://github.com/CGAL/cgal)

download the source code, then compile and install cgal:
```shell script
unzip CGAL-4.12.2.zip
cd CGAL-4.12.2/
mkdir build
cd build
cmake ..
make -j4
sudo make install
```
check libCGAL_Qt5.so :
```shell script
ls /usr/local/lib/libCGAL*
```

