//
// pcd2binvox
// Convert a .pcd file to .binvox
//
// pcd is Point Cloud Data from PCL (PointCloud Library).
// binvox is a binary format for a 3D voxel grid.
//
// David Butterworth, 2016.
//
// binvox was developed by Patrick Min (www.patrickmin.com)
// The RLE code below is based on binvox-rw-py by Daniel Maturana.
//

#include <string>
#include <fstream>
#include <iostream>
#include <iomanip> // setprecision

#include <cstdlib>
#include <cstring>

#include <boost/dynamic_bitset.hpp>

#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>

struct Voxel
{
  unsigned int x;
  unsigned int y;
  unsigned int z;
  Voxel() : x(0), y(0), z(0) {};
  Voxel(const unsigned int _x, const unsigned int _y, const unsigned int _z) : x(_x), y(_y), z(_z) {};
};

/*
// For debugging: Write the voxel indices to a file
void writeVoxelsToFile(const std::vector<Voxel>& voxels, const std::string& filename)
{
  std::ofstream* output = new std::ofstream(filename, std::ios::out);
  if (!output->good())
  {
    std::cerr << "Error: Could not open output file " << output << "! \n" << std::endl;
    exit(1);
  }
  for (size_t i = 0; i < voxels.size(); ++i)
  {
    // Write in order (X,Z,Y)
    *output << voxels.at(i).x << " "  << voxels.at(i).z << " "  << voxels.at(i).y << "\n";
  }
  output->close();
}
*/

template <typename PointT>
const bool loadPointCloud(const std::string& file_path,
                          typename pcl::PointCloud<PointT>& cloud_out)
{
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(file_path.c_str(), cloud_out) == -1)
  {
    PCL_ERROR("Failed to load PCD file \n");
    return false;
  }

  return true;
}

// Get the linear index into a 1D array of voxels,
// for a voxel at location (ix, iy, iz).
const unsigned int getLinearIndex(const Voxel& voxel, const int grid_size)
{
  return voxel.x * (grid_size * grid_size) + voxel.z * grid_size + voxel.y;
}

const Voxel getGridIndex(const pcl::PointXYZ& point, const pcl::PointXYZ& translate, const uint voxel_grid_size, const float scale)
{
  // Needs to be signed to prevent overflow, because index can
  // be slightly negative which then gets rounded to zero.
  const int i = std::round(static_cast<float>(voxel_grid_size)*((point.x - translate.x) / scale) - 0.5);
  const int j = std::round(static_cast<float>(voxel_grid_size)*((point.y - translate.y) / scale) - 0.5);
  const int k = std::round(static_cast<float>(voxel_grid_size)*((point.z - translate.z) / scale) - 0.5);
  return Voxel(i, j, k);  
}

// Format a float number to ensure it always has at least one decimal place
// 0 --> 0.0
// 1.1 --> 1.1
// 1.10 --> 1.1
const std::string formatFloat(const float value)
{
  std::stringstream ss;
  ss << std::setprecision(6) << std::fixed << value;
  std::string str;
  ss.str().swap(str);
  size_t last_zero_idx = str.find_last_not_of("0") + 1;
  if (last_zero_idx == str.length())
  {
    // No trailing zeros
    return str;
  }
  if (str[last_zero_idx - 1] == '.')
  {
    // Last zero is after decimal point
    last_zero_idx += 1;
  }
  str.resize(last_zero_idx);
  return str;
}

int main(int argc, char **argv)
{
  if (argc < 4)
  {
    pcl::console::print_error("Syntax is: %s -d <voxel_grid_size [32 to 1024]> input.pcd output.binvox \n", argv[0]);
    return -1;
  }

  // Parse the command line arguments for .pcd and .ply files
  std::vector<int> pcd_file_indices = pcl::console::parse_file_extension_argument(argc, argv, ".pcd");
  std::vector<int> binvox_file_indices = pcl::console::parse_file_extension_argument(argc, argv, ".binvox");
  if (pcd_file_indices.size() != 1 || binvox_file_indices.size() != 1)
  {
    pcl::console::print_error("Need one input PCD file and one output Binvox file. \n");
    return -1;
  }

  // In binvox, default is 256, max 1024
  uint voxel_grid_size;
  pcl::console::parse_argument(argc, argv, "-d", voxel_grid_size);
  std::cout << "Voxel grid size: " << voxel_grid_size << std::endl;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  const std::string input_file(argv[pcd_file_indices[0]]);
  if (!loadPointCloud<pcl::PointXYZ>(input_file, *cloud))
  {
    return -1;
  }

  pcl::PointXYZ min_point;
  pcl::PointXYZ max_point;
  pcl::getMinMax3D(*cloud, min_point, max_point);

  // Calculate the scale factor so the longest side of the volume
  // is split into the desired number of voxels
  const float x_range = max_point.x - min_point.x;
  const float y_range = max_point.y - min_point.y;
  const float z_range = max_point.z - min_point.z;

  const float max_cloud_extent = std::max(std::max(x_range, y_range), z_range);
  const float voxel_size = max_cloud_extent / (static_cast<float>(voxel_grid_size) - 1.0);
  std::cout << "voxel_size = " << voxel_size << std::endl;

  const float scale = (static_cast<float>(voxel_grid_size) * max_cloud_extent) / (static_cast<float>(voxel_grid_size) - 1.0);

  std::cout << "Bounding box: "
            << "[" << min_point.x << ", " << min_point.y << ", " << min_point.z << "] - "
            << "[" << max_point.x << ", " << max_point.y << ", " << max_point.z << "]" << std::endl;

  // Calculate the PointCloud's translation from the origin.
  // We need to subtract half the voxel size, because points
  // are located in the center of the voxel grid. 
  float tx = min_point.x - voxel_size / 2.0;
  float ty = min_point.y - voxel_size / 2.0;
  float tz = min_point.z - voxel_size / 2.0;
  // Hack, change -0.0 to 0.0
  const float epsilon = 0.0000001;
  if ((tx > -epsilon) && (tx < 0.0))
  {
    tx = -1.0 * tx;
  }
  if ((ty > -epsilon) && (ty < 0.0))
  {
    ty = -1.0 * ty;
  }
  if ((tz > -epsilon) && (tz < 0.0))
  {
    tz = -1.0 * tz;
  }
  const pcl::PointXYZ translate(tx, ty, tz);
  std::cout << "Normalization transform: (1) translate ["
            << formatFloat(translate.x) << ", " << formatFloat(translate.y) << ", " << formatFloat(translate.z) << "], " << std::endl;
  std::cout << "                         (2) scale " << scale << std::endl;

  const unsigned int num_voxels = voxel_grid_size * voxel_grid_size * voxel_grid_size;

  // Voxelize the PointCloud into a linear array
  boost::dynamic_bitset<> voxels_bitset(num_voxels);
  for (pcl::PointCloud<pcl::PointXYZ>::iterator it = cloud->begin(); it != cloud->end(); ++it)
  {
    const Voxel voxel = getGridIndex(*it, translate, voxel_grid_size, scale);
    const unsigned int idx = getLinearIndex(voxel, voxel_grid_size);
    voxels_bitset[idx] = 1;
  }

  /*
  // For debugging: Write voxel indices to a file
  std::vector<Voxel> voxels; // for debugging
  for (size_t i = 0; i < voxels_bitset.size(); ++i)
  {
    if (voxels_bitset[i] == 1)
    {
      const int voxel_grid_width = voxel_grid_size;
      const int voxel_grid_height = voxel_grid_size;
      const int idx = static_cast<int>(i);
      const float ix = static_cast<float>(idx / (voxel_grid_width * voxel_grid_height));
      const float iy = static_cast<float>(idx % voxel_grid_size);
      const float iz = static_cast<float>((idx / voxel_grid_width) % voxel_grid_height);
      voxels.push_back( Voxel(ix, iy, iz) );
    }
  }
  writeVoxelsToFile(voxels, "voxels_from_pcd.txt");
  */

  const std::string output_file(argv[binvox_file_indices[0]]);
  std::ofstream* output = new std::ofstream(output_file, std::ios::out | std::ios::binary);
  if (!output->good())
  {
    std::cerr << "Error: Could not open output file " << output << "!" << std::endl;
    exit(1);
  }

  // Write the binvox file using run-length encoding
  // where each pair of bytes is of the format (run value, run length)
  *output << "#binvox 1\n";
  *output << "dim " << voxel_grid_size << " " << voxel_grid_size << " " << voxel_grid_size << "\n";
  *output << "translate " << formatFloat(translate.x) << " " << formatFloat(translate.y) << " " << formatFloat(translate.z) << "\n";
  *output << "scale " << scale << "\n";
  *output << "data\n";
  unsigned int run_value = voxels_bitset[0];
  unsigned int run_length = 0;
  for (size_t i = 0; i < num_voxels; ++i)
  {
    if (voxels_bitset[i] == run_value)
    {
      // This is a run (repeated bit value)
      run_length++;
      if (run_length == 255)
      {
        *output << static_cast<char>(run_value);
        *output << static_cast<char>(run_length);
        run_length = 0;
      }
    }
    else
    {
      // End of a run
      *output << static_cast<char>(run_value);
      *output << static_cast<char>(run_length);
      run_value = voxels_bitset[i];
      run_length = 1;
    }
  }
  if (run_length > 0)
  {
    *output << static_cast<char>(run_value);
    *output << static_cast<char>(run_length);
  }
  output->close();

  std::cout << "done" << std::endl << std::endl;
  return 0;
}
