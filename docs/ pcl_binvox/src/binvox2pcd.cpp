//
// binvox2pcd
// Convert a .binvox file to .pcd
//
// binvox is a binary format for a 3D voxel grid.
// pcd is Point Cloud Data from PCL (PointCloud Library).
//
// David Butterworth, 2016.
//
// binvox was developed by Patrick Min (www.patrickmin.com)
// The code below is based on Patrick's read_binvox.cc
// and binvox2bt.cpp by Stefan Osswald.
//

#include <string>
#include <fstream>
#include <iostream>

#include <cstdlib>
#include <cstring>

#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>


// TODO: - Test input of multiple binvox files
//       - Use boost to check for missing file extension


/*
struct Voxel
{
  unsigned int x;
  unsigned int y;
  unsigned int z;
  Voxel() : x(0), y(0), z(0) {};
  Voxel(const unsigned int _x, const unsigned int _y, const unsigned int _z) : x(_x), y(_y), z(_z) {};
};

// For debugging: Write the voxel indices to a file
void writeVoxelsToFile(const std::vector<Voxel>& voxels, const std::string& filename)
{
  std::ofstream* output = new std::ofstream(filename, std::ios::out);
  if (!output->good())
  {
    std::cerr << "Error: Could not open output file " << output << "!" << std::endl;
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

int main(int argc, char **argv)
{
  bool show_help = false;
  if (argc == 1)
  {
    show_help = true;
  }
  for (int i = 1; i < argc && !show_help; ++i)
  {
    if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-help") == 0 ||
        strcmp(argv[i], "--usage") == 0 || strcmp(argv[i], "-usage") == 0 ||
        strcmp(argv[i], "-h") == 0)
    {
      show_help = true;
    }
  }
  if (show_help)
  {
    std::cout << "Usage: " << argv[0] << " [OPTIONS] <binvox filenames>" << std::endl;
    std::cout << "\tOPTIONS:" << std::endl;
    std::cout << "\t -o <file>        Output filename (default: first input filename + .pcd)\n";
    std::cout << "\t --mark-free      Mark not occupied cells as 'free' (default: unknown)\n";
    std::cout << "\t --bb <bbox_min_x> <bbox_min_y> <bbox_min_z> <bbox_max_x> <bbox_max_y> <bbox_max_z>: enforce bounding box\n";
    std::cout << "All options apply to the subsequent input files.\n\n";
    exit(0);
  }

  std::string output_filename;
  double bbox_min_x = 0.0;
  double bbox_min_y = 0.0;
  double bbox_min_z = 0.0;
  double bbox_max_x = 0.0;
  double bbox_max_y = 0.0;
  double bbox_max_z = 0.0;
  bool apply_bounding_box = false;
  std::vector<std::string> input_files;
    
  // Parse the command line arguments
  for (int i = 1; i < argc; ++i)
  {
    if (strcmp(argv[i], "-o") == 0 && i < argc - 1)
    {            
      i++;
      output_filename = argv[i];
    }
    else if (strcmp(argv[i], "--bb") == 0 && i < argc - 7)
    {
      i++;
      bbox_min_x = atof(argv[i]);
      i++;
      bbox_min_y = atof(argv[i]);
      i++;
      bbox_min_z = atof(argv[i]);
      i++;
      bbox_max_x = atof(argv[i]);
      i++;
      bbox_max_y = atof(argv[i]);
      i++;
      bbox_max_z = atof(argv[i]);
      apply_bounding_box = true;
    }
    else
    {
      input_files.push_back(argv[i]);
    }
  }

  pcl::PointCloud<pcl::PointXYZ> cloud;
  //std::vector<Voxel> voxels; // for debugging

  for (size_t i = 0; i < input_files.size(); ++i)
  {
    const std::string input_file(input_files.at(i));

    std::ifstream* input = new std::ifstream(input_file, std::ios::in | std::ios::binary);    
    if (!input->good())
    {
      std::cerr << "Error: Could not open input file " << input_file << "!" << std::endl;
      exit(1);
    }
    else
    {
      std::cout << "Reading binvox file " << input_file << "." << std::endl;

      if (output_filename.empty())
      { 
        output_filename = std::string(input_file).append(".pcd");
      }
    }

    // Read binvox header
    std::string line;
    *input >> line;
    if (line.compare("#binvox") != 0)
    {
      std::cerr << "Error: first line reads [" << line << "] instead of [#binvox]" << std::endl;
      delete input;
      return 0;
    }
    int binvox_version;
    *input >> binvox_version;
    std::cout << "Detected binvox version " << binvox_version << std::endl;

    unsigned int voxel_grid_depth;
    unsigned int voxel_grid_height;
    unsigned int voxel_grid_width;
    double tx;
    double ty;
    double tz;
    double scale;

    voxel_grid_depth = -1;
    int done = 0;
    while (input->good() && !done)
    {
      *input >> line;
      if (line.compare("data") == 0)
      {
        done = 1;
      }
      else if (line.compare("dim") == 0)
      {
        *input >> voxel_grid_depth >> voxel_grid_height >> voxel_grid_width;
      }
      else if (line.compare("translate") == 0)
      {
        *input >> tx >> ty >> tz;
      }
      else if (line.compare("scale") == 0)
      {
        *input >> scale;
      }
      else
      {
        std::cout << "Unsuported keyword found in file [" << line << "], skipping" << std::endl;
        char c;
        // Jump to end of line
        do
        {
          c = input->get();
        }
        while (input->good() && (c != '\n'));
      }
    }
    if (!done)
    {
      std::cerr << "Error reading header" << std::endl;
      return 0;
    }

    if (voxel_grid_depth == -1)
    {
      std::cout << "Error: missing dimensions in header" << std::endl;
      return 0;
    }

    const int voxel_grid_size = voxel_grid_depth;

    std::cout << "Voxel grid size: " << voxel_grid_size << std::endl;
    std::cout << "Normalization transform: (1) translate ["
              << tx << ", " << ty << ", " << tz << "], " << std::endl;
    std::cout << "                         (2) scale " << scale << std::endl;

    if (apply_bounding_box)
    {
      std::cout << "Bounding box: ["
                << bbox_min_x << ","<< bbox_min_y << "," << bbox_min_z << " - "
                << bbox_max_x << ","<< bbox_max_y << "," << bbox_max_z << "] \n";
    }

    std::cout << "Read data: ";
    std::cout.flush();
        
    // Process voxel data
    unsigned char value;
    unsigned char count;
    int index = 0;
    int end_index = 0;
    unsigned num_voxels_read = 0;
    unsigned num_voxels_skipped = 0;
    
    input->unsetf(std::ios::skipws);
    *input >> value; // read the linefeed char

    const int num_voxels = voxel_grid_width * voxel_grid_height * voxel_grid_depth;
    while ((end_index < num_voxels) && input->good())
    {
      *input >> value >> count;

      if (input->good())
      {
        end_index = index + count;

        if (end_index > num_voxels)
        {
          return 0;
        }

        for (int i = index; i < end_index; ++i)
        {
          // Output progress dots
          if (i % (num_voxels / 20) == 0)
          {
            std::cout << ".";            
            std::cout.flush();
          }

          // Get the voxel indices
          const float iy = static_cast<float>(i % voxel_grid_width);
          const float iz = static_cast<float>((i / voxel_grid_width) % voxel_grid_height);
          const float ix = static_cast<float>(i / (voxel_grid_width * voxel_grid_height));

          // Convert voxel indices from integer to values between 0.0 to 1.0
          // The 0.5 aligns the point with the center of the voxel.
          const float x = (ix + 0.5) / static_cast<float>(voxel_grid_size);
          const float y = (iy + 0.5) / static_cast<float>(voxel_grid_size);
          const float z = (iz + 0.5) / static_cast<float>(voxel_grid_size);

          const double px = scale * static_cast<double>(x) + tx;
          const double py = scale * static_cast<double>(y) + ty;
          const double pz = scale * static_cast<double>(z) + tz;
          
          pcl::PointXYZ point(static_cast<float>(px),
                              static_cast<float>(py),
                              static_cast<float>(pz));

          if (!apply_bounding_box
                || (px <= bbox_max_x && px >= bbox_min_x
                    && py <= bbox_max_y && py >= bbox_min_y
                    && pz <= bbox_max_z && pz >= bbox_min_z))
          {
            if (value == 1)
            {
              cloud.points.push_back(point);

              // For debugging: Save the voxel indices
              //voxels.push_back(Voxel(static_cast<unsigned int>(ix),
              //                       static_cast<unsigned int>(iy),
              //                       static_cast<unsigned int>(iz)));
            }
          }
          else
          {
            num_voxels_skipped++;
          }
        }
        
        if (value)
        {
          num_voxels_read += count;
        }

        index = end_index;
      }
    }
    
    std::cout << std::endl << std::endl;

    input->close();
    std::cout << "Read " << num_voxels_read << " voxels";
    if (num_voxels_skipped > 0)
    {
      std::cout << ", skipped " << num_voxels_skipped << " (outside bounding box)";
    }

    std::cout << "\n" << std::endl;
  } // end processing each binvox file

  // For debugging: Write voxel indices to a file
  //writeVoxelsToFile(voxels, "voxels_from_binvox.txt");

  std::cout << "PointCloud has " << cloud.points.size() << " points";
  pcl::PointXYZ min_point;
  pcl::PointXYZ max_point;
  pcl::getMinMax3D(cloud, min_point, max_point);
  std::cout << ", with extents: "
            << "(" << min_point.x << ", " << min_point.y << ", " << min_point.z << ") to "
            << "(" << max_point.x << ", " << max_point.y << ", " << max_point.z << ")" << std::endl;

  pcl::PCDWriter writer;
  writer.writeBinaryCompressed(output_filename, cloud);

  std::cout << "done" << std::endl << std::endl;
  return 0;
}
