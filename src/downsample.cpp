
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

void Downsampling(const pcl::PointCloud<pcl::PointXYZI>::Ptr& pcd_cloud, const double &leaf_size,
                  pcl::PointCloud<pcl::PointXYZI>::Ptr& pcd_cloud_ds) {
    pcl::VoxelGrid<pcl::PointXYZI> vg;
    vg.setInputCloud(pcd_cloud);
    vg.setLeafSize(leaf_size, leaf_size, leaf_size);
    vg.filter(*pcd_cloud_ds);
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " input.pcd leaf_size output.pcd" << std::endl;
        return -1;
    }

    std::string input_file = argv[1];
    double leaf_size = atof(argv[2]);
    std::string output_file;
    std::stringstream ss;
    ss << leaf_size;
    output_file = input_file.substr(0, input_file.size() - 4) + "_" + ss.str() + ".pcd";
    if (argc == 4)
        output_file = argv[3];

    pcl::PointCloud<pcl::PointXYZI>::Ptr pcd_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcd_cloud_ds(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::io::loadPCDFile(input_file, *pcd_cloud);
    Downsampling(pcd_cloud, leaf_size, pcd_cloud_ds);
    pcl::io::savePCDFile(output_file, *pcd_cloud_ds);
    std::cout << "Downsampled pcd saved to " << output_file << std::endl;
    return 0;
}
