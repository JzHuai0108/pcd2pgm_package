
#include <cmath>
#include <cstdlib>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <sstream>

#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>              // PassThrough filter
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/voxel_grid.h>               // VoxelGrid filter
#include <pcl/filters/radius_outlier_removal.h>   // Radius Outlier Removal filter

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <gflags/gflags.h>

DEFINE_double(thresh_z_min, 0.3, "Minimum z threshold for PassThrough filter");
DEFINE_double(thresh_z_max, 2.0, "Maximum z threshold for PassThrough filter");
DEFINE_bool(pass_through, true, "If true, keep points outside the limits (setFilterLimitsNegative)");
DEFINE_double(thresh_radius, 0.1, "Radius for outlier removal filter");
DEFINE_int32(min_neighbors, 10, "Minimum number of neighbors for radius outlier removal");
DEFINE_bool(align_gravity, true, "Align point cloud with gravity vector");
DEFINE_bool(align_xy, true, "Align the 3D point cloud normals to xy axes");
DEFINE_double(heading_offset_deg, 0.0, "Heading offset in degrees");
DEFINE_double(voxel_leaf_size, 0.1, "Leaf size for downsampling using VoxelGrid");
DEFINE_bool(downsampling, false, "If true, downsample the point cloud using PassThroughFilter, RadiusOutlierFilter, VoxelGrid");

pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_after_PassThrough(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_after_Radius(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr pcd_cloud(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr pcd_cloud_ds(new pcl::PointCloud<pcl::PointXYZI>);

void PassThroughFilter(const double thre_low, const double thre_high, const bool flag_neg);
void RadiusOutlierFilter(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud, const double radius, const int min_neighbors);
void DownSampling(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud, const double leaf_size);

float angleDiffDeg(float a, float b) {
    float diff = std::fmod(std::fabs(a - b), 360.0f);
    return diff > 180.0f ? 360.0f - diff : diff;
}

Eigen::Matrix3f computeHeadingAlignmentRotation(pcl::PointCloud<pcl::PointXYZI>::ConstPtr cloud) {
    // Step 1: Estimate normals
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    ne.setSearchMethod(tree);
    ne.setKSearch(20);  // Tune this based on density
    ne.compute(*normals);

    // Step 2: Build histogram of normal azimuth angles (horizontal projection)
    constexpr int NUM_BINS = 360; // We could have changed this to a histogram of [0, 180], but 
    // it may lose accuracy compared to distinguishing the two rays on a line.
    std::vector<int> histogram(NUM_BINS, 0);
    for (const auto& n : normals->points) {
        Eigen::Vector2f horiz(n.normal_x, n.normal_y);
        if (horiz.norm() < 1e-3) continue;  // Skip near-vertical normals
        float angle = std::atan2(n.normal_y, n.normal_x);  // [-π, π]
        int bin = static_cast<int>((angle + M_PI) * 180.0 / M_PI) % NUM_BINS; // [0, 360]
        histogram[bin]++;
    }

    // Step 3: Find two dominant directions from histogram
    std::vector<std::pair<int, int>> bin_counts;
    for (int i = 0; i < NUM_BINS; ++i) {
        bin_counts.emplace_back(histogram[i], i);
    }
    std::sort(bin_counts.rbegin(), bin_counts.rend());  // descending order

    auto getSubBinPeakAngle = [&](const std::vector<int>& hist, int peak_bin) {
        int i0 = (peak_bin - 1 + NUM_BINS) % NUM_BINS;
        int i1 = peak_bin;
        int i2 = (peak_bin + 1) % NUM_BINS;
    
        float y0 = hist[i0], y1 = hist[i1], y2 = hist[i2];
        float denominator = (y0 - 2 * y1 + y2);
        float offset = (denominator != 0.0f) ? 0.5f * (y0 - y2) / denominator : 0.0f;
    
        float peak_bin_f = peak_bin + offset;
        float peak_angle = (peak_bin_f * M_PI / 180.0f) - M_PI;  // map to [-π, π]
        return peak_angle;
    };

    // Convert bin index to degrees in [0, 360)
    auto binToDeg = [](float bin_idx) {
        return std::fmod((bin_idx * 1.0f), 360.0f);
    };

    int peak_bin_1 = bin_counts[0].second;
    float dominant_angle_1_deg = binToDeg(peak_bin_1);
    float dominant_angle_1 = getSubBinPeakAngle(histogram, peak_bin_1);

    // Mask out ±20° range from histogram
    std::vector<bool> suppressed(NUM_BINS, false);
    for (int i = -20; i <= 20; ++i) {
        int suppressed_bin = (peak_bin_1 + i + NUM_BINS) % NUM_BINS;
        suppressed[suppressed_bin] = true;
    }

    // Find second peak ≈ 90° different
    int peak_bin_2 = -1;
    int max_val = -1;
    for (int i = 0; i < NUM_BINS; ++i) {
        if (suppressed[i]) continue;
        float angle_i_deg = binToDeg(i);
        float diff = angleDiffDeg(angle_i_deg, dominant_angle_1_deg);
        if (diff >= 70.0f && diff <= 110.0f && histogram[i] > max_val) {
            max_val = histogram[i];
            peak_bin_2 = i;
        }
    }

    float dominant_angle_2 = getSubBinPeakAngle(histogram, peak_bin_2);
    float dominant_angle_2_deg = binToDeg(peak_bin_2);
    std::cout << "Found two dominant angles: " << dominant_angle_1_deg << "° and " << dominant_angle_2_deg << "°, with diff of "
              << angleDiffDeg(dominant_angle_1_deg, dominant_angle_2_deg) << "°." << std::endl;
    std::cout << "dorminant_angle_1: " << dominant_angle_1 << " rad, dominant_angle_2: " << dominant_angle_2 << " rad." << std::endl;
    std::string histogram_file = "histogram.txt";
    std::ofstream hist_out(histogram_file);
    if (hist_out.is_open()) {
        for (int i = 0; i < NUM_BINS; ++i) {
            hist_out << i << " " << histogram[i] << "\n";
        }
        hist_out.close();
        std::cout << "Histogram saved to " << histogram_file << std::endl;
    } else {
        std::cerr << "Error: Unable to open histogram file for writing." << std::endl;
    }
    // Step 4: Choose the angle closest to 0 as heading
    float heading_angle = dominant_angle_2;
    // for some cases, we have to adjust this manually to fix the misalignment for unknown reasons.
    if (heading_angle > M_PI * 0.5) {
        heading_angle -= M_PI * 0.5;
    }
    heading_angle += (FLAGS_heading_offset_deg * M_PI / 180.0f);
    Eigen::Matrix3f rotation;
    rotation = Eigen::AngleAxisf(-heading_angle, Eigen::Vector3f::UnitZ()).toRotationMatrix();
    return rotation;
}

// Helper function to parse a gravity vector string "x,y,z"
std::vector<double> ParseGravityVector(const std::string& gravity_str)
{
    std::vector<double> gravity_vec;
    std::stringstream ss(gravity_str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        try {
            gravity_vec.push_back(std::stod(item));
        } catch (const std::exception& e) {
            std::cerr << "Error parsing gravity vector component: " << item << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    return gravity_vec;
}

int main(int argc, char** argv)
{
    // Allow unknown flags so we can manually parse positional args like negative gravity values
    gflags::AllowCommandLineReparsing();
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Check that exactly 2 positional arguments are passed after gflags parsing
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " [gflags] <pcd_file> <gravity_vector>" << std::endl;
        std::cerr << "Example: " << argv[0] << " --downsampling=true -- pointcloud.pcd \"-0.3,0,-9.8\"" << std::endl;
        return EXIT_FAILURE;
    }

    // argv[1] and argv[2] are positional args after `--`
    std::string pcd_file = argv[1];
    std::string gravity_str = argv[2];

    // Parse gravity vector safely
    std::vector<double> gravity_vec = ParseGravityVector(gravity_str);
    if (gravity_vec.size() != 3) {
        std::cerr << "Error: Gravity vector must have exactly 3 components, e.g., \"0,0,-9.8\"" << std::endl;
        return EXIT_FAILURE;
    }

    Eigen::Vector3d gravity_w(gravity_vec[0], gravity_vec[1], gravity_vec[2]);

    // Compute the quaternion that rotates the gravity vector to align with (0, 0, -1)
    Eigen::Quaterniond Wzup_q_w = Eigen::Quaterniond::FromTwoVectors(gravity_w, Eigen::Vector3d(0, 0, -1));
    Eigen::Vector3d gravity_Wzup = Wzup_q_w * gravity_w;
    std::cout << "Original gravity vector: " << gravity_w.transpose()
              << " | Aligned gravity vector: " << gravity_Wzup.transpose() << std::endl;

    if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file, *pcd_cloud) == -1)
    {
        PCL_ERROR("Couldn't read file: %s \n", pcd_file.c_str());
        return -1;
    }
    std::cout << "Loaded point cloud with " << pcd_cloud->points.size() << " points." << std::endl;
    if (pcd_cloud->points.empty()) {
        std::cerr << "Error: Loaded point cloud is empty." << std::endl;
        return -1;
    }

    std::cout << "First point before alignment: ("
              << pcd_cloud->points[0].x << ", "
              << pcd_cloud->points[0].y << ", "
              << pcd_cloud->points[0].z << ")" << std::endl;

    if (FLAGS_align_gravity) {
        for (auto& pt : pcd_cloud->points) {
            Eigen::Vector3d pt_w(pt.x, pt.y, pt.z);
            Eigen::Vector3d pt_rot = Wzup_q_w * pt_w;
            pt.x = pt_rot.x();
            pt.y = pt_rot.y();
            pt.z = pt_rot.z();
        }
        std::cout << "First point after gravity alignment: ("
                  << pcd_cloud->points[0].x << ", "
                  << pcd_cloud->points[0].y << ", "
                  << pcd_cloud->points[0].z << ")" << std::endl;
    }

    // Generate output file prefix by removing the ".pcd" suffix if present.
    std::string pcd_suffix = ".pcd";
    std::string map_prefix = pcd_file;
    if (pcd_file.size() > pcd_suffix.size() && pcd_file.substr(pcd_file.size() - pcd_suffix.size()) == pcd_suffix)
    {
        map_prefix = pcd_file.substr(0, pcd_file.size() - pcd_suffix.size());
    }
    std::string aligned_file = map_prefix + "_gravity.pcd";
    pcl::io::savePCDFile(aligned_file, *pcd_cloud);
    std::cout << "Gravity aligned point cloud saved to " << aligned_file << std::endl;

    if (FLAGS_align_xy) {
        auto rotation_matrix = computeHeadingAlignmentRotation(pcd_cloud);
        pcl::transformPointCloud(*pcd_cloud, *pcd_cloud, Eigen::Affine3f(rotation_matrix));
        std::string aligned_file = map_prefix + "_aligned.pcd";
        pcl::io::savePCDFile(aligned_file, *pcd_cloud);
        std::cout << "Heading aligned point cloud saved to " << aligned_file << std::endl;
    }

    if (FLAGS_downsampling) {
        std::cout << "Original point cloud size: " << pcd_cloud->points.size() << " points." << std::endl;
        // Apply the PassThrough filter on the z-axis.
        PassThroughFilter(FLAGS_thresh_z_min, FLAGS_thresh_z_max, FLAGS_pass_through);
        RadiusOutlierFilter(cloud_after_PassThrough, FLAGS_thresh_radius, FLAGS_min_neighbors);
        DownSampling(pcd_cloud, FLAGS_voxel_leaf_size);
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(3) << FLAGS_voxel_leaf_size;
        std::string ds_file = map_prefix + "_" + oss.str() + ".pcd";
        pcl::io::savePCDFile(ds_file, *pcd_cloud_ds);
        std::cout << "Downsampled point cloud of " << pcd_cloud_ds->points.size() << " points saved to " << ds_file << std::endl;
    }
    return 0;
}

void PassThroughFilter(const double thre_low, const double thre_high, const bool flag_neg)
{
    pcl::PassThrough<pcl::PointXYZI> pass;
    pass.setInputCloud(pcd_cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(thre_low, thre_high);
    pass.setNegative(flag_neg);  // When true, keep points outside the limits.
    pass.filter(*cloud_after_PassThrough);
    std::cout << "After PassThrough filter, point cloud size: " 
              << cloud_after_PassThrough->points.size() << " points." << std::endl;
}

void RadiusOutlierFilter(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud, const double radius, const int min_neighbors)
{
    pcl::RadiusOutlierRemoval<pcl::PointXYZI> ror;
    ror.setInputCloud(input_cloud);
    ror.setRadiusSearch(radius);
    ror.setMinNeighborsInRadius(min_neighbors);
    ror.filter(*cloud_after_Radius);
    std::cout << "After Radius Outlier Removal filter, point cloud size: " 
              << cloud_after_Radius->points.size() << " points." << std::endl;
}

void DownSampling(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud, const double leaf_size)
{
    pcl::VoxelGrid<pcl::PointXYZI> vg;
    vg.setInputCloud(input_cloud);
    vg.setLeafSize(leaf_size, leaf_size, leaf_size);
    vg.filter(*pcd_cloud_ds);
}
