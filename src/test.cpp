#include <ros/ros.h>

#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/GetMap.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>  //直通滤波器头文件
#include <pcl/filters/voxel_grid.h>  //体素滤波器头文件
#include <pcl/filters/statistical_outlier_removal.h>   //统计滤波器头文件
#include <pcl/filters/conditional_removal.h>    //条件滤波器头文件
#include <pcl/filters/radius_outlier_removal.h>   //半径滤波器头文件

std::string pcd_file;

std::string map_topic_name;

nav_msgs::OccupancyGrid map_topic_msg;

double thre_z_min = 0.3;
double thre_z_max = 2.0;
int flag_pass_through = 0;

double map_resolution = 0.05;

double thre_radius = 0.1;
int min_neighbors = 10;

pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_after_PassThrough(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_after_Radius(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr pcd_cloud(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr pcd_cloud_ds(new pcl::PointCloud<pcl::PointXYZI>);

void PassThroughFilter(const double& thre_low, const double& thre_high, const bool& flag_in);

void RadiusOutlierFilter(const pcl::PointCloud<pcl::PointXYZI>::Ptr& pcd_cloud, const double &radius, const int &thre_count);

void SetMapTopicMsg(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, nav_msgs::OccupancyGrid& msg);

void DownSampling(const pcl::PointCloud<pcl::PointXYZI>::Ptr& pcd_cloud, const double &leaf_size);

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pcl_filters");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    ros::Rate loop_rate(1.0);

    private_nh.param("pcd_file", pcd_file, std::string("/path/to/pointcloudmap.pcd"));
    ROS_INFO("*** pcd_file = %s ***\n", pcd_file.c_str());

    private_nh.param("thre_z_min", thre_z_min, 0.2);
    private_nh.param("thre_z_max", thre_z_max, 0.6);
    private_nh.param("flag_pass_through", flag_pass_through, 0);
    private_nh.param("thre_radius", thre_radius, 0.1);
    private_nh.param("min_neighbors", min_neighbors, 10);
    private_nh.param("map_resolution", map_resolution, 0.05);
    private_nh.param("map_topic_name", map_topic_name, std::string("map"));
    Eigen::Vector3d gravity_w; // the gravity in the world frame used by fastlio mapping.
    std::string gravity;
    private_nh.getParam("gravity_vec", gravity);
    std::vector<double> gravity_vec;
    int start = 0;
    int j = 0;
    for (int i = 0; i < gravity.size(); ++i) {
      if (gravity[i] == ',') {
        std::string item = gravity.substr(start, i - start);
        gravity_vec.push_back(std::stod(item));
        start = i + 1;
        ++j;
      }
    }
    std::string item = gravity.substr(start, gravity.size() - start);
    gravity_vec.push_back(std::stod(item));

    gravity_w << gravity_vec[0], gravity_vec[1], gravity_vec[2];
    bool align_gravity = true;
    private_nh.param("align_gravity", align_gravity, true);

    Eigen::Quaterniond Wzup_q_w = Eigen::Quaterniond::FromTwoVectors(gravity_w, Eigen::Vector3d(0, 0, -1));
    Eigen::Vector3d gravity_Wzup = Wzup_q_w * gravity_w;
    std::cout << "gravity old w " << gravity_w.transpose() << " gravity Wzup" << gravity_Wzup.transpose() << std::endl;

    ros::Publisher map_topic_pub = nh.advertise<nav_msgs::OccupancyGrid>(map_topic_name, 1);

    if (pcl::io::loadPCDFile<pcl::PointXYZI> (pcd_file, *pcd_cloud) == -1)
    {
      PCL_ERROR ("Couldn't read file: %s \n", pcd_file.c_str());
      return (-1);
    }
    std::cout << "before first point " << pcd_cloud->points[0].x << " " << pcd_cloud->points[0].y << " " << pcd_cloud->points[0].z << std::endl;
    if (align_gravity) {
      for (auto& pt : pcd_cloud->points) {
        Eigen::Vector3d pt_w(pt.x, pt.y, pt.z);
        Eigen::Vector3d pt_w_rot = Wzup_q_w * pt_w;
        pt.x = pt_w_rot[0];
        pt.y = pt_w_rot[1];
        pt.z = pt_w_rot[2];
      }
      std::cout << "after first point " << pcd_cloud->points[0].x << " " << pcd_cloud->points[0].y << " " << pcd_cloud->points[0].z << std::endl;
    }

    std::string pcd_suff = ".pcd";
    std::string map_prefix = pcd_file.substr(0, pcd_file.size() - pcd_suff.size());

    // rename the original pcd file
    // std::string cmd = "mv " + pcd_file + " " + map_prefix + "_orig.pcd";
    // std::cout << "cmd: " << cmd << std::endl;
    // system(cmd.c_str());

    // save pcd file
    std::string fn = map_prefix + "_aligned.pcd";
    pcl::io::savePCDFile(fn, *pcd_cloud);

    std::cout << "初始点云数据点数：" << pcd_cloud->points.size() << std::endl;

    PassThroughFilter(thre_z_min, thre_z_max, bool(flag_pass_through));

    RadiusOutlierFilter(cloud_after_PassThrough, thre_radius, min_neighbors);

    SetMapTopicMsg(cloud_after_Radius, map_topic_msg);
    float leaf_size = 0.1;
    DownSampling(pcd_cloud, leaf_size);
    std::string fn2 = map_prefix + "_" + std::to_string(leaf_size) + ".pcd";
    pcl::io::savePCDFile(fn2, *pcd_cloud_ds);

    while(ros::ok())
    {
      map_topic_pub.publish(map_topic_msg);

      loop_rate.sleep();

      ros::spinOnce();
    }

    return 0;
}

void PassThroughFilter(const double &thre_low, const double &thre_high, const bool &flag_in)
{
    /*方法一：直通滤波器对点云进行处理。*/
    pcl::PassThrough<pcl::PointXYZI> passthrough;
    passthrough.setInputCloud(pcd_cloud);//输入点云
    passthrough.setFilterFieldName("z");//对z轴进行操作
    passthrough.setFilterLimits(thre_low, thre_high);//设置直通滤波器操作范围
    passthrough.setFilterLimitsNegative(flag_in);//true表示保留范围外，false表示保留范围内
    passthrough.filter(*cloud_after_PassThrough);//执行滤波，过滤结果保存在 cloud_after_PassThrough
    std::cout << "直通滤波后点云数据点数：" << cloud_after_PassThrough->points.size() << std::endl;
}

void RadiusOutlierFilter(const pcl::PointCloud<pcl::PointXYZI>::Ptr& pcd_cloud0, const double &radius, const int &thre_count)
{
    pcl::RadiusOutlierRemoval<pcl::PointXYZI> radiusoutlier;  //创建滤波器

    radiusoutlier.setInputCloud(pcd_cloud0);    //设置输入点云
    radiusoutlier.setRadiusSearch(radius);     //设置radius为100的范围内找临近点
    radiusoutlier.setMinNeighborsInRadius(thre_count); //设置查询点的邻域点集数小于2的删除

    radiusoutlier.filter(*cloud_after_Radius);
    std::cout << "半径滤波后点云数据点数：" << cloud_after_Radius->points.size() << std::endl;
}

void DownSampling(const pcl::PointCloud<pcl::PointXYZI>::Ptr& pcd_cloud, const double &leaf_size)
{
    pcl::VoxelGrid<pcl::PointXYZI> vg;
    vg.setInputCloud(pcd_cloud);
    vg.setLeafSize(leaf_size, leaf_size, leaf_size);
    vg.filter(*pcd_cloud_ds);
}

void SetMapTopicMsg(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, nav_msgs::OccupancyGrid& msg)
{
  msg.header.seq = 0;
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = "map";

  msg.info.map_load_time = ros::Time::now();
  msg.info.resolution = map_resolution;

  double x_min, x_max, y_min, y_max;
  double z_max_grey_rate = 0.05;
  double z_min_grey_rate = 0.95;
  double k_line = (z_max_grey_rate - z_min_grey_rate) / (thre_z_max - thre_z_min);
  double b_line = (thre_z_max * z_min_grey_rate - thre_z_min * z_max_grey_rate) / (thre_z_max - thre_z_min);

  if(cloud->points.empty())
  {
    ROS_WARN("pcd is empty!\n");

    return;
  }

  for(int i = 0; i < cloud->points.size() - 1; i++)
  {
    if(i == 0)
    {
      x_min = x_max = cloud->points[i].x;
      y_min = y_max = cloud->points[i].y;
    }

    double x = cloud->points[i].x;
    double y = cloud->points[i].y;

    if(x < x_min) x_min = x;
    if(x > x_max) x_max = x;

    if(y < y_min) y_min = y;
    if(y > y_max) y_max = y;
  }

  msg.info.origin.position.x = x_min;
  msg.info.origin.position.y = y_min;
  msg.info.origin.position.z = 0.0;
  msg.info.origin.orientation.x = 0.0;
  msg.info.origin.orientation.y = 0.0;
  msg.info.origin.orientation.z = 0.0;
  msg.info.origin.orientation.w = 1.0;

  msg.info.width = int((x_max - x_min) / map_resolution);
  msg.info.height = int((y_max - y_min) / map_resolution);

  msg.data.resize(msg.info.width * msg.info.height);
  msg.data.assign(msg.info.width * msg.info.height, 0);

  ROS_INFO("data size = %zu\n", msg.data.size());

  for(int iter = 0; iter < cloud->points.size(); iter++)
  {
    int i = int((cloud->points[iter].x - x_min) / map_resolution);
    if(i < 0 || i >= msg.info.width) continue;

    int j = int((cloud->points[iter].y - y_min) / map_resolution);
    if(j < 0 || j >= msg.info.height - 1) continue;

    msg.data[i + j * msg.info.width] = 100;
//    msg.data[i + j * msg.info.width] = int(255 * (cloud->points[iter].z * k_line + b_line)) % 255;
  }
}


