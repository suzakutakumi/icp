#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp6d.h>
#include <pcl/registration/gicp.h>
#include <pcl/filters/voxel_grid.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

using std::placeholders::_1;

class ICPNode : public rclcpp::Node
{
public:
  ICPNode()
      : Node("icp_node")
  {
    this->declare_parameter("max_iteration", 10);
    this->declare_parameter("max_correspondence_distance", std::sqrt(std::numeric_limits<double>::max()));
    this->declare_parameter("transform_epsilon", 0.0);
    this->declare_parameter("euclidean_fitness_epsilon", 0.0);
    this->declare_parameter("ref_prev_pose", false);
    this->declare_parameter("init_pos", std::vector<double>{});

    auto init_pos = this->get_parameter("init_pos").as_double_array();
    if (init_pos.size() >= 3)
    {
      global_transform.block<3, 1>(0, 3) << init_pos[0], init_pos[1], init_pos[2];
    }

    result_publisher = this->create_publisher<publisher_type>("icp/result", 10);
    map_publisher = this->create_publisher<publisher_type>("icp/map", 10);
    best_pose_tmp_publisher = this->create_publisher<std_msgs::msg::Float64MultiArray>("best_pose_tmp", 10);

    map_subscriber = this->create_subscription<subscribe_type>("init_map", 10, std::bind(&ICPNode::map_callback, this, _1));
    sensor_subscriber = this->create_subscription<subscribe_type>("target_pointcloud", 10, std::bind(&ICPNode::sensor_callback, this, _1));

    timer_ = this->create_wall_timer(std::chrono::milliseconds(10000), std::bind(&ICPNode::timer_callback, this));
  }

private:
  using subscribe_type = sensor_msgs::msg::PointCloud2;
  using publisher_type = sensor_msgs::msg::PointCloud2;

  rclcpp::Subscription<subscribe_type>::SharedPtr sensor_subscriber;
  rclcpp::Subscription<subscribe_type>::SharedPtr map_subscriber;

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<publisher_type>::SharedPtr result_publisher;
  rclcpp::Publisher<publisher_type>::SharedPtr map_publisher;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr best_pose_tmp_publisher;

  pcl::PointCloud<pcl::PointXYZRGB> map;

  Eigen::Matrix4d global_transform = Eigen::Matrix4d::Identity();

  using ICP_Type = pcl::PointXYZ;

  void map_callback(subscribe_type::UniquePtr msg)
  {
    RCLCPP_INFO_STREAM(get_logger(), "init map");
    pcl::fromROSMsg(*msg, map);
  }

  void sensor_callback(subscribe_type::UniquePtr msg)
  {
    RCLCPP_INFO_STREAM(get_logger(), "scan");
    if (map.size() <= 0)
    {
      return;
    }

    pcl::PointCloud<pcl::PointXYZRGB> sensor;
    pcl::fromROSMsg(*msg, sensor);

    RCLCPP_INFO_STREAM(get_logger(), "convert");
    pcl::PointCloud<ICP_Type> map_xyz, sensor_xyz;
    pcl::copyPointCloud(map, map_xyz);
    pcl::copyPointCloud(sensor, sensor_xyz);

    RCLCPP_INFO_STREAM(get_logger(), "icp");
    pcl::PointCloud<ICP_Type> result;
    pcl::IterativeClosestPoint<ICP_Type, ICP_Type> icp;
    // pcl::GeneralizedIterativeClosestPoint<ICP_Type,ICP_Type> icp;

    auto max_iteration = this->get_parameter("max_iteration").as_int();
    auto max_correspondence_distance = this->get_parameter("max_correspondence_distance").as_double();
    auto transform_epsilon = this->get_parameter("transform_epsilon").as_double();
    auto euclidean_fitness_epsilon = this->get_parameter("euclidean_fitness_epsilon").as_double();

    if (not this->get_parameter("ref_prev_pose").as_bool())
    {
      global_transform = Eigen::Matrix4d::Identity();
    }
    pcl::transformPointCloud(sensor_xyz, sensor_xyz, global_transform);

    icp.setMaximumIterations(max_iteration);                       // 1回の呼び出しでの最大反復回数
    icp.setMaxCorrespondenceDistance(max_correspondence_distance); // 対応点の最大距離
    icp.setTransformationEpsilon(transform_epsilon);               // 収束条件：変換行列の変化量
    icp.setEuclideanFitnessEpsilon(euclidean_fitness_epsilon);     // 収束条件：対応点間の平均二乗誤差

    icp.setInputTarget(map_xyz.makeShared());
    icp.setInputSource(sensor_xyz.makeShared());
    icp.align(result);

    if (not icp.hasConverged())
    {
      RCLCPP_WARN(get_logger(), "ICP has not converged");
      return;
    }
    RCLCPP_INFO_STREAM(get_logger(), "ICP has converged, score is " << icp.getFitnessScore());

    Eigen::Matrix4d tmat = Eigen::Matrix4d::Identity();

    RCLCPP_INFO_STREAM(get_logger(), "publish icp's result");
    publisher_type result_msg;
    pcl::toROSMsg(result, result_msg);
    result_msg.header.set__frame_id("nemui");
    result_publisher->publish(result_msg);

    global_transform = icp.getFinalTransformation().cast<double>() * global_transform;

    // 平行移動成分を取得
    Eigen::Vector3d translation = global_transform.block<3, 1>(0, 3);

    // 回転行列を取得
    Eigen::Matrix3d rotation = global_transform.block<3, 3>(0, 0);

    // オイラー角を取得 (ZYX順: yaw-pitch-roll)
    Eigen::Vector3d rpy = rotation.eulerAngles(2, 1, 0); // ZYX順
    double pitch = std::atan2(-rotation(2,0), std::sqrt(rotation(2,1) * rotation(2,1) + rotation(2,2) * rotation(2,2)));
    pitch *= 180 / M_PI;

    std_msgs::msg::Float64MultiArray::UniquePtr best_pose_tmp(new std_msgs::msg::Float64MultiArray);
    best_pose_tmp->data = std::vector<double>{translation[0], translation[2], pitch};
    best_pose_tmp->data.push_back(0.0);
    best_pose_tmp_publisher->publish(std::move(best_pose_tmp));

    RCLCPP_INFO_STREAM(get_logger(), "global_transform is " << std::endl
                                                            << translation.transpose() << std::endl
                                                            << rpy.transpose());

    tmat = global_transform.cast<double>();
    pcl::PointCloud<ICP_Type> sensor_;
    pcl::transformPointCloud(sensor_xyz, sensor_, tmat);
    sensor_xyz = sensor_;

    pcl::PointCloud<pcl::PointXYZRGB> converted_sensor;
    pcl::transformPointCloud(sensor, converted_sensor, tmat);
    // map += converted_sensor;
  }

  void timer_callback()
  {
    if (map.size() <= 0)
    {
      return;
    }

    pcl::PointCloud<pcl::PointXYZRGB> map_down;
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud(map.makeShared());
    sor.setLeafSize(0.05, 0.05, 0.05);
    sor.filter(map_down);

    publisher_type msg;
    pcl::toROSMsg(map_down, msg);

    msg.header.set__frame_id("nemui");
    map_publisher->publish(msg);
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ICPNode>());
  rclcpp::shutdown();

  return 0;
}