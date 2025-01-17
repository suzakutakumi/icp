#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
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
    result_publisher = this->create_publisher<publisher_type>("icp/result", 10);
    map_publisher = this->create_publisher<publisher_type>("icp/map", 10);

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

  pcl::PointCloud<pcl::PointXYZRGB> map;

  using ICP_Type = pcl::PointXYZRGB;

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
    // icp.setMaximumIterations(1000);        // 1回の呼び出しでの最大反復回数
    // icp.setMaxCorrespondenceDistance(0.1); // 対応点の最大距離
    // icp.setTransformationEpsilon(1e-8);    // 収束条件：変換行列の変化量
    // icp.setEuclideanFitnessEpsilon(1e-5);  // 収束条件：対応点間の平均二乗誤差

    Eigen::Matrix4d tmat = Eigen::Matrix4d::Identity();
    while (rclcpp::ok())
    {
      icp.setInputTarget(map_xyz.makeShared());
      icp.setInputSource(sensor_xyz.makeShared());
      icp.align(result);

      RCLCPP_INFO_STREAM(get_logger(), "publish icp's result");
      publisher_type result_msg;
      pcl::toROSMsg(result, result_msg);
      result_msg.header.set__frame_id("nemui");
      result_publisher->publish(result_msg);

      if (not icp.hasConverged())
      {
        RCLCPP_WARN(get_logger(), "ICP has not converged");
        return;
      }

      RCLCPP_INFO_STREAM(get_logger(), "ICP has converged, score is " << icp.getFitnessScore());

      tmat = icp.getFinalTransformation().cast<double>();

      pcl::PointCloud<ICP_Type> sensor_;
      pcl::transformPointCloud(sensor_xyz, sensor_, tmat);
      sensor_xyz = sensor_;

      if (icp.getFitnessScore() < 0.2)
      {
        break;
      }
    }
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