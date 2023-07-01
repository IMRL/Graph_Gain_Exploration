#include <ros/ros.h>
#include <chunkmap_ros/chunk_map_client.h>

#include <sensor_msgs/PointCloud2.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/registration/icp.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include "gd/lidarRoadDetect.h"
#include "gd/ground_detect.h"

class Node
{
    ChunkMap::Ptr chunk_map;
    GroundDetect::Ptr ground_detector;

    ros::Subscriber subLidar;
    ros::Subscriber subOdom;

    ros::Publisher pubOdom;

    float submap_radius;

    std::map<ChunkMap::Index, pcl::PointCloud<pcl::PointXYZ>::Ptr> chunkObstacle;

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;

    Eigen::Matrix4f last_pose;

    void updateChunk(const std::set<ChunkMap::Index> &updateList)
    {
        for (const auto &it : updateList)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto &layer : chunk_map->at(it).getLayers())
            {
                const auto &obstacle = layer.obstacle_info;
                const auto &elevation = layer.elevation;
                for (int r = 0; r < obstacle.rows; r++)
                    for (int c = 0; c < obstacle.cols; c++)
                    {
                        uint64_t value = ObstacleTypeDesc_at(chunk_map->obstacle_segments, chunk_map->obstacle_bits, obstacle, r, c);
                        for (int b = 0; b < chunk_map->obstacle_segments; b++)
                            if ((value & (1 << b)) != 0)
                            {
                                pcl::PointXYZ p;
                                p.x = c * chunk_map->resolution() + it.x * chunk_map->chunkBase();
                                p.y = r * chunk_map->resolution() + it.y * chunk_map->chunkBase();
                                p.z = b + elevation(r, c);
                                cloud->push_back(p);
                            }
                    }
            }
            chunkObstacle[it] = cloud;
        }
    }

public:
    Node(ros::NodeHandle &nh)
    {
        last_pose = Eigen::Matrix4f::Identity();
        float robot_height;
        float hit_probability;
        float miss_probability;
        nh.param<float>("hit_probability", hit_probability, 0.45f);
        nh.param<float>("miss_probability", miss_probability, 0.51f);
        nh.param<float>("submap_radius", submap_radius, 20.0f);
        nh.param<float>("robot_height", robot_height, 0.5f);
        // init chunk map
        chunk_map = std::make_shared<ChunkMapClient>(nh, "/chunkmap", [&](const std::set<ChunkMap::Index> &arg)
                                                     {   // things to be done when updated
                                                        this->updateChunk(arg); 
                                                     });
        chunk_map->desc_type = ChunkMap::DescType::compressed;
        std::static_pointer_cast<ChunkMapClient>(chunk_map)->init();
        // init ground detector
        ground_detector = GroundDetect::create(
            {
                .resolution = chunk_map->resolution(),
                .hit_probability = hit_probability,
                .miss_probability = miss_probability,
                .submap_radius = submap_radius,
                .robot_height = robot_height,
            });

        std::string points_topic;
        nh.param<std::string>("points_topic", points_topic, "/velodyne_points");

        std::string odom_topic;
        // std::string new_topic =  "/robot_pose_ekf/odom_combined";
        std::string rbtlc_topic = "/odometry/filtered";
        nh.param<std::string>("odom_topic", odom_topic, rbtlc_topic);

        subLidar = nh.subscribe<sensor_msgs::PointCloud2>(points_topic, /*2*/ 1, &Node::pointHandler, this);
        subOdom = nh.subscribe<nav_msgs::Odometry>(odom_topic, /*2*/ 1, &Node::odomHandler, this);
        // subOdom = nh.subscribe<geometry_msgs::PoseWithCovarianceStamped>(odom_topic, /*2*/ 1, &Node::odomHandler, this);

        pubOdom = nh.advertise<nav_msgs::Odometry>("/odometry", 1);

        icp.setMaxCorrespondenceDistance(4.0f);
        icp.setMaximumIterations(30);
        icp.setEuclideanFitnessEpsilon(0.001);
    }

    void pointHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

        pcl::fromROSMsg(*msg, *cloud);
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
        if (cloud->size() == 0)
            return;

        auto detect_info = ground_detector->process(cloud);

        auto localPC = generateLocalCloud(chunk_map->obstacle_segments, chunk_map->obstacle_bits, detect_info->obstacle_info, chunk_map->resolution(), submap_radius);

        pcl::PointCloud<pcl::PointXYZ>::Ptr globalPC(new pcl::PointCloud<pcl::PointXYZ>);

        Eigen::Vector3f base_pose = last_pose.block<3, 1>(0, 3);
        ChunkMap::Index center_block{int(base_pose.x() / chunk_map->chunkBase()), int(base_pose.y() / chunk_map->chunkBase())};

        int step = int(std::ceil(100 / chunk_map->chunkBase()));

        for (int x = -step; x <= step; x++)
            for (int y = -step; y <= step; y++)
            {
                ChunkMap::Index it{center_block.x + x, center_block.y + y};
                if (chunkObstacle.find(it) != chunkObstacle.end())
                {
                    *globalPC += *chunkObstacle.at(it);
                }
            }

        icp.setInputTarget(globalPC);

        pcl::PointCloud<pcl::PointXYZ>::Ptr temp(new pcl::PointCloud<pcl::PointXYZ>);
        icp.setInputSource(localPC);
        icp.align(*temp, last_pose);
        Eigen::Matrix4f pose = icp.getFinalTransformation();

        // last_pose = pose;  // default update

        // odomHandler convert it to odom for ekf update
        nav_msgs::Odometry odom;
        odom.header.frame_id = "odom";  //"map";
        odom.header.stamp = ros::Time::now();
        odom.child_frame_id = "base_link_lidar";

        Eigen::Vector3f T = pose.block<3, 1>(0, 3);
        Eigen::Quaternionf Q = Eigen::Quaternionf(pose.block<3, 3>(0, 0));

        odom.pose.pose.position.x = T.x();
        odom.pose.pose.position.y = T.y();
        odom.pose.pose.position.z = T.z();
        odom.pose.pose.orientation.w = Q.w();
        odom.pose.pose.orientation.x = Q.x();
        odom.pose.pose.orientation.y = Q.y();
        odom.pose.pose.orientation.z = Q.z();

        odom.pose.covariance[0*6 + 0] = 0.1;
        odom.pose.covariance[1*6 + 1] = 0.1;
        odom.pose.covariance[2*6 + 2] = 0.1;
        odom.pose.covariance[3*6 + 3] = 0.1;
        odom.pose.covariance[4*6 + 4] = 0.1;
        odom.pose.covariance[5*6 + 5] = 0.1;

        // odom.twist.twist.linear.x = 0.0;
        // odom.twist.twist.linear.y = 0.0;
        // odom.twist.twist.linear.z = 0.0;

        // odom.twist.covariance[0*6 + 0] = 0.001;
        // odom.twist.covariance[1*6 + 1] = 0.001;
        // odom.twist.covariance[2*6 + 2] = 0.001;

        pubOdom.publish(odom);

        return;
    }

    void odomHandler(const nav_msgs::OdometryConstPtr &msg) {
        Eigen::Matrix4f new_pose = Eigen::Matrix4f::Identity();
        new_pose.block<3, 1>(0, 3) = Eigen::Vector3f(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
        new_pose.block<3, 3>(0, 0) = Eigen::Quaternionf(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z).toRotationMatrix();
        last_pose = new_pose;
    }

    // void odomHandler(const geometry_msgs::PoseWithCovarianceStampedConstPtr &msg) {
    //     Eigen::Matrix4f new_pose = Eigen::Matrix4f::Identity();
    //     new_pose.block<3, 1>(0, 3) = Eigen::Vector3f(msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
    //     new_pose.block<3, 3>(0, 0) = Eigen::Quaternionf(msg->pose.pose.orientation.w, msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z).toRotationMatrix();
    //     last_pose = new_pose;
    // }

private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr generateLocalCloud(const ObstacleFeature::Segments &seg, const ObstacleFeature::Bits &bit, const ObstacleTypeDesc::Type &obstacle, float resolution, float radius)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (int r = 0; r < obstacle.rows; r++)
            for (int c = 0; c < obstacle.cols; c++)
            {
                uint64_t value = ObstacleTypeDesc_at(seg, bit, obstacle, r, c);
                for (int b = 0; b < seg; b++)
                    if ((value & (1 << b)) != 0)
                    {
                        pcl::PointXYZ p;
                        p.x = c * resolution - radius;
                        p.y = r * resolution - radius;
                        p.z = b;
                        cloud->push_back(p);
                    }
            }
        return cloud;
    }
};

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "localization");
    ros::NodeHandle nh("~");

    Node node{nh};

    ros::spin();

    return 0;
}
