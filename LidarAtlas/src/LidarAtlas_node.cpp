#include <ros/ros.h>
#include <chunkmap_ros/chunk_map_service.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/Path.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/conditional_removal.h>

#include "gd/lidarRoadDetect.h"
#include "gd/ground_detect.h"
#include "gd/map_builder.h"

class Node
{
    ChunkMap::Ptr chunk_map;
    GroundDetect::Ptr ground_detector;
    MapBuilder::Ptr map_builder;

    ros::Subscriber subLidar;
    ros::Subscriber subPath;
    ros::Publisher pubLaserScan;
    ros::Timer drawTimer;

    std::string virtual_scan_frame;

    std::vector<size_t> need_update_submaps;

    std::string save_path;

public:
    Node(ros::NodeHandle &nh)
    {
        float submap_radius;
        float robot_height;
        float resolution;
        float chunk_base;
        float hit_probability;
        float miss_probability;
        nh.param<float>("resolution", resolution, 0.1f);
        nh.param<float>("chunk_base", chunk_base, 3.0f);
        nh.param<float>("hit_probability", hit_probability, 0.45f);
        nh.param<float>("miss_probability", miss_probability, 0.51f);
        nh.param<float>("submap_radius", submap_radius, 20.0f);
        nh.param<float>("robot_height", robot_height, 0.5f);
        nh.param<std::string>("save_path", save_path, "");
        if(save_path != "") ROS_INFO_STREAM("will be saved to " << save_path);
        // init chunk map
        chunk_map = std::make_shared<ChunkMapService>(ChunkMap::Config{
            .resolution = resolution,
            .chunk_base = chunk_base,
        }, nh);
        chunk_map->desc_type = ChunkMap::DescType::compressed;
        chunk_map->obstacle_segments = obstacle_segments;
        chunk_map->obstacle_bits = obstacle_bits;
        // init ground detector
        ground_detector = GroundDetect::create(
            {
                .resolution = resolution,
                .hit_probability = hit_probability,
                .miss_probability = miss_probability,
                .submap_radius = submap_radius,
                .robot_height = robot_height,
            });
        map_builder = MapBuilder::create(
            {
                .resolution = resolution,
                .submap_radius = submap_radius,
            },
            chunk_map);

        std::string indexed_points_topic;
        std::string poses_topic;
        std::string virtual_scan_topic;
        nh.param<std::string>("indexed_points_topic", indexed_points_topic, "indexed_points");
        nh.param<std::string>("poses_topic", poses_topic, "indexed_poses");
        nh.param<std::string>("virtual_scan_topic", virtual_scan_topic, "/virtual_scan");

        nh.param<std::string>("virtual_scan_frame", virtual_scan_frame, "virtual_scan");

        subLidar = nh.subscribe<sensor_msgs::PointCloud2>(indexed_points_topic, /*2*/ 20, &Node::pointHandler, this);
        subPath = nh.subscribe<nav_msgs::Path>(poses_topic, 2, &Node::posesHandler, this);
        pubLaserScan = nh.advertise<sensor_msgs::LaserScan>(virtual_scan_topic, 2);
        drawTimer = nh.createTimer(ros::Duration(1), &Node::drawHandler, this);
        drawTimer.start();
    }

    ~Node()
    {
        if(save_path != "")
        {
            map_builder->saveMap(save_path);
        }
    }

    void pointHandler(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

        pcl::fromROSMsg(*msg, *cloud);
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
        if (cloud->size() == 0)
            return;

        size_t submap_index = std::stoi(msg->header.frame_id);
        auto detect_info = ground_detector->process(cloud);
        auto submap_info = std::make_shared<SubmapInfo>();
        submap_info->detect = detect_info;
        submap_info->has_old_pose = false;
        submap_info->inited = false;
        map_builder->submaps.emplace(submap_index, submap_info);

        // if (pubLaserScan.getNumSubscribers() > 0)
            publishScan(detect_info->map_info, msg->header);
        return;
    }

    void posesHandler(const nav_msgs::PathConstPtr &msg)
    {
        for (const auto &pose : msg->poses)
        {
            size_t submap_index = std::stoi(pose.header.frame_id);
            if (map_builder->submaps.find(submap_index) == map_builder->submaps.end())
                continue;

            auto &translation = pose.pose.position;
            auto &rotation = pose.pose.orientation;
            Eigen::Isometry3d pose_eigen = Eigen::Isometry3d::Identity();
            pose_eigen.translate(Eigen::Vector3d{translation.x, translation.y, translation.z});
            pose_eigen.rotate(Eigen::Quaterniond(rotation.w, rotation.x, rotation.y, rotation.z));

            const auto &current = map_builder->submaps.at(submap_index);
            if (current->inited)
            {
                Eigen::Isometry3d delta = current->pose.inverse() * pose_eigen;
                Eigen::Quaterniond q(delta.rotation());
                float x = delta.translation().cwiseAbs().x();
                float y = delta.translation().cwiseAbs().y();
                float z = delta.translation().cwiseAbs().z();
                float w = std::acos(q.w());
                if (x > 5e-2 || y > 5e-2 || z > 5e-2 || w > 0.143)
                {
                    need_update_submaps.push_back(submap_index);
                }
            }
            else
            {
                need_update_submaps.push_back(submap_index);
            }

            current->pose = pose_eigen;
            current->inited = true;
        }
    }

    void drawHandler(const ros::TimerEvent &)
    {
        auto ret = map_builder->draw_multi<8, 8>(need_update_submaps);
        std::static_pointer_cast<ChunkMapService>(chunk_map)->update(ret);
    }

private:
    void publishScan(const cv::Mat1f &hits, const std_msgs::Header &header)
    {
        cv::Mat1f finally = cv::Mat1f::ones(hits.size());
        finally = finally * 15;  //  if no obstacle  range = 15  szz
        hits.copyTo(finally, hits > 0);
        sensor_msgs::LaserScan outMsg;
        outMsg.header = header;
        outMsg.header.frame_id = virtual_scan_frame;
        outMsg.angle_min = 0;
        outMsg.angle_max = 2 * M_PI;
        outMsg.angle_increment = 2 * M_PI / LidarRoadDetect::H_Ang_Num;
        outMsg.time_increment = 0;
        outMsg.scan_time = 0;
        double mini, maxi;
        cv::minMaxIdx(finally, &mini, &maxi);
        outMsg.range_min = 0.1;
        outMsg.range_max = maxi+0.1;
        outMsg.ranges.reserve(LidarRoadDetect::H_Ang_Num);
        outMsg.ranges.assign((float *)finally.data, (float *)finally.data + LidarRoadDetect::H_Ang_Num);
        pubLaserScan.publish(outMsg);
    }
};

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "LidarAtlas");
    ros::NodeHandle nh("~");

    Node node{nh};

    ros::spin();

    return 0;
}
