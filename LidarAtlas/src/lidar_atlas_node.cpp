#include <ros/ros.h>
#include <chunkmap_ros/chunk_map_service.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/Path.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/conditional_removal.h>

#include "gd/lidarRoadDetect.h"
#include "gd/ground_detect.h"

#define USE_SUBMAP

#ifndef USE_SUBMAP
#include "gd/map_builder.h"
#else
#include "gd/globalmap_builder.h"
#endif

class Node
{
    ChunkMap::Ptr chunk_map;
    GroundDetect::Ptr ground_detector;
#ifndef USE_SUBMAP
    MapBuilder::Ptr map_builder;
#else
    GlobalmapBuilder::Ptr map_builder;
#endif

    ros::Subscriber subLidar;
    ros::Subscriber subPath;
    ros::Publisher pubLaserScan;
    ros::Publisher pubObs;
    ros::Timer drawTimer;

    std::string virtual_scan_frame;

#ifndef USE_SUBMAP
    std::vector<size_t> need_update_submaps;
#else
    std::vector<GlobalmapBuilder::LocalmapIndex> need_update_submaps;
#endif

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
        if (save_path != "")
            ROS_INFO_STREAM("will be saved to " << save_path);
        // init chunk map
        chunk_map = std::make_shared<ChunkMapService>(ChunkMap::Config{
                                                          .resolution = resolution,
                                                          .chunk_base = chunk_base,
                                                      },
                                                      nh);
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
#ifndef USE_SUBMAP
        map_builder = MapBuilder::create(
            {
                .resolution = resolution,
                .submap_radius = submap_radius,
            },
            chunk_map);
#else
        map_builder = GlobalmapBuilder::create(
            {
                .resolution = resolution,
                .submap_radius = submap_radius,
            },
            chunk_map);
#endif

        std::string indexed_points_topic;
        std::string poses_topic;
        std::string virtual_scan_topic;
        nh.param<std::string>("indexed_points_topic", indexed_points_topic, "/indexed_points");
        nh.param<std::string>("poses_topic", poses_topic, "/indexed_poses");
        nh.param<std::string>("virtual_scan_topic", virtual_scan_topic, "/virtual_scan");

        nh.param<std::string>("virtual_scan_frame", virtual_scan_frame, "virtual_scan");

        subLidar = nh.subscribe<sensor_msgs::PointCloud2>(indexed_points_topic, /*2*/ 20, &Node::pointHandler, this);
        subPath = nh.subscribe<nav_msgs::Path>(poses_topic, 2, &Node::posesHandler, this);
        pubLaserScan = nh.advertise<sensor_msgs::LaserScan>(virtual_scan_topic, 2);
        pubObs = nh.advertise<sensor_msgs::PointCloud2>("/obstacles", 2);
        drawTimer = nh.createTimer(ros::Duration(1), &Node::drawHandler, this);
        drawTimer.start();
    }

    ~Node()
    {
        if (save_path != "")
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

#ifndef USE_SUBMAP
        size_t submap_index = std::stoi(msg->header.frame_id);
#else
        size_t splitter = msg->header.frame_id.find(':');
        size_t submap_index = std::stoi(msg->header.frame_id.substr(0, splitter));
        size_t localmap_index = std::stoi(msg->header.frame_id.substr(splitter + 1));
#endif
        auto detect_info = ground_detector->process(cloud);
        auto submap_info = std::make_shared<LocalmapInfo>();
        submap_info->detect = detect_info;
        submap_info->has_old_pose = false;
        submap_info->inited = false;
#ifndef USE_SUBMAP
        map_builder->submaps.emplace(submap_index, submap_info);
#else
        map_builder->addLocalmap({submap_index, (int64_t)localmap_index}, submap_info);
#endif

        if (pubLaserScan.getNumSubscribers() > 0)
            publishScan(detect_info->map_info, msg->header);
        return;
    }

    size_t currentSubmap = 0;
    std::set<size_t> tryFreeze;

    void posesHandler(const nav_msgs::PathConstPtr &msg)
    {
        for (const auto &pose : msg->poses)
        {
#ifndef USE_SUBMAP
            size_t submap_index = std::stoi(pose.header.frame_id);
            if (map_builder->submaps.find(submap_index) == map_builder->submaps.end())
                continue;
#else
            size_t splitter = pose.header.frame_id.find(':');
            if (splitter == std::string::npos)
            {
                size_t submap_index = std::stoi(pose.header.frame_id);
                if (!map_builder->hasSubmap(submap_index))
                    continue;
                auto &translation = pose.pose.position;
                auto &rotation = pose.pose.orientation;
                Eigen::Isometry3d pose_eigen = Eigen::Isometry3d::Identity();
                pose_eigen.translate(Eigen::Vector3d{translation.x, translation.y, translation.z});
                pose_eigen.rotate(Eigen::Quaterniond(rotation.w, rotation.x, rotation.y, rotation.z));
                {
                    // check stage
                    Eigen::Matrix4d oldMat = map_builder->submapPose(submap_index).cast<double>();
                    Eigen::Isometry3d oldIso = Eigen::Isometry3d::Identity();
                    oldIso.translate(oldMat.block<3,1>(0,3));
                    oldIso.rotate(oldMat.block<3,3>(0,0));
                    Eigen::Isometry3d delta = oldIso.inverse() * pose_eigen;
                    Eigen::Quaterniond q(delta.rotation());
                    float x = delta.translation().cwiseAbs().x();
                    float y = delta.translation().cwiseAbs().y();
                    float z = delta.translation().cwiseAbs().z();
                    float w = std::acos(q.w());
                    if (!(x > 5e-2 || y > 5e-2 || z > 5e-2 || w > 0.143))
                    {
                        continue;
                    }
                }
                // std::cout << "old\n" << map_builder->submapPose(submap_index).matrix() << std::endl;
                // std::cout << "new\n" << pose_eigen.matrix() << std::endl;
                map_builder->submapOldPose(submap_index) = map_builder->submapPose(submap_index);
                map_builder->submapPose(submap_index) = pose_eigen.matrix().cast<float>();
                need_update_submaps.push_back({submap_index, -1});
                continue;
            }
            size_t submap_index = std::stoi(pose.header.frame_id.substr(0, splitter));
            size_t localmap_index = std::stoi(pose.header.frame_id.substr(splitter + 1));
            const auto &current = map_builder->getLocalmap({submap_index, localmap_index});
            if (current == nullptr)
                continue;

            if (submap_index > currentSubmap)
            {
                tryFreeze.insert(currentSubmap);
                currentSubmap = submap_index;
            }
            else if (submap_index < currentSubmap && tryFreeze.find(submap_index) == tryFreeze.end())
                continue;
#endif
            auto &translation = pose.pose.position;
            auto &rotation = pose.pose.orientation;
            Eigen::Isometry3d pose_eigen = Eigen::Isometry3d::Identity();
            pose_eigen.translate(Eigen::Vector3d{translation.x, translation.y, translation.z});
            pose_eigen.rotate(Eigen::Quaterniond(rotation.w, rotation.x, rotation.y, rotation.z));

#ifndef USE_SUBMAP
            const auto &current = map_builder->submaps.at(submap_index);
#else
            if (map_builder->submapPose(submap_index) == Eigen::Matrix4f::Zero())
            {
                // std::cout << "init\n" << pose_eigen.matrix() << std::endl;
                map_builder->submapPose(submap_index) = pose_eigen.matrix().cast<float>();
                map_builder->submapOldPose(submap_index) = map_builder->submapPose(submap_index);
                pose_eigen = Eigen::Isometry3d::Identity();
            }
            else
            {
                Eigen::Matrix4d pose = map_builder->submapPose(submap_index).cast<double>().inverse() * pose_eigen.matrix();
                pose_eigen = Eigen::Isometry3d::Identity();
                pose_eigen.translate(pose.block<3, 1>(0, 3));
                pose_eigen.rotate(pose.block<3, 3>(0, 0));
            }
#endif
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
#ifndef USE_SUBMAP
                    need_update_submaps.push_back(submap_index);
#else
                    need_update_submaps.push_back({submap_index, (int64_t)localmap_index});
#endif
                }
            }
            else
            {
#ifndef USE_SUBMAP
                need_update_submaps.push_back(submap_index);
#else
                need_update_submaps.push_back({submap_index, (int64_t)localmap_index});
#endif
            }

            current->pose = pose_eigen;
            current->inited = true;
        }
    }

    void drawHandler(const ros::TimerEvent &)
    {
        auto ret = map_builder->draw_multi<8, 8>(need_update_submaps);
        std::static_pointer_cast<ChunkMapService>(chunk_map)->update(ret);
        if (pubObs.getNumSubscribers() > 0)
            publishOBScloud();

        for (const auto &it : tryFreeze)
        {
            map_builder->freezeSubmap(it);
        }
        tryFreeze.clear();
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

    void publishOBScloud()
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (auto &k : chunk_map->keys())
        {
            const auto &chunk = chunk_map->at(k);
            // for (auto &layer : chunk.second.getLayers())
            for (auto &layer : chunk.getLayers())
                for (int r = 0; r < layer.obstacle_info.rows; r++)
                    for (int c = 0; c < layer.obstacle_info.cols; c++)
                    {
                        uint64_t value = ObstacleTypeDesc_at(chunk_map->obstacle_segments, chunk_map->obstacle_bits, layer.obstacle_info, r, c);
                        for (int b = 0; b < chunk_map->obstacle_segments; b++)
                            if ((value & (1 << b)) != 0)
                            {
                                pcl::PointXYZ p;
                                p.x = c * chunk_map->resolution() + k.x * chunk_map->chunkBase();
                                p.y = r * chunk_map->resolution() + k.y * chunk_map->chunkBase();
                                p.z = b + layer.elevation(r, c);
                                cloud->push_back(p);
                            }
                    }
        }
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(*cloud, msg);
        msg.header.frame_id = "map";
        msg.header.stamp = ros::Time::now();
        pubObs.publish(msg);
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
