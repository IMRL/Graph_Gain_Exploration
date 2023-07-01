#ifndef _CHUNK_MAP_ROS_CHUNK_MAP_SERVICE_H_
#define _CHUNK_MAP_ROS_CHUNK_MAP_SERVICE_H_

#include <ros/ros.h>
#include <chunkmap_msgs/GetChunkMapInfo.h>
#include <chunkmap_msgs/GetChunkData.h>
#include <chunkmap_msgs/GetChunkVizData.h>
#include <chunkmap_msgs/GetKeyFrameInfo.h>
#include <chunkmap_msgs/GetKeyFrameData.h>
#include "../chunkmap/chunk_map.h"

class ChunkMapService : public ChunkMap
{
public:
    ChunkMapService(const Config &config, ros::NodeHandle &nh);
    virtual ~ChunkMapService() {}

    void update(const std::set<Index> &update_set);

    bool getInfoCallback(chunkmap_msgs::GetChunkMapInfo::Request &req, chunkmap_msgs::GetChunkMapInfo::Response &res);
    bool getChunkDataCallback(chunkmap_msgs::GetChunkData::Request &req, chunkmap_msgs::GetChunkData::Response &res);
    bool getChunkVizDataCallback(chunkmap_msgs::GetChunkVizData::Request &req, chunkmap_msgs::GetChunkVizData::Response &res);

    bool getKeyFrameInfoCallback(chunkmap_msgs::GetKeyFrameInfo::Request &req, chunkmap_msgs::GetKeyFrameInfo::Response &res);
    bool getKeyFrameDataCallback(chunkmap_msgs::GetKeyFrameData::Request &req, chunkmap_msgs::GetKeyFrameData::Response &res);

private:
    std::string frame_id_;

    ros::Publisher pubUpdate;
    ros::ServiceServer mapInfoService;
    ros::ServiceServer chunkDataService;
    ros::ServiceServer chunkVizDataService;
    ros::ServiceServer keyFrameInfoService;
    ros::ServiceServer keyFrameDataService;
};

#endif
