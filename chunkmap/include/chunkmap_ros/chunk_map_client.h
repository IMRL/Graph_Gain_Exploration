#ifndef _CHUNK_MAP_ROS_CHUNK_MAP_CLIENT_H_
#define _CHUNK_MAP_ROS_CHUNK_MAP_CLIENT_H_

#include <functional>
#include <ros/ros.h>
#include <chunkmap_msgs/UpdateList.h>
#include "../chunkmap/chunk_map.h"
#include <mutex>

class ChunkMapClient : public ChunkMap
{
public:
    using Fn = std::function<void(const std::set<Index> &)>;

    ChunkMapClient(ros::NodeHandle &nh, const std::string &base_topic, Fn fn);
    virtual ~ChunkMapClient() {}

    void updateListHandler(const chunkmap_msgs::UpdateList::ConstPtr &msg);

    void refreshKeyFrame();


    // init to process already published
    void init() { fn(init_set); }

protected:
    std::mutex mtx;

private:
    void loadChunks(const std::vector<ChunkMap::Index> &index_list);

    Fn fn;

    cv::Size key_frame_img_size;
    cv::Size key_frame_T_size;
    cv::Size key_frame_M_size;
    std::map<int64_t, size_t> key_frame_map;

    ros::Subscriber subUpdate;
    ros::ServiceClient chunkDataService;
    ros::ServiceClient keyFrameInfoService;
    ros::ServiceClient keyFrameDataService;

    std::set<Index> init_set;
};

#endif
