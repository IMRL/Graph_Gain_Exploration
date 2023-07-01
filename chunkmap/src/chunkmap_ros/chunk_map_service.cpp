#include <chunkmap_ros/chunk_map_service.h>
#include <chunkmap_msgs/UpdateList.h>

ChunkMapService::ChunkMapService(const Config &config, ros::NodeHandle &nh)
    : ChunkMap(config)
{
    std::string base_topic;
    nh.param<std::string>("/chunkmap_service/topic", base_topic, "/chunkmap");
    nh.param<std::string>("/chunkmap_service/frame", frame_id_, "global_map");
    ROS_INFO_STREAM("chunkmap service at: " << base_topic);
    pubUpdate = nh.advertise<chunkmap_msgs::UpdateList>(base_topic + "/update_list", 10);
    mapInfoService = nh.advertiseService(base_topic + "/get_info", &ChunkMapService::getInfoCallback, this);
    chunkDataService = nh.advertiseService(base_topic + "/get_chunk_data", &ChunkMapService::getChunkDataCallback, this);
    chunkVizDataService = nh.advertiseService(base_topic + "/get_chunk_viz_data", &ChunkMapService::getChunkVizDataCallback, this);
    keyFrameInfoService = nh.advertiseService(base_topic + "/get_key_frame_info", &ChunkMapService::getKeyFrameInfoCallback, this);
    keyFrameDataService = nh.advertiseService(base_topic + "/get_key_frame_data", &ChunkMapService::getKeyFrameDataCallback, this);
}

void ChunkMapService::update(const std::set<Index> &update_set)
{
    chunkmap_msgs::UpdateList msg;
    msg.header.frame_id = frame_id_;
    msg.header.stamp = ros::Time::now();
    for (const auto &it : update_set)
    {
        chunkmap_msgs::ChunkIndex index;
        index.x = it.x;
        index.y = it.y;
        msg.chunks.push_back(index);
    }
    pubUpdate.publish(msg);
}

bool ChunkMapService::getInfoCallback(chunkmap_msgs::GetChunkMapInfo::Request &req, chunkmap_msgs::GetChunkMapInfo::Response &res)
{
    res.ChunkMapInfo.resolution = config_.resolution;
    res.ChunkMapInfo.chunk_size = chunk_size;
    res.ChunkMapInfo.obstacle_bits = ObstacleTypeDesc_serial_id(obstacle_segments, obstacle_bits);
    for (const auto &it : chunks)
    {
        chunkmap_msgs::ChunkIndex index;
        index.x = it.first.x;
        index.y = it.first.y;
        res.ChunkMapInfo.chunk_index.push_back(index);
    }
    return true;
}

bool ChunkMapService::getChunkDataCallback(chunkmap_msgs::GetChunkData::Request &req, chunkmap_msgs::GetChunkData::Response &res)
{
    for (const auto &req_index : req.index)
    {
        Index index{req_index.x, req_index.y};
        auto chunk = chunks.find(index);
        if (chunk == chunks.end())
        {
            res.has_chunk.push_back(false);
            res.data.push_back({});
            continue;
        }
        else
        {
            res.has_chunk.push_back(true);
            const auto &layers = chunk->second.getLayers();
            chunkmap_msgs::ChunkData chunk_msg;
            for (const auto &layer : layers)
            {
                chunkmap_msgs::ChunkLayer layer_msg;
                layer_msg.observe.reserve(layer.observe.total() * layer.observe.channels());
                layer_msg.observe.assign((uint8_t *)layer.observe.data, (uint8_t *)layer.observe.data + layer.observe.total() * layer.observe.channels());
                layer_msg.occupancy.reserve(layer.occupancy.total() * layer.occupancy.channels());
                layer_msg.occupancy.assign((float *)layer.occupancy.data, (float *)layer.occupancy.data + layer.occupancy.total() * layer.occupancy.channels());
                layer_msg.elevation.reserve(layer.elevation.total() * layer.elevation.channels());
                layer_msg.elevation.assign((float *)layer.elevation.data, (float *)layer.elevation.data + layer.elevation.total() * layer.elevation.channels());
                layer_msg.elevation_sigma.reserve(layer.elevation_sigma.total() * layer.elevation_sigma.channels());
                layer_msg.elevation_sigma.assign((float *)layer.elevation_sigma.data, (float *)layer.elevation_sigma.data + layer.elevation_sigma.total() * layer.elevation_sigma.channels());
                layer_msg.obstacle_info.reserve(layer.obstacle_info.total() * layer.obstacle_info.elemSize());
                layer_msg.obstacle_info.assign((uint8_t *)layer.obstacle_info.data, (uint8_t *)layer.obstacle_info.data + layer.obstacle_info.total() * layer.obstacle_info.elemSize());
                chunk_msg.layers.push_back(layer_msg);
            }
            res.data.push_back(chunk_msg);
        }
    }
    return true;
}

bool ChunkMapService::getChunkVizDataCallback(chunkmap_msgs::GetChunkVizData::Request &req, chunkmap_msgs::GetChunkVizData::Response &res)
{
    for (const auto &req_index : req.index)
    {
        Index index{req_index.x, req_index.y};
        auto chunk = chunks.find(index);
        if (chunk == chunks.end())
        {
            res.has_chunk.push_back(false);
            res.data.push_back({});
            continue;
        }
        else
        {
            res.has_chunk.push_back(true);
            const auto &layers = chunk->second.getLayers();
            chunkmap_msgs::ChunkVizData chunk_msg;
            for (const auto &layer : layers)
            {
                const auto &observe = layer.observe;
                float elevation_alpha, elevation_beta;
                cv::Mat elevation = layer.elevation.clone();
                {
                    double mini, maxi;
                    cv::minMaxIdx(elevation, &mini, &maxi, nullptr, nullptr, observe);
                    elevation_alpha = maxi - mini;
                    elevation_beta = mini;
                    if (elevation_beta != 0)
                    {
                        elevation = (elevation - elevation_beta) / elevation_alpha * 255.0;
                        elevation.convertTo(elevation, CV_8U);
                    }
                }
                cv::Mat1b occupancy;
                layer.occupancy.convertTo(occupancy, CV_8U);
                cv::Mat finally;
                cv::merge(std::vector<cv::Mat1b>{occupancy, observe}, finally);
                chunkmap_msgs::ChunkVizLayer layer_msg;
                layer_msg.elevation.reserve(elevation.total() * elevation.channels());
                layer_msg.elevation.assign(elevation.data, elevation.data + elevation.total() * elevation.channels());
                layer_msg.occupancy.reserve(finally.total() * finally.channels());
                layer_msg.occupancy.assign(finally.data, finally.data + finally.total() * finally.channels());
                layer_msg.elevation_alpha = elevation_alpha;
                layer_msg.elevation_beta = elevation_beta;
                chunk_msg.layers.push_back(layer_msg);
            }
            res.data.push_back(chunk_msg);
        }
    }
    return true;
}

bool ChunkMapService::getKeyFrameInfoCallback(chunkmap_msgs::GetKeyFrameInfo::Request &req, chunkmap_msgs::GetKeyFrameInfo::Response &res)
{
    for (size_t i = 0; i < key_frames.size(); i++)
    {
        const auto &key_frame = key_frames[i];
        chunkmap_msgs::KeyFrameInfo info;
        info.index = i;
        info.pose.push_back(key_frame.pose(0,0));
        info.pose.push_back(key_frame.pose(0,1));
        info.pose.push_back(key_frame.pose(0,2));
        info.pose.push_back(key_frame.pose(0,3));
        info.pose.push_back(key_frame.pose(1,0));
        info.pose.push_back(key_frame.pose(1,1));
        info.pose.push_back(key_frame.pose(1,2));
        info.pose.push_back(key_frame.pose(1,3));
        info.pose.push_back(key_frame.pose(2,0));
        info.pose.push_back(key_frame.pose(2,1));
        info.pose.push_back(key_frame.pose(2,2));
        info.pose.push_back(key_frame.pose(2,3));
        res.key_frames.push_back(info);
    }
    return true;
}

bool ChunkMapService::getKeyFrameDataCallback(chunkmap_msgs::GetKeyFrameData::Request &req, chunkmap_msgs::GetKeyFrameData::Response &res)
{
    for (const auto &req_index : req.index)
    {
        if (req_index >= key_frames.size())
        {
            res.has_frame.push_back(false);
            res.data.push_back({});
        }
        else
        {
            res.has_frame.push_back(true);
            const auto &frame = key_frames[req_index];
            chunkmap_msgs::KeyFrameData frame_msg;
            if(desc_type == DescType::uncompressed)
            {
                cv::imencode(".png", std::get<0>(frame.feature).img, frame_msg.img);
                cv::imencode(".png", std::get<0>(frame.feature).T, frame_msg.T);
                cv::imencode(".png", std::get<0>(frame.feature).M, frame_msg.M);
            }
            else
            {
                frame_msg.img = std::get<1>(frame.feature).img;
                frame_msg.T = std::get<1>(frame.feature).T;
                frame_msg.M = std::get<1>(frame.feature).M;
            }
            res.data.push_back(frame_msg);
        }
    }
    return true;
}

