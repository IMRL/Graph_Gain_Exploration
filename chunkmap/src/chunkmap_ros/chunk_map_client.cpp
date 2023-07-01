#include <chunkmap_ros/chunk_map_client.h>
#include <chunkmap_msgs/GetChunkMapInfo.h>
#include <chunkmap_msgs/GetChunkData.h>
#include <chunkmap_msgs/GetKeyFrameInfo.h>
#include <chunkmap_msgs/GetKeyFrameData.h>

ChunkMapClient::ChunkMapClient(ros::NodeHandle &nh, const std::string &base_topic, Fn fn)
    : ChunkMap({}), fn(fn)
{

    chunkmap_msgs::GetChunkMapInfo mapInfo;
    chunkDataService = nh.serviceClient<chunkmap_msgs::GetChunkData>(base_topic + "/get_chunk_data");
    if (ros::service::call(base_topic + "/get_info", mapInfo))
    {
        config_.resolution = mapInfo.response.ChunkMapInfo.resolution;
        chunk_size = mapInfo.response.ChunkMapInfo.chunk_size;
        config_.chunk_base = chunk_size * config_.resolution;
        obstacle_segments = ObstacleFeature::Segments(mapInfo.response.ChunkMapInfo.obstacle_bits & 0xffff);
        obstacle_bits = ObstacleFeature::Bits(mapInfo.response.ChunkMapInfo.obstacle_bits >> 16);
        // init_map();
        std::vector<Index> index_list;
        std::set<Index> update_set;
        for (const auto &index : mapInfo.response.ChunkMapInfo.chunk_index)
        {
            index_list.push_back({index.x, index.y});
            update_set.insert({index.x, index.y});
        }
        loadChunks(index_list);
        // fn(update_set);
        init_set = update_set;
    }
    subUpdate = nh.subscribe<chunkmap_msgs::UpdateList>(base_topic + "/update_list", 10, &ChunkMapClient::updateListHandler, this);
    keyFrameInfoService = nh.serviceClient<chunkmap_msgs::GetKeyFrameInfo>(base_topic + "/get_key_frame_info");
    keyFrameDataService = nh.serviceClient<chunkmap_msgs::GetKeyFrameData>(base_topic + "/get_key_frame_data");
}

void ChunkMapClient::updateListHandler(const chunkmap_msgs::UpdateList::ConstPtr &msg)
{
    std::vector<ChunkMap::Index> index_list;
    std::set<ChunkMap::Index> update_set;
    for (const auto &index : msg->chunks)
    {
        index_list.push_back({index.x, index.y});
        update_set.insert({index.x, index.y});
    }
    std::lock_guard<std::mutex> guard(mtx);
    loadChunks(index_list);
    fn(update_set);
}

void ChunkMapClient::loadChunks(const std::vector<ChunkMap::Index> &index_list)
{
    if (chunk_size == 0)
        return;

    chunkmap_msgs::GetChunkData msg;
    for (const auto &index : index_list)
    {
        chunkmap_msgs::ChunkIndex req_index;
        req_index.x = index.x;
        req_index.y = index.y;
        msg.request.index.push_back(req_index);
    }
    if (chunkDataService.call(msg))
    {
        for (int i = 0; i < index_list.size(); i++)
        {
            if (msg.response.has_chunk[i])
            {
                const auto &index = index_list[i];
                const auto &data = msg.response.data[i];
                auto &item = (*this)[index];
                std::vector<ChunkLayer> layers;
                for (const auto &layer_data : data.layers)
                {
                    auto layer = item.createLayer();
                    cv::Mat1b(layer_data.observe, false).reshape(1, layer.observe.rows).copyTo(layer.observe);
                    cv::Mat1f(layer_data.occupancy, false).reshape(1, layer.occupancy.rows).copyTo(layer.occupancy);
                    cv::Mat1f(layer_data.elevation, false).reshape(1, layer.elevation.rows).copyTo(layer.elevation);
                    cv::Mat1f(layer_data.elevation_sigma, false).reshape(1, layer.elevation_sigma.rows).copyTo(layer.elevation_sigma);
                    memcpy((void *)layer.obstacle_info.data, (void *)layer_data.obstacle_info.data(), layer_data.obstacle_info.size());
                    layers.push_back(layer);
                }
                item.setLayers(layers);
            }
        }
    }
}

void ChunkMapClient::refreshKeyFrame()
{
    std::vector<int64_t> need_load;
    chunkmap_msgs::GetKeyFrameInfo msg;
    if (keyFrameInfoService.call(msg))
    {
        for (const auto &item : msg.response.key_frames)
        {
            if (key_frame_map.find(item.index) == key_frame_map.end())
            {
                key_frames.push_back({});
                key_frame_map[item.index] = key_frames.size() - 1;
                need_load.push_back(item.index);
            }
            Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
            pose(0, 0) = item.pose[0];
            pose(0, 1) = item.pose[1];
            pose(0, 2) = item.pose[2];
            pose(0, 3) = item.pose[3];
            pose(1, 0) = item.pose[4];
            pose(1, 1) = item.pose[5];
            pose(1, 2) = item.pose[6];
            pose(1, 3) = item.pose[7];
            pose(2, 0) = item.pose[8];
            pose(2, 1) = item.pose[9];
            pose(2, 2) = item.pose[10];
            pose(2, 3) = item.pose[11];
            key_frames[key_frame_map[item.index]].pose = pose;
        }
    }
    if (need_load.size() > 0)
    {
        chunkmap_msgs::GetKeyFrameData msg;
        msg.request.index.swap(need_load);
        if (keyFrameDataService.call(msg))
        {
            for (int i = 0; i < msg.response.has_frame.size(); i++)
            {
                if (msg.response.has_frame[i])
                {
                    auto &frame = key_frames[key_frame_map[msg.request.index[i]]];
                    if (desc_type == DescType::uncompressed)
                    {
                        LidarIris::FeatureDesc desc;
                        desc.img = cv::imdecode(msg.response.data[i].img, cv::IMREAD_UNCHANGED);
                        desc.T = cv::imdecode(msg.response.data[i].T, cv::IMREAD_UNCHANGED);
                        desc.M = cv::imdecode(msg.response.data[i].M, cv::IMREAD_UNCHANGED);
                        frame.feature = desc;
                    }
                    else
                    {
                        CompressedDesc desc;
                        desc.img = msg.response.data[i].img;
                        desc.T = msg.response.data[i].T;
                        desc.M = msg.response.data[i].M;
                        frame.feature = desc;
                    }
                }
            }
        }
    }
}
